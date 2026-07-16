"""
前插网络预训练 — SpinPretrainNN
任务: 分子图 → GINEConv → 每个原子的 Mulliken 自旋密度

预训练完成后:
  冻结主干 → 原子级嵌入注入 BDEGNNv3 → BDE 精调
"""
import torch
import torch.nn as nn
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


# ======== 1. 模型 ========
class SpinPretrainNN(nn.Module):
    """
    前插网络: 分子图 → 每原子自旋密度 + 电荷 + 键级

    三个预测头（多任务学习）:
      - spin_head:   每个原子 Mulliken 自旋密度
      - charge_head: 每个原子 NPA 电荷
      - bond_head:   每个原子对 贡献的键级（简化: 每原子键级之和）
    """

    def __init__(self, node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.3):
        super().__init__()
        self.hidden = hidden

        # ---- GINEConv 主干 (与 BDEGNNv2 共享架构) ----
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = nn.Dropout(dropout)

        # ---- 多任务预测头 ----
        self.spin_head = nn.Sequential(         # 自旋密度 (1)
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.charge_head = nn.Sequential(       # Mulliken电荷 (1)
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.graph_head = nn.Sequential(        # 轨道 + 偶极 (5)
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 5),  # alpha_homo,lumo,beta_homo,lumo,dipole_mag
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new).relu()
            h_new = self.dropout(h_new)
            h = h + h_new

        sp = self.spin_head(h).squeeze(-1)           # [N]
        ch = self.charge_head(h).squeeze(-1)          # [N]
        g = global_mean_pool(h, batch)
        gr = self.graph_head(g)                       # [B, 5]

        return sp, ch, gr, h


# ======== 2. 数据准备: SMILES → PyG Data (仿 QM9star 字段) ========
def smiles_to_spin_data(smiles, spin_densities=None, charges=None):
    """
    构建分子图，可选地附带 DFT 标签。

    Args:
        smiles: 分子 SMILES
        spin_densities: list[float], 每个原子的 Mulliken 自旋密度 (可选)
        charges: list[float], 每个原子的 NPA 电荷 (可选)

    Returns:
        PyG Data 对象
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    # ---- 节点特征 (10维, 与 BDEGNNv2 训练数据一致) ----
    node_feats = []
    for atom in mol.GetAtoms():
        f = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.GetFormalCharge()),
            float(atom.GetHybridization() == Chem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            float(atom.IsInRing()),
            0.0,  # atom_label (预训练时无目标键, 填0)
        ]
        node_feats.append(f)

    x = torch.tensor(node_feats, dtype=torch.float32)

    # ---- 边特征 (4维) ----
    edge_idx = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_idx[0].extend([i, j])
        edge_idx[1].extend([j, i])
        bt = bond.GetBondType()
        feats = [
            float(bt == Chem.BondType.SINGLE),
            float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE),
            float(bt == Chem.BondType.AROMATIC),
        ]
        edge_attr.extend([feats, feats])

    edge_index = torch.tensor(edge_idx, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    # ---- 标签 ----
    y_spin = torch.zeros(n_atoms, dtype=torch.float32)
    y_charge = torch.zeros(n_atoms, dtype=torch.float32)
    has_labels = False
    if spin_densities is not None:
        y_spin = torch.tensor(spin_densities, dtype=torch.float32)
        has_labels = True
    if charges is not None:
        y_charge = torch.tensor(charges, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t,
                y_spin=y_spin, y_charge=y_charge, has_labels=has_labels)


# ======== 3. 训练 ========
def pretrain_spin(model, train_loader, val_loader, epochs=500, lr=0.001,
                  device='cpu', ckpt_path='spin_pretrain_best.pt'):
    """前插网络预训练: 预测自旋密度 + 电荷"""

    model = model.to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            spin_pred, charge_pred, _ = model(batch)
            loss = loss_fn(spin_pred, batch.y_spin) + 0.5 * loss_fn(charge_pred, batch.y_charge)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                spin_pred, charge_pred, _ = model(batch)
                val_loss += loss_fn(spin_pred, batch.y_spin).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': best_val}, ckpt_path)

        if epoch % 50 == 0 or epoch < 10:
            print(f"E{epoch:4d}  Train={train_loss:.4f}  Val={val_loss:.4f}  Best={best_val:.4f}")

    print(f"\nBest val loss: {best_val:.4f} → {ckpt_path}")
    return best_val


# ======== 4. 冻结 + 导出嵌入 ========
def freeze_and_export(model, ckpt_path='spin_pretrain_best.pt',
                      output_path='spin_pretrain_frozen.pt'):
    """加载预训练权重, 返回冻结的主干权重"""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])

    # 只导出 GNN 主干, 不导出预测头
    frozen = {}
    for k, v in model.state_dict().items():
        if not any(k.startswith(h) for h in ['spin_head', 'charge_head', 'graph_head']):
            frozen[k] = v

    torch.save({'backbone': frozen, 'hidden': model.hidden}, output_path)
    print(f"Frozen backbone saved → {output_path}")
    return frozen


if __name__ == '__main__':
    # 快速冒烟测试: 随机数据验证管线能跑通
    print("=" * 60)
    print("SpinPretrainNN 冒烟测试")
    print("=" * 60)

    # 构造假数据
    test_smiles = ["C" + "C" * i for i in range(20)]  # C, CC, CCC, ...
    data_list = []
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        n = mol.GetNumAtoms()
        d = smiles_to_spin_data(smi,
                                spin_densities=[0.0] * n,  # 假标签
                                charges=[0.0] * n)
        if d is not None:
            data_list.append(d)

    print(f"Fake data: {len(data_list)} molecules")

    # 模型初始化
    model = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=128, n_layers=3, dropout=0.3)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播测试
    d = data_list[0]
    spin_pred, charge_pred, embeddings = model(d)
    print(f"Input atoms={d.x.shape[0]}, Output spin={spin_pred.shape}, emb={embeddings.shape}")

    # 训练迭代测试 (1 epoch, 小数据)
    loader = DataLoader(data_list, batch_size=4, shuffle=True)
    pretrain_spin(model, loader, loader, epochs=3, lr=0.001, device='cpu')

    # 冻结导出测试
    freeze_and_export(model)
    print("\n管线验证通过! 等QM9star数据到了替换假数据即可训.")

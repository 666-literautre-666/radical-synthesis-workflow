"""
BDEGNNv4 — 冻结 SpinPretrainNN 注入 + BDE 精调
对比基准: BDEGNN v2 (随机初始化, MAE 1.46)
"""
import torch, torch.nn as nn, numpy as np, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from gnn_data_utils import load_gnn_data
from config import CONFIG
from spin_pretrain import SpinPretrainNN


# ======== BDEGNNv4: 冻结前插网络 + GNN 主干 ========
class BDEGNNv4(nn.Module):
    """
    冻结 SpinPretrainNN → 提取每原子嵌入(256维) → 拼入节点特征 → GINEConv → BDE
    """

    def __init__(self, frozen_backbone, node_dim=10, spin_emb_dim=256,
                 hidden=256, n_layers=4, dropout=0.3):
        super().__init__()
        self.hidden = hidden

        # ---- 冻结的前插网络 ----
        self.spin_backbone = frozen_backbone
        for p in self.spin_backbone.parameters():
            p.requires_grad = False  # 锁死

        # ---- BDE GNN 主干 ----
        total_node_dim = node_dim + spin_emb_dim  # 10 + 256 = 266
        self.input_proj = nn.Linear(total_node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(nn_mlp, edge_dim=4, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 冻结前插网络: 提取每个原子的自旋嵌入
        with torch.no_grad():
            _, _, spin_emb = self.spin_backbone(data)  # [N_atoms, 256]

        # 拼接原始节点特征 + 自旋嵌入
        x_aug = torch.cat([x, spin_emb], dim=1)  # [N, 10+256=266]

        # BDE GNN 主干
        h = self.input_proj(x_aug)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new).relu()
            h_new = self.dropout(h_new)
            h = h + h_new

        # 目标原子提取 (atom_label == 3.0)
        is_target = (x[:, -1] == 3.0)
        target_embs = h[is_target]
        target_embs = target_embs.view(-1, self.hidden * 2)

        return self.fc(target_embs)


# ======== 训练 ========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("BDEGNNv4 — Frozen Spin Backbone + BDE Fine-tuning")
    print(f"Device: {device}")
    print("=" * 60)

    # ======== 1. 加载冻结主干 ========
    print("\n[1/4] Loading frozen spin backbone...")
    # 先构建一个空 SpinPretrainNN, 加载 frozen 权重
    spin_model = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0)
    frozen_ckpt = torch.load('spin_pretrain_frozen.pt', map_location=device, weights_only=True)
    # frozen_ckpt 结构: {'backbone': {state_dict}, 'hidden': 256}
    # 需要把 backbone 里的权重映射回 SpinPretrainNN
    backbone_state = frozen_ckpt['backbone']
    # 去掉可能的 'module.' 前缀
    spin_model.load_state_dict(backbone_state, strict=False)
    spin_model = spin_model.to(device)
    spin_model.eval()
    print(f"  Frozen backbone loaded: {frozen_ckpt['hidden']} dim")

    # ======== 2. 加载 BDE 数据 ========
    print("\n[2/4] Loading BDE data...")
    nrows = CONFIG.get('nrows', 200000)  # 先用20万对比，和v2同条件
    data_list, bde_mean, bde_std = load_gnn_data(CONFIG['data_path'], nrows=nrows)
    print(f"  BDE mean={bde_mean:.1f} std={bde_std:.1f} | {len(data_list)} molecules")

    idx = list(range(len(data_list)))
    tr_idx, tmp = train_test_split(idx, test_size=0.2, random_state=42)
    va_idx, te_idx = train_test_split(tmp, test_size=0.5, random_state=42)

    bs = CONFIG['batch_size']
    tr_loader = DataLoader([data_list[i] for i in tr_idx], batch_size=bs, shuffle=True)
    va_loader = DataLoader([data_list[i] for i in va_idx], batch_size=bs)
    te_loader = DataLoader([data_list[i] for i in te_idx], batch_size=bs)
    print(f"  Train: {len(tr_idx)}  Val: {len(va_idx)}  Test: {len(te_idx)}")

    # ======== 3. 构建 BDEGNNv4 ========
    print("\n[3/4] Building BDEGNNv4...")
    model = BDEGNNv4(
        frozen_backbone=spin_model,
        node_dim=10,    # 原始节点特征
        spin_emb_dim=256,  # 冻结自旋嵌入
        hidden=CONFIG['gnn_hidden'],
        n_layers=CONFIG['gnn_layers'],
        dropout=CONFIG.get('dropout', 0.3),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable:,}  Frozen: {frozen:,}")

    # ======== 4. 训练 BDE 精调 ========
    print("\n[4/4] Training...")
    loss_fn = nn.MSELoss()
    # 只优化 BDE 部分（前插网络已冻结）
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['epochs'], eta_min=1e-6)

    best_va, best_ep, t0 = float('inf'), 0, time.time()

    for ep in range(CONFIG['epochs']):
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1, 1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                va_loss += loss_fn(model(batch), batch.y.view(-1, 1)).item()
        va_loss /= len(va_loader)
        sch.step()

        if va_loss < best_va:
            best_va, best_ep = va_loss, ep
            torch.save({'model': model.state_dict(), 'best_val': best_va, 'epoch': ep,
                        'bde_mean': bde_mean, 'bde_std': bde_std}, 'gnn_bde_v4_best.pt')

        if ep % 50 == 0 or ep < 5:
            print(f"E{ep:4d}  tr={tr_loss:.4f}  va={va_loss:.4f}  best={best_va:.4f}@{best_ep}  {((time.time()-t0)/60):.0f}min")

        if ep - best_ep > CONFIG['patience']:
            print(f"Early stop at {ep}"); break

    # ======== 5. 测试 ========
    print("\nTesting...")
    ckpt = torch.load('gnn_bde_v4_best.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model']); model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            p = model(batch).cpu().squeeze().tolist()
            preds.extend(p if isinstance(p, list) else [p])
            ys.extend(batch.y.cpu().tolist())

    preds = np.array(preds) * bde_std + bde_mean
    ys = np.array(ys) * bde_std + bde_mean
    mae = np.mean(np.abs(preds - ys))
    rmse = np.sqrt(np.mean((preds - ys)**2))
    r2 = np.corrcoef(preds, ys)[0,1]**2

    print(f"\n{'='*60}")
    print(f"BDEGNNv4 TEST RESULTS")
    print(f"  MAE:  {mae:.2f} kcal/mol")
    print(f"  RMSE: {rmse:.2f} kcal/mol")
    print(f"  R²:   {r2:.4f}")
    print(f"  BDE range: {ys.min():.0f}-{ys.max():.0f} kcal/mol")
    print(f"  vs v2:  MAE 1.46 (baseline)")
    print(f"{'='*60}")

    return mae, rmse, r2


if __name__ == '__main__':
    main()

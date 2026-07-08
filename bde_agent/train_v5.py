"""
BDEGNNv5 — 双通道物理特征注入 + 多任务正则化
  Channel A (压缩): SpinPretrainNN 256→64, 拼入化学特征 → GNN主体
  Channel B (残差): SpinPretrainNN 256→hidden, 注入GNN末层 → 信息不丢
  多任务: BDE + spin_density + charge 三个预测头联合训练
"""
import torch, torch.nn as nn, numpy as np, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from gnn_data_utils import load_gnn_data
from spin_pretrain import SpinPretrainNN


# ======== v5 模型: 双通道 + 多任务 ========
class BDEGNNv5(nn.Module):
    """
    改进点:
    1. 双通道注入: compact(256→64) + residual(256→hidden)
    2. 三头多任务: BDE + spin + charge
    3. 可学习残差门控: 模型自己决定用多少物理信息
    4. 物理特征每原子门控: token-wise gating
    """

    def __init__(self, frozen_spin, node_dim=10, edge_dim=4, hidden=256,
                 n_layers=4, dropout=0.3, spin_compact_dim=64):
        super().__init__()
        self.hidden = hidden
        self.spin = frozen_spin
        for p in self.spin.parameters():
            p.requires_grad = False

        # ---- Channel A: 压缩摘要 ----
        self.spin_compact = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, spin_compact_dim),
        )

        # ---- Channel B: 残差旁路 (含门控) ----
        self.spin_residual = nn.Linear(hidden, hidden)
        self.residual_gate = nn.Parameter(torch.tensor(0.0))  # 从0开始学

        # ---- 输入投影: chem 10 + spin_compact 64 = 74 → hidden ----
        self.input_proj = nn.Linear(node_dim + spin_compact_dim, hidden)

        # ---- GINEConv 主干 ----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = nn.Dropout(dropout)

        # ---- 多任务预测头 ----
        # 自旋密度 (节点级辅助任务)
        self.spin_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )
        # 电荷 (节点级辅助任务)
        self.charge_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )
        # BDE (边级主任务)
        self.bde_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        # 1. 冻结前插网络 → 物理特征
        with torch.no_grad():
            spin_pseudo, charge_pseudo, emb = self.spin(data)

        # 2. Channel A: 压缩摘要 + 拼接化学特征
        spin_c = self.spin_compact(emb)                     # [N, 64]
        x_aug = torch.cat([data.x, spin_c], dim=-1)         # [N, 10+64=74]

        # 3. GNN 主干
        h = self.input_proj(x_aug)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index, data.edge_attr)
            h = h + self.dropout(norm(h_new).relu())

        # 4. Channel B: 残差注入 (门控)
        spin_r = self.spin_residual(emb)                    # [N, hidden]
        gate = torch.sigmoid(self.residual_gate)            # 标量, 0→1
        h = h + gate * spin_r

        # 5. 多任务预测
        spin_pred = self.spin_head(h).squeeze(-1)           # [N]
        charge_pred = self.charge_head(h).squeeze(-1)       # [N]

        # 6. BDE (目标原子池化)
        is_target = (data.x[:, -1] == 3.0)
        t_emb = h[is_target].view(-1, self.hidden * 2)      # [B, 512]
        bde_pred = self.bde_head(t_emb)                      # [B, 1]

        return bde_pred, spin_pred, charge_pred, spin_pseudo, charge_pseudo


# ======== 训练配置 ========
CFG = {
    'data_path': 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz',
    'nrows': 800000,
    'hidden': 256, 'n_layers': 4, 'dropout': 0.3,
    'batch_size': 256, 'epochs': 400, 'lr': 0.001, 'weight_decay': 0.0001,
    'patience': 120,
    # 多任务 loss 权重
    'lambda_spin': 0.3,    # 自旋密度辅助 loss 权重
    'lambda_charge': 0.1,  # 电荷辅助 loss 权重
}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"BDEGNNv5: Dual-channel + Multi-task | Device: {dev}")
    print("=" * 60)

    # 1. 冻结前插网络
    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(dev)
    spin.load_state_dict(torch.load('spin_pretrain_frozen.pt', weights_only=True)['backbone'], strict=False)
    spin.eval()
    n_spin = sum(p.numel() for p in spin.parameters())
    print(f"SpinPretrainNN: {n_spin:,} params (frozen)")

    # 2. BDE 数据
    data_list, bde_m, bde_s = load_gnn_data(CFG['data_path'], nrows=CFG['nrows'])
    print(f"BDE data: {len(data_list)} molecules, mean={bde_m:.1f}, std={bde_s:.1f}")

    idx = list(range(len(data_list)))
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    tr_ld = DataLoader([data_list[i] for i in tr], batch_size=CFG['batch_size'], shuffle=True)
    va_ld = DataLoader([data_list[i] for i in va], batch_size=CFG['batch_size'])
    te_ld = DataLoader([data_list[i] for i in te], batch_size=CFG['batch_size'])
    print(f"Train: {len(tr)}  Val: {len(va)}  Test: {len(te)}")

    # 3. 模型
    model = BDEGNNv5(spin, node_dim=10, hidden=CFG['hidden'], n_layers=CFG['n_layers'],
                     dropout=CFG['dropout']).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Model: {trainable:,} trainable + {frozen:,} frozen")
    print(f"  residual_gate init = {model.residual_gate.item():.4f} (sigmoid -> {torch.sigmoid(model.residual_gate).item():.4f})")

    # 4. 训练
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-6)
    best_va, best_ep, t0 = float('inf'), 0, time.time()
    progress_path = r'C:\Users\xushaobo\Desktop\v5_progress.txt'

    for ep in range(CFG['epochs']):
        # ---- Train ----
        model.train()
        tr_loss_bde, tr_loss_spin, tr_loss_charge = 0.0, 0.0, 0.0
        for b in tr_ld:
            b = b.to(dev)
            bde_pred, spin_pred, charge_pred, spin_pseudo, charge_pseudo = model(b)

            loss_bde = mse(bde_pred, b.y.view(-1, 1))
            loss_spin = mse(spin_pred, spin_pseudo)
            loss_charge = mse(charge_pred, charge_pseudo)
            loss = loss_bde + CFG['lambda_spin'] * loss_spin + CFG['lambda_charge'] * loss_charge

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss_bde += loss_bde.item()
            tr_loss_spin += loss_spin.item()
            tr_loss_charge += loss_charge.item()

        tr_loss_bde /= len(tr_ld)
        tr_loss_spin /= len(tr_ld)
        tr_loss_charge /= len(tr_ld)

        # ---- Val ----
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in va_ld:
                b = b.to(dev)
                bde_pred, _, _, _, _ = model(b)
                va_loss += mse(bde_pred, b.y.view(-1, 1)).item()
        va_loss /= len(va_ld)
        sch.step()

        # ---- 保存 ----
        if va_loss < best_va:
            best_va, best_ep = va_loss, ep
            torch.save({
                'model': model.state_dict(), 'epoch': ep, 'best_val': best_va,
                'bde_mean': bde_m, 'bde_std': bde_s,
                'residual_gate': model.residual_gate.item(),
            }, 'gnn_bde_v5_best.pt')

        if ep % 25 == 0 or ep < 5:
            elapsed = time.time() - t0
            gate_val = torch.sigmoid(model.residual_gate).item()
            msg = (f"E{ep:4d} BDE_tr={tr_loss_bde:.4f} sp_tr={tr_loss_spin:.4f} ch_tr={tr_loss_charge:.4f}  "
                   f"va={va_loss:.4f} best={best_va:.4f}@{best_ep}  gate={gate_val:.3f}  {elapsed/60:.0f}min")
            print(msg)
            with open(progress_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

        if ep - best_ep > CFG['patience']:
            print(f"Early stop at epoch {ep}")
            break

    # ======== 5. 测试 ========
    print("\n" + "=" * 60)
    print("Evaluating best model on test set...")
    ckpt = torch.load('gnn_bde_v5_best.pt', map_location=dev, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()

    preds, ys = [], []
    with torch.no_grad():
        for b in te_ld:
            b = b.to(dev)
            bde_pred, _, _, _, _ = model(b)
            preds.extend(bde_pred.cpu().squeeze().tolist())
            ys.extend(b.y.cpu().tolist())

    preds_arr = np.array([p * bde_s + bde_m for p in preds])
    y_arr = np.array([y * bde_s + bde_m for y in ys])
    mae = np.mean(np.abs(preds_arr - y_arr))
    rmse = np.sqrt(np.mean((preds_arr - y_arr) ** 2))
    r2 = np.corrcoef(preds_arr, y_arr)[0, 1] ** 2

    result = (
        f"\n{'='*60}\n"
        f"v5 TEST RESULTS\n"
        f"{'='*60}\n"
        f"MAE:           {mae:.2f} kcal/mol\n"
        f"RMSE:          {rmse:.2f} kcal/mol\n"
        f"R^2:           {r2:.4f}\n"
        f"Best epoch:    {ckpt['epoch']}\n"
        f"Best val_loss: {ckpt['best_val']:.6f}\n"
        f"Residual gate: {torch.sigmoid(torch.tensor(ckpt.get('residual_gate', 0))).item():.4f}\n"
        f"BDE range:     {y_arr.min():.0f}-{y_arr.max():.0f} kcal/mol\n"
    )
    print(result)
    with open('C:/Users/xushaobo/Desktop/v5_results.txt', 'w', encoding='utf-8') as f:
        f.write(result)

    print("Saved: gnn_bde_v5_best.pt")


if __name__ == '__main__':
    main()

"""
BDEGNNv4 = BDEGNNv2(10→256) + SpinPretrainNN(冻结)
节点特征 10+256=266 → input_proj → GINEConv×4 → fc → BDE
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

# ======== v4 模型 ========
class BDEGNNv4(nn.Module):
    def __init__(self, frozen_spin, node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.spin = frozen_spin
        for p in self.spin.parameters():
            p.requires_grad = False

        total_dim = node_dim + hidden  # 10 + 256 = 266
        self.input_proj = nn.Linear(total_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_mlp = nn.Sequential(nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,hidden))
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        with torch.no_grad():
            _, _, emb = self.spin(data)       # [N, 256]
        x = torch.cat([data.x, emb], dim=1)    # [N, 10+256=266]
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index, data.edge_attr)
            h = h + self.dropout(norm(h_new).relu())
        is_target = (data.x[:, -1] == 3.0)
        t_emb = h[is_target].view(-1, self.hidden*2)
        return self.fc(t_emb)


# ======== 训练配置 ========
CFG = {
    'data_path': 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz',
    'nrows': 800000,
    'hidden': 256, 'n_layers': 4, 'dropout': 0.3,
    'batch_size': 256, 'epochs': 300, 'lr': 0.001, 'weight_decay': 0.0001,
    'patience': 100,
}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"BDEGNNv4 training | Device: {dev}")
    print("=" * 60)

    # 1. 冻结前插网络
    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(dev)
    spin.load_state_dict(torch.load('spin_pretrain_frozen.pt', weights_only=True)['backbone'], strict=False)
    spin.eval()
    print(f"Spin loaded: {sum(p.numel() for p in spin.parameters()):,} params (frozen)")

    # 2. BDE数据
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
    model = BDEGNNv4(spin, node_dim=10, hidden=CFG['hidden'], n_layers=CFG['n_layers'],
                     dropout=CFG['dropout']).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Model: {trainable:,} trainable + {frozen:,} frozen")

    # 4. 训练
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-6)
    best_va, best_ep, t0 = float('inf'), 0, time.time()

    for ep in range(CFG['epochs']):
        model.train()
        tr_loss = 0.0
        for b in tr_ld:
            b = b.to(dev)
            pred = model(b)
            loss = loss_fn(pred, b.y.view(-1, 1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in va_ld:
                b = b.to(dev)
                va_loss += loss_fn(model(b), b.y.view(-1, 1)).item()
        va_loss /= len(va_ld)
        sch.step()

        if va_loss < best_va:
            best_va, best_ep = va_loss, ep
            torch.save({'model': model.state_dict(), 'epoch': ep, 'best_val': best_va,
                        'bde_mean': bde_m, 'bde_std': bde_s}, 'gnn_bde_v4_best.pt')

        if ep % 50 == 0 or ep < 5:
            print(f"E{ep:4d} tr={tr_loss:.4f} va={va_loss:.4f} best={best_va:.4f}@{best_ep}  {((time.time()-t0)/60):.0f}min")
        if ep - best_ep > CFG['patience']:
            print(f"Early stop at {ep}"); break

    # 5. 测试
    ckpt = torch.load('gnn_bde_v4_best.pt', map_location=dev, weights_only=True)
    model.load_state_dict(ckpt['model']); model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in te_ld:
            b = b.to(dev)
            preds.extend(model(b).cpu().squeeze().tolist())
            ys.extend(b.y.cpu().tolist())
    preds, ys = np.array(preds)*bde_s+bde_m, np.array(ys)*bde_s+bde_m
    mae = np.mean(np.abs(preds-ys))
    rmse = np.sqrt(np.mean((preds-ys)**2))
    r2 = np.corrcoef(preds, ys)[0,1]**2

    result = (
        f"\n{'='*60}\n"
        f"v4 TEST: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}\n"
        f"v2 baseline: MAE=1.46  RMSE=2.10  R²=0.9836\n"
        f"Improvement: MAE {'↓' if mae < 1.46 else '↑'}{abs(mae-1.46):.2f}  "
        f"R² {'↑' if r2 > 0.9836 else '↓'}\n"
        f"{'='*60}"
    )
    print(result)
    with open('C:/Users/xushaobo/Desktop/v4_results.txt', 'w') as f:
        f.write(result)


if __name__ == '__main__':
    main()

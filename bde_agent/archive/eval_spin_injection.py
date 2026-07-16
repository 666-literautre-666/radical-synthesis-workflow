"""
SpinPretrainNN 注入验证 — v2 全冻结, 只训适配器, 直接对比
"""
import torch, torch.nn as nn, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from gnn_inference import BDEGNNv2
from gnn_data_utils import load_gnn_data
from config import CONFIG
from spin_pretrain import SpinPretrainNN


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("Spin Injection Validation")
    print(f"Device: {dev}")
    print("=" * 60)

    # 1. 加载 SpinPretrainNN (冻结)
    print("\n[1] Frozen SpinPretrainNN...")
    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(dev)
    spin.load_state_dict(torch.load('spin_pretrain_frozen.pt', weights_only=True)['backbone'], strict=False)
    spin.eval()
    for p in spin.parameters(): p.requires_grad = False

    # 2. 加载 v2 (冻结)
    print("[2] Frozen BDEGNN v2...")
    v2 = torch.load('gnn_bde_best_v2.pt', weights_only=True)
    mean, std = v2['bde_mean'], v2['bde_std']
    bde = BDEGNNv2(node_dim=10, edge_dim=4, hidden=256, n_layers=4).to(dev)
    bde.load_state_dict(v2['model'])
    bde.eval()
    for p in bde.parameters(): p.requires_grad = False

    # 3. 适配器 266→10 (只训这个)
    print("[3] Adapter 266→10...")
    adapter = nn.Linear(265, 9).to(dev)  # 9维（去标签）+ 保留原始atom_label

    # 4. 数据
    print("[4] Loading BDE data (50k)...")
    data, _, _ = load_gnn_data(CONFIG['data_path'], nrows=50000)
    tr_idx, te_idx = train_test_split(list(range(len(data))), test_size=0.3, random_state=42)
    tr = DataLoader([data[i] for i in tr_idx], batch_size=256, shuffle=True)
    te = DataLoader([data[i] for i in te_idx], batch_size=256)
    print(f"   Train: {len(tr_idx)}  Test: {len(te_idx)}")

    # 5. v2 基线
    print("\n[5] v2 baseline...")
    bde.eval()
    py, yy = [], []
    with torch.no_grad():
        for b in te:
            b = b.to(dev)
            py.extend(bde(b).cpu().squeeze().tolist())
            yy.extend(b.y.cpu().tolist())
    py, yy = np.array(py)*std+mean, np.array(yy)*std+mean
    v2_mae = np.mean(np.abs(py-yy))
    v2_rmse = np.sqrt(np.mean((py-yy)**2))
    print(f"   MAE={v2_mae:.2f}  RMSE={v2_rmse:.2f}")

    # 6. 训适配器
    print("\n[6] Training adapter (500 steps)...")
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(adapter.parameters(), lr=0.001)
    t0 = time.time()
    for step, b in enumerate(tr):
        if step >= 500: break
        b = b.to(dev)
        with torch.no_grad():
            _, _, emb = spin(b)  # 提取自旋嵌入
        aug = torch.cat([b.x[:, :9], emb], dim=1)   # [9+256=265]
        b.x = torch.cat([adapter(aug), b.x[:, 9:]], dim=1)  # 新9维 + 原始标签列
        pred = bde(b)
        loss = loss_fn(pred, b.y.view(-1, 1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"   Step {step:4d}  loss={loss.item():.2f}  {time.time()-t0:.0f}s")

    # 7. 验证
    print("\n[7] v2+spin evaluation...")
    adapter.eval()
    py, yy = [], []
    with torch.no_grad():
        for b in te:
            b = b.to(dev)
            _, _, emb = spin(b)
            aug = torch.cat([b.x[:, :9], emb], dim=1)
            b.x = torch.cat([adapter(aug), b.x[:, 9:]], dim=1)
            py.extend(bde(b).cpu().squeeze().tolist())
            yy.extend(b.y.cpu().tolist())
    py, yy = np.array(py)*std+mean, np.array(yy)*std+mean
    v4_mae = np.mean(np.abs(py-yy))
    v4_rmse = np.sqrt(np.mean((py-yy)**2))

    # 8. 对比
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  v2 (no spin):     MAE={v2_mae:.2f}  RMSE={v2_rmse:.2f}")
    print(f"  v2 + spin adapter: MAE={v4_mae:.2f}  RMSE={v4_rmse:.2f}")
    print(f"  Delta:             MAE {'↓' if v4_mae < v2_mae else '↑'}{abs(v4_mae-v2_mae):.2f}")
    print(f"{'='*60}")
    return v2_mae, v4_mae


if __name__ == '__main__':
    main()

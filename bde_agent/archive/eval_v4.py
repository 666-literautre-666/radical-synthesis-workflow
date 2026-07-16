"""
v4 = v2(锁死) + spin(锁死) + 新input_proj(265→256, 只训这个)
v2基线 vs v4, 同一测试集对比
"""
import torch, torch.nn as nn, numpy as np, time, warnings, copy
warnings.filterwarnings('ignore')
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from gnn_inference import BDEGNNv2
from gnn_data_utils import load_gnn_data
from config import CONFIG
from spin_pretrain import SpinPretrainNN


def evaluate(model, spin_model, use_spin, new_proj, loader, dev, mean, std):
    """评估模型, use_spin=True 时注入自旋嵌入"""
    model.eval()
    spin_model.eval()
    if use_spin and new_proj is not None:
        new_proj.eval()

    py, yy = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(dev)
            if use_spin:
                _, _, emb = spin_model(b)
                aug = torch.cat([b.x[:, :9], emb], dim=1)   # [N, 265]
                b.x = new_proj(aug)                         # [N, 256]
            py.extend(model(b).cpu().squeeze().tolist())
            yy.extend(b.y.cpu().tolist())

    py, yy = np.array(py)*std+mean, np.array(yy)*std+mean
    return np.mean(np.abs(py-yy)), np.sqrt(np.mean((py-yy)**2))


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("v4: v2(frozen) + spin(frozen) + new input_proj(train)")
    print(f"Device: {dev}")
    print("=" * 60)

    # ==== v2权重 ====
    v2_ckpt = torch.load('gnn_bde_best_v2.pt', map_location=dev, weights_only=True)
    mean, std = v2_ckpt['bde_mean'], v2_ckpt['bde_std']
    v2 = BDEGNNv2(node_dim=10, edge_dim=4, hidden=256, n_layers=4).to(dev)
    v2.load_state_dict(v2_ckpt['model'])

    # ==== 前插网络(冻结) ====
    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(dev)
    spin.load_state_dict(torch.load('spin_pretrain_frozen.pt', weights_only=True)['backbone'], strict=False)
    spin.eval()
    for p in spin.parameters(): p.requires_grad = False

    # ==== 锁死 v2 全部权重 ====
    v2.eval()
    for p in v2.parameters(): p.requires_grad = False

    # ==== 新建 input_proj 265→256 (替换 v2 的 10→256) ====
    new_proj = nn.Linear(265, 256).to(dev)
    # 从 v2.input_proj 拷贝偏置 (维度不同, 只拷偏置)
    new_proj.bias.data.copy_(v2.input_proj.bias.data)

    print(f"\nTrainable params: {sum(p.numel() for p in new_proj.parameters()):,}")
    print(f"v2 params: {sum(p.numel() for p in v2.parameters()):,} (all frozen)")
    print(f"spin params: {sum(p.numel() for p in spin.parameters()):,} (all frozen)")

    # ==== 数据 ====
    print("\nLoading BDE data (100k)...")
    data, _, _ = load_gnn_data(CONFIG['data_path'], nrows=100000)
    tr_idx, te_idx = train_test_split(range(len(data)), test_size=0.3, random_state=42)
    _, te_idx = train_test_split(te_idx, test_size=0.5, random_state=42)
    tr = DataLoader([data[i] for i in tr_idx], batch_size=256, shuffle=True)
    te = DataLoader([data[i] for i in te_idx], batch_size=256)
    print(f"Train: {len(tr_idx)}  Test: {len(te_idx)}")

    # ==== v2 基线 (原始 input_proj 10→256) ====
    v2_mae, v2_rmse = evaluate(v2, spin, use_spin=False, new_proj=None,
                               loader=te, dev=dev, mean=mean, std=std)
    print(f"\nv2 baseline (no spin):  MAE={v2_mae:.2f}  RMSE={v2_rmse:.2f}")

    # ==== 训练 new_proj 265→256 ====
    print(f"\nTraining new_proj (v4), 800 steps...")
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(new_proj.parameters(), lr=0.001)
    t0 = time.time()

    # 临时替换 v2.input_proj
    orig_proj = v2.input_proj
    v2.input_proj = new_proj

    for step, b in enumerate(tr):
        if step >= 800: break
        b = b.to(dev)

        with torch.no_grad():
            _, _, emb = spin(b)
        b.x = torch.cat([b.x[:, :9], emb], dim=1)   # [N, 265]
        pred = v2(b)                                  # 走 input_proj→GINEConv→fc

        loss = loss_fn(pred, b.y.view(-1, 1))
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 200 == 0:
            print(f"  Step {step:4d}  loss={loss.item():.2f}  {time.time()-t0:.0f}s")

    # ==== v4 评估 ====
    v4_mae, v4_rmse = evaluate(v2, spin, use_spin=True, new_proj=orig_proj,
                               loader=te, dev=dev, mean=mean, std=std)
    # 不对——evaluate 里 use_spin=True 时会用 new_proj 替换 x, 但这里 v2.input_proj 已经是 new_proj
    # 简化: 手动评估
    v2.eval(); spin.eval(); new_proj.eval()
    py, yy = [], []
    with torch.no_grad():
        for b in te:
            b = b.to(dev)
            _, _, emb = spin(b)
            b.x = new_proj(torch.cat([b.x[:, :9], emb], dim=1))
            py.extend(v2(b).cpu().squeeze().tolist())
            yy.extend(b.y.cpu().tolist())
    py, yy = np.array(py)*std+mean, np.array(yy)*std+mean
    v4_mae = np.mean(np.abs(py-yy))
    v4_rmse = np.sqrt(np.mean((py-yy)**2))

    # 恢复
    v2.input_proj = orig_proj

    # ==== 对比 ====
    print(f"\n{'='*50}")
    print(f"v2 (random init):    MAE={v2_mae:.2f}  RMSE={v2_rmse:.2f}")
    print(f"v4 (+spin pretrain):  MAE={v4_mae:.2f}  RMSE={v4_rmse:.2f}")
    delta = v2_mae - v4_mae
    print(f"Improvement:          {'↓' if delta > 0 else '↑'}{abs(delta):.2f} kcal/mol")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

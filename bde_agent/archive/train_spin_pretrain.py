"""
SpinPretrainNN 预训练 — QM9star 71万自由基
训练任务: 分子图 → 每原子自旋密度 + Mulliken电荷
"""
import torch
import torch.nn as nn
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from spin_pretrain import SpinPretrainNN
from spin_dataset import QM9starRadicalDataset

# ======== 配置 ========
CONFIG = {
    'csv_path': 'E:/qm9star_radicals.csv',
    'n_molecules': 50000,        # 5万起步，后续调大
    'gnn_hidden': 256,
    'gnn_layers': 4,
    'dropout': 0.3,
    'batch_size': 128,
    'epochs': 300,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'patience': 80,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def main():
    cfg = CONFIG
    print("=" * 60)
    print("SpinPretrainNN — QM9star Pre-training")
    print(f"Device: {cfg['device']} | Molecules: {cfg['n_molecules']:,}")
    print("=" * 60)

    # ======== 1. 数据 ========
    print("\n[1/4] Loading data...")
    ds = QM9starRadicalDataset(
        cfg['csv_path'],
        max_molecules=cfg['n_molecules'],
        root='data/qm9star_processed'
    )

    # 划分
    idx = list(range(len(ds)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.15, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    train_data = [ds[i] for i in train_idx]
    val_data = [ds[i] for i in val_idx]
    test_data = [ds[i] for i in test_idx]

    bs = cfg['batch_size']
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs)
    test_loader = DataLoader(test_data, batch_size=bs)

    print(f"Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

    # 统计样本
    sample = ds[0]
    print(f"Node dim: {sample.x.shape[1]}  Edge dim: {sample.edge_attr.shape[1]}")

    # ======== 2. 模型 ========
    print("\n[2/4] Building model...")
    model = SpinPretrainNN(
        node_dim=10,
        edge_dim=4,
        hidden=cfg['gnn_hidden'],
        n_layers=cfg['gnn_layers'],
        dropout=cfg['dropout'],
    ).to(cfg['device'])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ======== 3. 训练 ========
    print("\n[3/4] Training...")
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg['epochs'], eta_min=1e-6)

    best_val = float('inf')
    best_epoch = 0
    t0 = time.time()
    device = cfg['device']

    for epoch in range(cfg['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            spin_pred, charge_pred, _ = model(batch)
            loss_spin = loss_fn(spin_pred, batch.y_spin)
            loss_charge = loss_fn(charge_pred, batch.y_charge)
            loss = loss_spin + 0.3 * loss_charge  # 自旋密度为主, 电荷辅助
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                spin_pred, _, _ = model(batch)
                val_loss += loss_fn(spin_pred, batch.y_spin).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
                'config': cfg,
            }, 'spin_pretrain_best.pt')

        if epoch % 25 == 0 or epoch < 5:
            elapsed = time.time() - t0
            lr_now = opt.param_groups[0]['lr']
            print(f"E{epoch:4d}  Tr={train_loss:.4f}  Va={val_loss:.4f}  "
                  f"Best={best_val:.4f}@{best_epoch}  lr={lr_now:.2e}  {elapsed/60:.0f}min")

        if epoch - best_epoch > cfg['patience']:
            print(f"Early stop at epoch {epoch}")
            break

    # ======== 4. 评估 ========
    print(f"\n[4/4] Evaluating...")
    ckpt = torch.load('spin_pretrain_best.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()

    test_mae = 0.0
    all_preds, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            spin_pred, _, _ = model(batch)
            all_preds.extend(spin_pred.cpu().tolist())
            all_y.extend(batch.y_spin.cpu().tolist())
            test_mae += torch.abs(spin_pred - batch.y_spin).mean().item()
    test_mae /= len(test_loader)

    preds = np.array(all_preds)
    ys = np.array(all_y)
    rmse = np.sqrt(np.mean((preds - ys) ** 2))
    corr = np.corrcoef(preds, ys)[0, 1]

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (Pre-training)")
    print(f"  Best epoch: {ckpt['epoch']}")
    print(f"  Spin Density MAE: {test_mae:.4f}")
    print(f"  Spin Density RMSE: {rmse:.4f}")
    print(f"  Pearson r: {corr:.4f}")
    print(f"  Spin range: [{ys.min():.3f}, {ys.max():.3f}]")
    print(f"{'='*60}")

    # 导出冻结主干
    from spin_pretrain import freeze_and_export
    freeze_and_export(model, 'spin_pretrain_best.pt', 'spin_pretrain_frozen.pt')


if __name__ == '__main__':
    main()

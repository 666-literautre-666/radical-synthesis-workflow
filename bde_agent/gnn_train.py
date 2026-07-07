"""GNN BDE v2 — GINEConv + 边特征 + 残差 + 边界原子池化"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from gnn_data_utils import load_gnn_data
from config import CONFIG


# ======== 1. GNN 模型 v2 ========
class BDEGNNv3(nn.Module):
    """双通道: GINEConv 图通道 + MLP 物理特征通道"""

    def __init__(self, node_dim=10, edge_dim=4, phys_dim=12, hidden=256, n_layers=4, dropout=0.4):
        super().__init__()
        self.hidden = hidden

        # ==== GNN 图通道 ====
        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        # ==== 物理特征 MLP 通道 ====
        self.phys_net = nn.Sequential(
            nn.Linear(phys_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
        )

        # ==== 汇聚层 ====
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2 + 32, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # GNN 图通道
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new).relu()
            h_new = self.dropout(h_new)
            h = h + h_new

        # 目标原子提取
        is_target = (x[:, -1] == 3.0)
        target_embs = h[is_target]
        target_embs = target_embs.view(-1, self.hidden * 2)

        # 物理特征通道
        phys_emb = self.phys_net(data.phys)

        # 双通道汇聚
        combined = torch.cat([target_embs, phys_emb], dim=1)
        return self.fc(combined)


if __name__ == '__main__':
    # ======== 2. 载入数据 ========
    print("=" * 60)
    print("GNN BDE v2 — GINEConv + Edge Features + Target Atom Pooling")
    print("=" * 60)

    nrows = CONFIG.get('nrows', 100000)
    data_list, bde_mean, bde_std = load_gnn_data(CONFIG['data_path'], nrows=nrows)
    print(f"BDE mean={bde_mean:.1f}, std={bde_std:.1f} kcal/mol")

    idx = list(range(len(data_list)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    bs = CONFIG['batch_size']
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs)
    test_loader = DataLoader(test_data, batch_size=bs)
    print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    # ======== 3. 模型 + 优化器 ========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(16)
    print(f"Device: {device}  Threads: {torch.get_num_threads()}")

    model = BDEGNNv3(
        node_dim=12,
        edge_dim=4,
        hidden=CONFIG['gnn_hidden'],
        n_layers=CONFIG['gnn_layers'],
        dropout=CONFIG['dropout'],
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['epochs'], eta_min=1e-6)
    best_val = float('inf')
    best_epoch = 0
    start_epoch = 0

    # 断点续训
    ckpt_path = 'gnn_bde_checkpoint_v2.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val', float('inf'))
        best_epoch = ckpt.get('best_epoch', 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

    # ======== 4. 训练 ========
    t0 = time.time()
    progress_path = r'C:\Users\xushaobo\Desktop\gnn_progress.txt'

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y.view(-1, 1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += loss_fn(pred, batch.y.view(-1, 1)).item()
        val_loss /= len(val_loader)

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
                'best_epoch': best_epoch,
                'bde_mean': bde_mean,
                'bde_std': bde_std,
            }, 'gnn_bde_best_v2.pt')

        if epoch % 100 == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
                'best_epoch': best_epoch,
                'bde_mean': bde_mean,
                'bde_std': bde_std,
            }, ckpt_path)

        if epoch % 10 == 0:
            elapsed = time.time() - t0
            lr_now = opt.param_groups[0]['lr']
            msg = (f"E{epoch:4d} Tr={train_loss:.4f} Va={val_loss:.4f}  "
                   f"Best={best_val:.4f}@{best_epoch} lr={lr_now:.2e}  {elapsed/60:.0f}min")
            print(msg)
            with open(progress_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

        if epoch - best_epoch > CONFIG['patience']:
            print(f"Early stop at epoch {epoch}")
            break

    # ======== 5. 测试 ========
    print("\n" + "=" * 60)
    print("Evaluating best model on test set...")
    best_ckpt = torch.load('gnn_bde_best_v2.pt', map_location=device)
    model.load_state_dict(best_ckpt['model'])
    model.eval()

    test_mse = 0
    all_preds, all_y = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_mse += loss_fn(pred, batch.y.view(-1, 1)).item()
            all_preds.extend(pred.cpu().squeeze().tolist())
            all_y.extend(batch.y.cpu().tolist())

    test_mse /= len(test_loader)
    preds_arr = np.array([p * bde_std + bde_mean for p in all_preds])
    y_arr = np.array([y * bde_std + bde_mean for y in all_y])
    mae = np.mean(np.abs(preds_arr - y_arr))
    rmse = np.sqrt(np.mean((preds_arr - y_arr) ** 2))
    r2 = np.corrcoef(preds_arr, y_arr)[0, 1] ** 2

    msg = (
        f"\n=== TEST RESULTS ===\n"
        f"Best epoch: {best_ckpt['epoch']}\n"
        f"MSE (norm): {test_mse:.4f}\n"
        f"MAE: {mae:.2f} kcal/mol\n"
        f"RMSE: {rmse:.2f} kcal/mol\n"
        f"R²: {r2:.4f}\n"
        f"BDE range: {y_arr.min():.0f}-{y_arr.max():.0f} kcal/mol\n"
    )
    print(msg)
    with open(progress_path, 'a', encoding='utf-8') as f:
        f.write(msg)

    torch.save({
        'model': model.state_dict(),
        'bde_mean': bde_mean,
        'bde_std': bde_std,
    }, 'gnn_bde_model_v2.pt')
    print("Saved: gnn_bde_model_v2.pt")

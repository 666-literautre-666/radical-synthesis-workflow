"""
BDEGNNv5 — 双通道物理特征注入 + 多任务正则化 + Spin Dropout + Information Bottleneck
  Channel A (压缩): SpinPretrainNN 256→16 (瓶颈压缩), 拼入化学特征 → GNN主体
  Channel B (残差): SpinPretrainNN 256→hidden, 注入GNN末层 → 信息不丢
  多任务: BDE + spin_density + charge 三个预测头联合训练
  Spin Dropout: 训练时随机概率将spin特征置零, 强制GNN自主学习化学特征
"""
import torch, torch.nn as nn, numpy as np, os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from gnn_data_utils import load_gnn_data
from spin_pretrain import SpinPretrainNN


# ======== v5 模型: 双通道 + 多任务 + Spin Dropout + Bottleneck ========
class BDEGNNv5(nn.Module):
    """
    v5.1 改进:
    1. Information Bottleneck: spin 256→16 (原64), 降低spin信息优势
    2. Spin Dropout: 训练时以概率p将spin特征置零, 强制GNN用化学特征
    3. Channel B残差保留: 即使Channel A被dropout, 末层仍可注入spin信息
    4. 三头多任务: BDE + spin + charge
    """

    def __init__(self, frozen_spin, node_dim=20, edge_dim=6, hidden=256,
                 n_layers=4, dropout=0.1, spin_compact_dim=56,
                 spin_dropout_prob=0.3):
        super().__init__()
        self.hidden = hidden
        self.spin_compact_dim = spin_compact_dim
        self.spin_dropout_prob = spin_dropout_prob
        self.spin = frozen_spin
        for p in self.spin.parameters():
            p.requires_grad = False

        # ---- Channel A: 阶梯 Bottleneck 256→128→64→32 + LayerNorm ----
        self.spin_compact = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 4),
            nn.LayerNorm(hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, spin_compact_dim),
        )

        # ---- Channel B: 残差旁路 (门控) ----
        self.spin_residual = nn.Linear(hidden, hidden)
        self.residual_gate = nn.Parameter(torch.tensor(1.0))  # sigmoid(1.0)≈0.73, 开门充分

        # ---- 输入投影: node(x+x_phys=20) + spin_compact(16) = 36 → hidden ----
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
        self.spin_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )
        self.bde_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        # 1. 冻结前插网络 → 物理特征 (spin 返回 sp,ch,bo,gr,emb, 取需要的)
        with torch.no_grad():
            out = self.spin(data)
            spin_pseudo, charge_pseudo, emb = out[0], out[1], out[-1]

        # 2. Channel A: Bottleneck压缩 + Spin Dropout
        spin_c = self.spin_compact(emb)  # [N, 16]

        # Spin Dropout: 训练时以概率p将spin特征置零
        if self.training and self.spin_dropout_prob > 0:
            mask = (torch.rand(1, device=spin_c.device) > self.spin_dropout_prob).float()
            spin_c = spin_c * mask  # 30%概率全零

        edge_full = torch.cat([data.edge_attr,
                               data.edge_phys if hasattr(data, 'edge_phys') else
                               torch.zeros(data.edge_attr.shape[0], 2, device=data.edge_attr.device)], dim=-1)
        x_phys = data.x_phys if hasattr(data, 'x_phys') else \
                 torch.zeros(data.x.shape[0], 10, device=data.x.device)
        x_aug = torch.cat([data.x, x_phys, spin_c], dim=-1)

        # 3. GNN 主干
        h = self.input_proj(x_aug)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index, edge_full)
            h = h + self.dropout(norm(h_new).relu())

        # 4. Channel B: 残差注入 (门控) — 即使spin被dropout, 这里提供安全网
        spin_r = self.spin_residual(emb)  # [N, hidden]
        gate = torch.sigmoid(self.residual_gate)
        h = h + gate * spin_r

        # 5. 多任务预测
        spin_pred = self.spin_head(h).squeeze(-1)
        charge_pred = self.charge_head(h).squeeze(-1)

        # 6. BDE (目标原子池化)
        is_target = (data.x[:, -1] == 3.0)
        t_emb = h[is_target].view(-1, self.hidden * 2)
        bde_pred = self.bde_head(t_emb)

        return bde_pred, spin_pred, charge_pred, spin_pseudo, charge_pseudo


# ======== 训练配置 ========
CFG = {
    'data_path': 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz',
    'nrows': 800000,
    'hidden': 256, 'n_layers': 4, 'dropout': 0.3,
    'batch_size': 256, 'epochs': 400, 'lr': 0.001, 'weight_decay': 0.0001,
    'patience': 120,
    'lambda_spin': 0.3,
    'lambda_charge': 0.1,
    'spin_compact_dim': 48,      # 阶梯 Bottleneck: 256→128→64→48
    'spin_dropout_prob': 0.2,    # 20%概率关闭spin特征
    'lr_gnn': 0.0005,            # GNN 分支学习率 (扶持弱势群体)
    'lr_spin': 0.00005,          # Spin 分支学习率 (强约束物理特征梯度)
}


def main():
    import hashlib, datetime, json
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 留痕: 日志目录 + 时间戳
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'training_logs')
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'v5.1_{run_id}.log')
    progress_path = os.path.join(log_dir, f'v5.1_{run_id}_progress.txt')

    # 数据指纹
    data_hash = hashlib.md5(open(CFG['data_path'], 'rb').read(4096)).hexdigest()[:6]
    git_commit = os.popen('git rev-parse --short HEAD 2>/dev/null').read().strip() or 'unknown'

    # 所有输出双写: 终端 + 文件
    class Tee:
        def __init__(self, f):
            self.f = f; self.stdout = sys.stdout
        def write(self, s):
            self.f.write(s); self.stdout.write(s)
        def flush(self):
            self.f.flush(); self.stdout.flush()
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(log_f)

    print("=" * 60)
    print(f"BDEGNNv5.1: Bottleneck + Spin Dropout + Physical Features | Device: {dev}")
    print(f"Run ID: {run_id} | Git: {git_commit} | Data: {CFG['data_path']} (md5={data_hash})")
    print(f"Log: {log_path}")
    print("=" * 60)

    # 1. 冻结前插网络
    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(dev)
    spin_pt = 'spin_pretrain_frozen.pt'
    if not os.path.exists(spin_pt):
        spin_pt = os.path.join(os.path.expanduser('~'), 'spin_pretrain_frozen.pt')
    backbone = torch.load(spin_pt, map_location=dev, weights_only=True)['backbone']
    backbone = {k: v for k, v in backbone.items() if not any(k.startswith(h) for h in ['spin_head', 'charge_head', 'graph_head', 'bond_head', 'gap_head'])}
    spin.load_state_dict(backbone, strict=False)
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
    model = BDEGNNv5(spin, node_dim=20, hidden=CFG['hidden'], n_layers=CFG['n_layers'],
                     dropout=CFG['dropout'], spin_compact_dim=CFG['spin_compact_dim'],
                     spin_dropout_prob=CFG['spin_dropout_prob']).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Model: {trainable:,} trainable + {frozen:,} frozen")
    print(f"  spin_compact: 256->{CFG['spin_compact_dim']} (bottleneck)")
    print(f"  spin_dropout: {CFG['spin_dropout_prob']}")
    print(f"  residual_gate init = {model.residual_gate.item():.4f}")

    # 4. 训练 — 差分学习率: GNN 1e-3, Spin分支 1e-4
    mse = nn.MSELoss()
    spin_param_names = ['spin_compact', 'spin_residual', 'residual_gate']
    spin_params, gnn_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(n) for n in spin_param_names):
            spin_params.append(p)
        else:
            gnn_params.append(p)
    opt = torch.optim.AdamW([
        {'params': gnn_params, 'lr': CFG['lr_gnn']},
        {'params': spin_params, 'lr': CFG['lr_spin']},
    ], weight_decay=CFG['weight_decay'])
    print(f"Optimizer: GNN params={sum(p.numel() for p in gnn_params):,} lr={CFG['lr_gnn']}, "
          f"Spin params={sum(p.numel() for p in spin_params):,} lr={CFG['lr_spin']}")
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-6)
    best_va, best_ep, t0 = float('inf'), 0, time.time()

    for ep in range(CFG['epochs']):
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

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for b in va_ld:
                b = b.to(dev)
                bde_pred, _, _, _, _ = model(b)
                va_loss += mse(bde_pred, b.y.view(-1, 1)).item()
        va_loss /= len(va_ld)
        sch.step()

        if va_loss < best_va:
            best_va, best_ep = va_loss, ep
            torch.save({
                'model': model.state_dict(), 'epoch': ep, 'best_val': best_va,
                'bde_mean': bde_m, 'bde_std': bde_s,
                'residual_gate': model.residual_gate.item(),
                'spin_dropout_prob': CFG['spin_dropout_prob'],
                'spin_compact_dim': CFG['spin_compact_dim'],
                '_meta': {'run_id': run_id, 'git': git_commit, 'data_hash': data_hash,
                          'config': CFG, 'timestamp': datetime.datetime.now().isoformat()},
            }, 'gnn_bde_v5.1_best.pt')

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
    ckpt = torch.load('gnn_bde_v5.1_best.pt', map_location=dev, weights_only=True)
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
        f"v5.1 TEST RESULTS (Bottleneck + Spin Dropout + Physical Features)\n"
        f"{'='*60}\n"
        f"MAE:           {mae:.2f} kcal/mol\n"
        f"RMSE:          {rmse:.2f} kcal/mol\n"
        f"R^2:           {r2:.4f}\n"
        f"Best epoch:    {ckpt['epoch']}\n"
        f"Best val_loss: {ckpt['best_val']:.6f}\n"
        f"Residual gate: {torch.sigmoid(torch.tensor(ckpt.get('residual_gate', 0))).item():.4f}\n"
        f"Spin dropout:  {ckpt.get('spin_dropout_prob', 'N/A')}\n"
        f"Bottleneck:    256->{ckpt.get('spin_compact_dim', 'N/A')}\n"
        f"BDE range:     {y_arr.min():.0f}-{y_arr.max():.0f} kcal/mol\n"
    )
    print(result)
    with open('C:/Users/xushaobo/Desktop/v5.1_results.txt', 'w', encoding='utf-8') as f:
        f.write(result)

    # 可复现元数据 JSON
    import json
    meta = {
        'run_id': run_id, 'git': git_commit, 'data_hash': data_hash,
        'model_version': 'v5.1', 'nrows': CFG.get('nrows',800000),
        'hidden': CFG['gnn_hidden'], 'n_layers': CFG['gnn_layers'],
        'compact_dim': CFG.get('spin_compact_dim',16),
        'spin_dropout': CFG.get('spin_dropout',0.3),
        'frozen_spin': spin_pt,
        'lr_main': CFG['lr'], 'lr_spin': CFG['lr_spin'],
        'batch_size': CFG['batch_size'],
        'epochs': ckpt['epoch'], 'best_val': ckpt['best_val'],
        'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2,
        'final_gate': gate_val, 'spin_mae': spin_mae,
        'bde_range': [float(y_arr.min()), float(y_arr.max())],
    }
    with open('C:/Users/xushaobo/Desktop/v5.1_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Saved: gnn_bde_v5.1_best.pt")


if __name__ == '__main__':
    main()

"""
BDEGNNv5-Δ — 与 v5 架构完全相同，唯一区别：
  label 从 BDE 改成 delta = BDE - rule_BDE
  推理时: BDE_pred = rule_BDE + delta_pred

不改模型一行代码，纯粹换 label。不行随时切回 v5。
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, os, time, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger, Chem
RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from gnn_data_utils import mol_to_data
from spin_pretrain import SpinPretrainNN
from rule_bde import estimate_bond_bde
from train_v5 import BDEGNNv5  # 复用 v5 架构, 不动


# ======== Δ-learning 数据加载 ========
def load_delta_data(csv_path, nrows=None):
    """
    加载 BDE 数据，为每个分子计算 rule_bde，target 改为 delta.
    返回与 load_gnn_data 相同格式：(data_list, delta_mean, delta_std)
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    total = len(df)

    # 第一遍: 计算所有 rule_bde 以得到 delta 统计量
    print(f"Computing rule BDE for {total:,} molecules...")
    bde_all = df['bde'].values.astype(float)
    rule_all = np.zeros(total, dtype=np.float32)

    for i in range(total):
        if i % 100000 == 0:
            print(f"  Rule BDE: {i}/{total}...")
        try:
            mol = Chem.MolFromSmiles(df.iloc[i]['molecule'])
            if mol is not None:
                mol_exp = Chem.AddHs(mol)  # CSV bond_index is on explicit-H
                rule_all[i] = estimate_bond_bde(mol_exp, int(df.iloc[i]['bond_index']))
            else:
                rule_all[i] = bde_all[i]  # fallback
        except Exception:
            rule_all[i] = bde_all[i]

    delta_all = bde_all - rule_all
    delta_mean = float(np.mean(delta_all))
    delta_std = float(np.std(delta_all))
    print(f"Delta stats: mean={delta_mean:.2f}, std={delta_std:.2f} kcal/mol")
    print(f"Rule BDE MAE vs true: {np.mean(np.abs(delta_all)):.2f} kcal/mol")

    # 第二遍: 构建图数据, target = normalized delta
    data_list = []
    skipped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  Building graphs: {i}/{total} molecules...")
        delta_norm = (delta_all[i] - delta_mean) / (delta_std + 1e-8)

        d = mol_to_data(
            smiles=row['molecule'],
            frag1_smi=str(row.get('fragment1', '')),
            frag2_smi=str(row.get('fragment2', '')),
            bond_idx=int(row['bond_index']),
            bde_value=delta_norm,  # 存的是归一化 delta
        )
        if d is not None:
            d.rule_bde = rule_all[i]  # 保存原始 rule_BDE 用于推理时反推
            data_list.append(d)
        else:
            skipped += 1

    print(f"Loaded {len(data_list)} graphs (skipped {skipped})")
    return data_list, delta_mean, delta_std


# ======== 训练配置 (同 v5) ========
CFG = {
    'data_path': 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz',
    'nrows': 800000,
    'hidden': 256, 'n_layers': 4, 'dropout': 0.3,
    'batch_size': 128, 'epochs': 400, 'lr': 0.001, 'weight_decay': 0.0001,
    'patience': 120,
    'lambda_spin': 0.3,
    'lambda_charge': 0.1,
}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"BDEGNNv5-Delta: rule_BDE baseline + GNN residual | Device: {dev}")
    print("=" * 60)

    # 1. 冻结前插网络
    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(dev)
    spin.load_state_dict(torch.load('spin_pretrain_frozen.pt', weights_only=True)['backbone'], strict=False)
    spin.eval()
    print(f"SpinPretrainNN: {sum(p.numel() for p in spin.parameters()):,} params (frozen)")

    # 2. Δ数据: label = BDE - rule_BDE
    data_list, delta_m, delta_s = load_delta_data(CFG['data_path'], nrows=CFG['nrows'])
    print(f"Delta data: mean={delta_m:.2f}, std={delta_s:.2f} kcal/mol")

    idx = list(range(len(data_list)))
    tr, tmp = train_test_split(idx, test_size=0.2, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    tr_ld = DataLoader([data_list[i] for i in tr], batch_size=CFG['batch_size'], shuffle=True)
    va_ld = DataLoader([data_list[i] for i in va], batch_size=CFG['batch_size'])
    te_ld = DataLoader([data_list[i] for i in te], batch_size=CFG['batch_size'])
    print(f"Train: {len(tr)}  Val: {len(va)}  Test: {len(te)}")

    # 3. 模型 (完全复用 v5 架构)
    model = BDEGNNv5(spin, node_dim=10, hidden=CFG['hidden'], n_layers=CFG['n_layers'],
                     dropout=CFG['dropout']).to(dev)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {trainable:,} trainable (same as v5)")

    # 4. 训练
    mse = nn.MSELoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG['epochs'], eta_min=1e-6)
    best_va, best_ep, t0 = float('inf'), 0, time.time()
    progress_path = r'C:\Users\xushaobo\Desktop\v5_delta_progress.txt'

    for ep in range(CFG['epochs']):
        # ---- Train ----
        model.train()
        tr_bde, tr_sp, tr_ch = 0.0, 0.0, 0.0
        for b in tr_ld:
            b = b.to(dev)
            bde_pred, spin_pred, charge_pred, spin_pseudo, charge_pseudo = model(b)

            loss_bde = mse(bde_pred, b.y.view(-1, 1))
            loss_spin = mse(spin_pred, spin_pseudo)
            loss_charge = mse(charge_pred, charge_pseudo)
            loss = loss_bde + CFG['lambda_spin'] * loss_spin + CFG['lambda_charge'] * loss_charge

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_bde += loss_bde.item(); tr_sp += loss_spin.item(); tr_ch += loss_charge.item()

        tr_bde /= len(tr_ld); tr_sp /= len(tr_ld); tr_ch /= len(tr_ld)

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

        if va_loss < best_va:
            best_va, best_ep = va_loss, ep
            torch.save({
                'model': model.state_dict(), 'epoch': ep, 'best_val': best_va,
                'delta_mean': delta_m, 'delta_std': delta_s,
            }, 'gnn_bde_v5_delta_best.pt')

        if ep % 25 == 0 or ep < 5:
            elapsed = time.time() - t0
            msg = (f"E{ep:4d} d_tr={tr_bde:.4f} sp_tr={tr_sp:.4f} ch_tr={tr_ch:.4f}  "
                   f"va={va_loss:.4f} best={best_va:.4f}@{best_ep}  {elapsed/60:.0f}min")
            print(msg)
            with open(progress_path, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

        if ep - best_ep > CFG['patience']:
            print(f"Early stop at epoch {ep}")
            break

    # ======== 5. 测试 (反推回 BDE) ========
    print("\n" + "=" * 60)
    print("Evaluating best model on test set...")
    ckpt = torch.load('gnn_bde_v5_delta_best.pt', map_location=dev, weights_only=True)
    model.load_state_dict(ckpt['model']); model.eval()

    preds_bde, ys_bde = [], []
    with torch.no_grad():
        for b in te_ld:
            b = b.to(dev)
            delta_pred, _, _, _, _ = model(b)
            # 反推: delta_norm → delta → BDE = rule_BDE + delta
            delta_pred_np = delta_pred.cpu().squeeze().numpy()
            delta_real = delta_pred_np * delta_s + delta_m
            bde_pred = delta_real + b.rule_bde.cpu().numpy()
            preds_bde.extend(bde_pred.tolist())

            # 真实 BDE 也从 delta 标签反推
            y_delta_norm = b.y.cpu().numpy()
            y_delta = y_delta_norm * delta_s + delta_m
            y_bde = y_delta + b.rule_bde.cpu().numpy()
            ys_bde.extend(y_bde.tolist())

    preds_arr = np.array(preds_bde)
    y_arr = np.array(ys_bde)
    mae = np.mean(np.abs(preds_arr - y_arr))
    rmse = np.sqrt(np.mean((preds_arr - y_arr) ** 2))
    r2 = np.corrcoef(preds_arr, y_arr)[0, 1] ** 2

    result = (
        f"\n{'='*60}\n"
        f"v5-DELTA TEST RESULTS\n"
        f"{'='*60}\n"
        f"MAE:             {mae:.2f} kcal/mol\n"
        f"RMSE:            {rmse:.2f} kcal/mol\n"
        f"R^2:             {r2:.4f}\n"
        f"Best epoch:      {ckpt['epoch']}\n"
        f"Best delta_loss: {ckpt['best_val']:.6f}\n"
        f"Delta mean/std:  {delta_m:.1f}/{delta_s:.1f} kcal/mol\n"
        f"Rule BDE baseline MAE:  {np.mean(np.abs(delta_m)):.2f} kcal/mol\n"
        f"BDE range:       {y_arr.min():.0f}-{y_arr.max():.0f} kcal/mol\n"
    )
    print(result)
    with open('C:/Users/xushaobo/Desktop/v5_delta_results.txt', 'w', encoding='utf-8') as f:
        f.write(result)

    print("Saved: gnn_bde_v5_delta_best.pt")


if __name__ == '__main__':
    main()

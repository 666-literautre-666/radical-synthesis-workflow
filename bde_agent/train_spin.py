"""
SpinPretrainNN 预训练 — 直接脚本, 不用 PyG Dataset 框架
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, os, time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

from spin_pretrain import SpinPretrainNN, freeze_and_export

# ======== 配置 ========
CSV_PATH = 'E:/qm9star_radicals_v2.csv'
N_MOLECULES = 700000
LOG_PATH = 'C:/Users/xushaobo/Desktop/spin_v2_progress.txt'
HIDDEN = 256
N_LAYERS = 4
BATCH_SIZE = 128
EPOCHS = 500
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== 数据加载 ========
def load_data(csv_path, n_molecules):
    """分块读 CSV, 收集前 n_molecules 个分子, 构建 PyG Data 列表"""
    print(f"Scanning {csv_path} for {n_molecules:,} molecules...")
    t0 = time.time()

    mol_data = {}  # {mol_id: [(atom_idx, spin, charge, f_rad, smiles), ...]}
    chunk_size = 300000  # ~15MB per chunk

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        for _, row in chunk.iterrows():
            mid = int(row['mol_id'])
            if mid not in mol_data:
                if len(mol_data) >= n_molecules:
                    break
                mol_data[mid] = []
            mol_data[mid].append((
                int(row['atom_idx']), float(row['spin_density']),
                float(row['mulliken_charge']), float(row['npa_charge']),
                int(row['formal_radical']),
                float(row['wiberg_bo_sum']), float(row['nbo_bo_sum']),
                row['smiles'],
                float(row['alpha_gap']), float(row['beta_gap']),
                float(row['alpha_homo']), float(row['alpha_lumo']),
                float(row['beta_homo']), float(row['beta_lumo']),
                float(row['dipole_mag'])
            ))
        if len(mol_data) >= n_molecules:
            break
        print(f"  Chunk {i+1}: scanned {len(chunk):,} rows, collected {len(mol_data):,} molecules")

    print(f"Collected {len(mol_data):,} molecules in {(time.time()-t0)/60:.1f} min")

    # 构建图
    print(f"Building molecular graphs...")
    data_list = []
    atom_symbols = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'Cl':17,'Br':35,'I':53}
    skipped = 0

    for idx, (mid, rows) in enumerate(mol_data.items()):
        smiles = rows[0][7]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue

        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()

        # 节点标签: spin, mulliken_charge, f_rad
        spin_densities = np.zeros(n_atoms, dtype=np.float32)
        charges_mull = np.zeros(n_atoms, dtype=np.float32)
        f_radicals = np.zeros(n_atoms, dtype=np.float32)
        for row in rows:
            idx_a = row[0]
            if idx_a < n_atoms:
                spin_densities[idx_a] = row[1]
                charges_mull[idx_a] = row[2]
                f_radicals[idx_a] = row[4]

        # 图级标签: alpha_homo, alpha_lumo, beta_homo, beta_lumo, dipole_mag (5个)
        r0 = rows[0]
        y_graph = np.array([r0[10], r0[11], r0[12], r0[13], r0[14]],
                          dtype=np.float32)

        # 节点特征 (10维)
        node_feats = []
        for atom in mol.GetAtoms():
            a_idx = atom.GetIdx()
            node_feats.append([
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetTotalNumHs()),
                float(atom.GetIsAromatic()),
                float(atom.GetFormalCharge()),
                float(atom.GetHybridization() == Chem.HybridizationType.SP),
                float(atom.GetHybridization() == Chem.HybridizationType.SP2),
                float(atom.GetHybridization() == Chem.HybridizationType.SP3),
                float(atom.IsInRing()),
                f_radicals[a_idx],
            ])

        x = torch.tensor(node_feats, dtype=torch.float32)

        # 边特征 (4维)
        ei0, ei1, ea = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ei0.extend([i, j]); ei1.extend([j, i])
            bt = bond.GetBondType()
            f = [float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE),
                 float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC)]
            ea.extend([f, f])

        edge_index = torch.tensor([ei0, ei1], dtype=torch.long)
        edge_attr = torch.tensor(ea, dtype=torch.float32)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
            y_spin=torch.tensor(spin_densities, dtype=torch.float32),
            y_charge=torch.tensor(charges_mull, dtype=torch.float32),
            y_graph=torch.tensor(y_graph, dtype=torch.float32).unsqueeze(0),
            smiles=smiles))

        if (idx+1) % 10000 == 0:
            print(f"  {idx+1:,}/{len(mol_data):,} graphs")

    print(f"Built {len(data_list)} graphs (skipped {skipped}) in {(time.time()-t0)/60:.1f} min")
    return data_list


# ======== 训练 ========
def main():
    print("=" * 60)
    print(f"SpinPretrainNN Pre-training | Device: {DEVICE} | N: {N_MOLECULES:,}")
    print("=" * 60)

    # 数据
    data_list = load_data(CSV_PATH, N_MOLECULES)

    idx = list(range(len(data_list)))
    tr_idx, tmp = train_test_split(idx, test_size=0.15, random_state=42)
    va_idx, te_idx = train_test_split(tmp, test_size=0.5, random_state=42)
    tr_loader = DataLoader([data_list[i] for i in tr_idx], batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader([data_list[i] for i in va_idx], batch_size=BATCH_SIZE)
    print(f"Train: {len(tr_idx):,}  Val: {len(va_idx):,}  Test: {len(te_idx):,}")

    # 模型
    model = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=HIDDEN, n_layers=N_LAYERS, dropout=0.3).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_va, best_ep, t0 = float('inf'), 0, time.time()

    for ep in range(EPOCHS):
        model.train()
        tr_loss = 0.0
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            sp, ch, gr, _ = model(batch)
            loss = (loss_fn(sp, batch.y_spin) + 0.3 * loss_fn(ch, batch.y_charge) +
                    0.3 * loss_fn(gr, batch.y_graph))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(DEVICE)
                sp, _, _, _ = model(batch)
                va_loss += loss_fn(sp, batch.y_spin).item()
        va_loss /= len(va_loader)
        sch.step()

        if va_loss < best_va:
            best_va, best_ep = va_loss, ep
            torch.save({'model': model.state_dict(), 'epoch': ep, 'best_val': best_va}, 'spin_pretrain_best.pt')

        if ep % 25 == 0 or ep < 5:
            msg = f"E{ep:4d}  tr={tr_loss:.4f}  va={va_loss:.4f}  best={best_va:.4f}@{best_ep}  {((time.time()-t0)/60):.0f}min"
            print(msg)
            with open(LOG_PATH, 'a') as lf: lf.write(msg + '\n')

        if ep - best_ep > 80:
            msg = f"Early stop at {ep}"
            print(msg)
            with open(LOG_PATH, 'a') as lf: lf.write(msg + '\n')
            break

    # 测试
    print("\nTesting...")
    ckpt = torch.load('spin_pretrain_best.pt', map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model']); model.eval()
    te_loader = DataLoader([data_list[i] for i in te_idx], batch_size=BATCH_SIZE)
    preds, ys = [], []
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(DEVICE); sp, _, _, _ = model(batch)
            preds.extend(sp.cpu().tolist()); ys.extend(batch.y_spin.cpu().tolist())
    preds, ys = np.array(preds), np.array(ys)
    mae = np.mean(np.abs(preds - ys))
    rmse = np.sqrt(np.mean((preds - ys)**2))
    r = np.corrcoef(preds, ys)[0,1]
    result = f"Test MAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.4f}  range=[{ys.min():.3f},{ys.max():.3f}]"
    print(result)
    with open(LOG_PATH, 'a') as lf:
        lf.write('\n' + result + '\n')
        lf.write(f"v1 baseline: MAE=0.0191  RMSE=0.0428  r=0.9783\n")

    freeze_and_export(model, 'spin_pretrain_best.pt', 'spin_pretrain_frozen.pt')

if __name__ == '__main__':
    main()

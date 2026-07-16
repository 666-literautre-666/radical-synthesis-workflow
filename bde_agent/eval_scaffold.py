"""
Scaffold split 评估 — 用现有 v5 checkpoint，按 Murcko 骨架分组评估泛化能力
"""
import torch, numpy as np, pandas as pd, os, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.loader import DataLoader
from gnn_data_utils import mol_to_data
from spin_pretrain import SpinPretrainNN
from train_v5 import BDEGNNv5

N_SAMPLE = 50000
BATCH_SIZE = 128
DATA_PATH = 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== 1. 采样 + 计算 scaffold ========
print(f"Loading {N_SAMPLE} samples + computing Murcko scaffolds...")
df = pd.read_csv(DATA_PATH, nrows=N_SAMPLE)
total = len(df)
print(f"Loaded {total} rows")

scaffold_groups = defaultdict(list)
idx_to_scaff = {}
skipped = 0
t0 = time.time()
for i, (_, row) in enumerate(df.iterrows()):
    if i % 10000 == 0:
        print(f"  Scaffold: {i}/{total}...")
    try:
        mol = Chem.MolFromSmiles(row['molecule'])
        if mol is None:
            skipped += 1; continue
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        scaff_smi = Chem.MolToSmiles(scaff) if scaff.GetNumAtoms() > 0 else 'NO_SCAFFOLD'
        scaffold_groups[scaff_smi].append(i)
        idx_to_scaff[i] = scaff_smi
    except Exception:
        skipped += 1

n_scaffolds = len(scaffold_groups)
n_valid = total - skipped
print(f"Found {n_scaffolds} unique scaffolds from {n_valid} valid molecules ({time.time()-t0:.1f}s)")

sizes = sorted([len(v) for v in scaffold_groups.values()], reverse=True)
print(f"Top 10 scaffold sizes: {sizes[:10]}")
print(f"Singletons (size=1): {sum(1 for s in sizes if s == 1)}")

# ======== 2. Scaffold split: 随机分配骨架到 train/val/test (70/15/15) ========
scaff_list = list(scaffold_groups.keys())
np.random.seed(42)
np.random.shuffle(scaff_list)

n_scaff = len(scaff_list)
n_train_scaff = int(n_scaff * 0.7)
n_val_scaff = int(n_scaff * 0.15)

train_scaff_names = set(scaff_list[:n_train_scaff])
val_scaff_names = set(scaff_list[n_train_scaff:n_train_scaff + n_val_scaff])
test_scaff_names = set(scaff_list[n_train_scaff + n_val_scaff:])

train_idx = [i for s in train_scaff_names for i in scaffold_groups[s]]
val_idx = [i for s in val_scaff_names for i in scaffold_groups[s]]
test_idx = [i for s in test_scaff_names for i in scaffold_groups[s]]

def count_scaffs(indices):
    return len(set(idx_to_scaff.get(i, '?') for i in indices))

train_set = set(train_idx)
test_set = set(test_idx)
train_scaffs = set(idx_to_scaff.get(i, '?') for i in train_idx)
test_scaffs = set(idx_to_scaff.get(i, '?') for i in test_idx)
novel = test_scaffs - train_scaffs

print(f"\nScaffold split:")
print(f"  Train: {len(train_idx):,} mols, {count_scaffs(train_idx):,} scaffolds")
print(f"  Val:   {len(val_idx):,} mols, {count_scaffs(val_idx):,} scaffolds")
print(f"  Test:  {len(test_idx):,} mols, {count_scaffs(test_idx):,} scaffolds")
print(f"  Novel scaffolds in test: {len(novel)}/{len(test_scaffs)} ({len(novel)/max(1,len(test_scaffs))*100:.0f}%)")

# ======== 3. 构建图 ========
print(f"\nBuilding graphs...")
bde_mean = float(df['bde'].mean())
bde_std = float(df['bde'].std())
print(f"BDE mean={bde_mean:.1f} std={bde_std:.1f}")

def build_data_list(indices, name):
    data_list, orig_idx = [], []
    for i in indices:
        row = df.iloc[i]
        bde_norm = (float(row['bde']) - bde_mean) / (bde_std + 1e-8)
        d = mol_to_data(
            smiles=row['molecule'], frag1_smi=str(row.get('fragment1', '')),
            frag2_smi=str(row.get('fragment2', '')),
            bond_idx=int(row['bond_index']), bde_value=bde_norm,
        )
        if d is not None:
            data_list.append(d)
            orig_idx.append(i)
    print(f"  {name}: {len(data_list)} graphs")
    return data_list, orig_idx

va_data, va_orig = build_data_list(val_idx, "Val")
te_data, te_orig = build_data_list(test_idx, "Test")

# ======== 4. 加载 v5 ========
print(f"\nLoading v5 checkpoint...")
spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(DEVICE)
spin_pt = 'spin_pretrain_frozen.pt'
if not os.path.exists(spin_pt):
    spin_pt = os.path.join(os.path.expanduser('~'), 'spin_pretrain_frozen.pt')
spin.load_state_dict(torch.load(spin_pt, map_location=DEVICE, weights_only=True)['backbone'], strict=False)
spin.eval()

ckpt = torch.load('gnn_bde_v5_best.pt', map_location=DEVICE, weights_only=True)
# 旧v5架构: node_dim=10, edge_dim=4, spin_compact_dim=64
model = BDEGNNv5(spin, node_dim=10, edge_dim=4, hidden=256, n_layers=4,
                 dropout=0.0, spin_compact_dim=64, spin_dropout_prob=0.0).to(DEVICE)
# 过滤掉spin.*的key (spin已单独加载), 避免charge_head尺寸不匹配
model_state = {k: v for k, v in ckpt['model'].items() if not k.startswith('spin.')}
model.load_state_dict(model_state, strict=False)
model.eval()

# ======== 5. Evaluate ========
def evaluate_old_v5(loader, name):
    """用旧v5的精确forward逻辑评估, 避免新旧架构不兼容"""
    preds, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            # 旧v5 forward (node_dim=10, edge_dim=4, spin_compact=64)
            out = model.spin(b)
            emb = out[-1]  # node embeddings
            spin_c = model.spin_compact(emb)       # [N, 64]
            x_aug = torch.cat([b.x, spin_c], dim=-1)  # [N, 10+64=74]
            h = model.input_proj(x_aug)
            for conv, norm in zip(model.convs, model.norms):
                h_new = conv(h, b.edge_index, b.edge_attr)
                h = h + model.dropout(norm(h_new).relu())
            spin_r = model.spin_residual(emb)
            gate = torch.sigmoid(model.residual_gate)
            h = h + gate * spin_r
            is_target = (b.x[:, -1] == 3.0)
            t_emb = h[is_target].view(-1, model.hidden * 2)
            bde_pred = model.bde_head(t_emb)
            preds.extend(bde_pred.cpu().squeeze().tolist())
            ys.extend(b.y.cpu().tolist())
    preds = np.array(preds) * bde_std + bde_mean
    ys = np.array(ys) * bde_std + bde_mean
    mae = np.mean(np.abs(preds - ys))
    rmse = np.sqrt(np.mean((preds - ys) ** 2))
    r2 = np.corrcoef(preds, ys)[0, 1] ** 2
    print(f"  {name:6s}: MAE={mae:.2f}  RMSE={rmse:.2f}  R^2={r2:.4f}  N={len(ys)}")
    return mae, rmse, r2, preds, ys

te_loader = DataLoader(te_data, batch_size=BATCH_SIZE)
print(f"\n{'='*60}")
print(f"SCAFFOLD SPLIT RESULTS (v5, {N_SAMPLE} samples)")
print(f"{'='*60}")

mae_te, rmse_te, r2_te, preds_te, ys_te = evaluate_old_v5(te_loader, "Test")

# Per-bond-type breakdown
print(f"\n--- Per Bond Type (Test set, scaffold split) ---")
bond_types = defaultdict(list)
for j, orig_idx in enumerate(te_orig):
    bt = str(df.iloc[orig_idx].get('bond_type', '?'))
    bond_types[bt].append(j)

for bt in sorted(bond_types.keys(), key=lambda x: -len(bond_types[x])):
    indices = bond_types[bt]
    if len(indices) < 20:
        continue
    p = preds_te[indices]
    y = ys_te[indices]
    mae = np.mean(np.abs(p - y))
    print(f"  {bt:8s}: N={len(indices):5d}  MAE={mae:.2f}  range=[{y.min():.0f},{y.max():.0f}]")

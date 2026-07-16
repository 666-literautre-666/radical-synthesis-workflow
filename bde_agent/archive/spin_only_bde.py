"""
纯 Spin MLP: 只用 SpinPretrainNN embedding → MLP → BDE，砍掉全部 GNN
如果效果接近 v5（MAE 1.03），说明"预训练即预测"，叙事完全翻转
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, os, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_data_utils import mol_to_data
from spin_pretrain import SpinPretrainNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz'

# ======== 1. 提取 spin embedding（冻结） + scaffold split ========
print("Loading SpinPretrainNN + scaffold split...")
spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(DEVICE)
spin_pt = 'spin_pretrain_frozen.pt'
if not os.path.exists(spin_pt):
    spin_pt = os.path.join(os.path.expanduser('~'), 'spin_pretrain_frozen.pt')
spin.load_state_dict(torch.load(spin_pt, map_location=DEVICE, weights_only=True)['backbone'], strict=False)
spin.eval()

df = pd.read_csv(DATA_PATH, nrows=50000)

# Scaffold split (same as before)
scaffold_groups = defaultdict(list)
for i, (_, row) in enumerate(df.iterrows()):
    try:
        mol = Chem.MolFromSmiles(row['molecule'])
        if mol is None: continue
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaff) if scaff.GetNumAtoms() > 0 else 'NO_SCAFFOLD'
        scaffold_groups[s].append(i)
    except: pass

scaff_list = list(scaffold_groups.keys())
np.random.seed(42); np.random.shuffle(scaff_list)
n_scaff = len(scaff_list)
n_train = int(n_scaff * 0.7); n_val = int(n_scaff * 0.15)
train_scaffs = set(scaff_list[:n_train])
val_scaffs = set(scaff_list[n_train:n_train + n_val])
test_scaffs = set(scaff_list[n_train + n_val:])
train_idx = [i for s in train_scaffs for i in scaffold_groups[s]]
val_idx = [i for s in val_scaffs for i in scaffold_groups[s]]
test_idx = [i for s in test_scaffs for i in scaffold_groups[s]]
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

# ======== 2. 提取 spin embeddings ========
bde_mean = float(df['bde'].mean())
bde_std = float(df['bde'].std())

def extract_spin_embeddings(indices, name):
    """对每个分子提取目标键两端原子的 spin embedding，返回 [X, y]"""
    X_list, y_list = [], []
    for cnt, i in enumerate(indices):
        if cnt % 5000 == 0:
            print(f"  {name}: {cnt}/{len(indices)}...")
        row = df.iloc[i]
        d = mol_to_data(smiles=row['molecule'], frag1_smi=str(row.get('fragment1', '')),
                         frag2_smi=str(row.get('fragment2', '')),
                         bond_idx=int(row['bond_index']),
                         bde_value=float(row['bde']))  # 原始 BDE，不归一化
        if d is None: continue
        d = d.to(DEVICE)
        with torch.no_grad():
            _, _, emb = spin(d)  # [N, 256]

        # 目标键两个原子的 embedding 取平均
        is_target = (d.x[:, -1] == 3.0)
        target_emb = emb[is_target]  # [2, 256]
        if target_emb.shape[0] != 2:
            continue
        # 两个原子的 embedding concat → 512 维
        feat = target_emb.flatten().cpu().numpy()  # [512]

        bde_raw = float(row['bde'])
        X_list.append(feat)
        y_list.append(bde_raw)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    print(f"  {name}: {len(X)} samples, X shape={X.shape}")
    return X, y

print("\nExtracting spin embeddings...")
X_tr, y_tr = extract_spin_embeddings(train_idx, "Train")
X_va, y_va = extract_spin_embeddings(val_idx, "Val")
X_te, y_te = extract_spin_embeddings(test_idx, "Test")

# 归一化 y
y_mean = float(np.mean(y_tr))
y_std = float(np.std(y_tr))
y_tr_norm = (y_tr - y_mean) / (y_std + 1e-8)
y_va_norm = (y_va - y_mean) / (y_std + 1e-8)
y_te_norm = (y_te - y_mean) / (y_std + 1e-8)

# ======== 3. MLP 训练 ========
class SpinMLP(nn.Module):
    def __init__(self, in_dim=512, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, x):
        return self.net(x)

print(f"\n{'='*60}")
print(f"Pure Spin MLP: 512-dim spin embedding -> MLP -> BDE")
print(f"{'='*60}")
print(f"Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
print(f"y_mean={y_mean:.1f}  y_std={y_std:.1f}")

tr_dataset = torch.utils.data.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr_norm))
va_dataset = torch.utils.data.TensorDataset(torch.tensor(X_va), torch.tensor(y_va_norm))
te_dataset = torch.utils.data.TensorDataset(torch.tensor(X_te), torch.tensor(y_te_norm))
tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=256, shuffle=True)
va_loader = torch.utils.data.DataLoader(va_dataset, batch_size=256)
te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=256)

model = SpinMLP(in_dim=X_tr.shape[1], hidden=256, dropout=0.3).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

mse = nn.MSELoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-6)
best_va, best_ep = float('inf'), 0

for ep in range(200):
    model.train()
    tr_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = mse(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item()
    tr_loss /= len(tr_loader)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in va_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            va_loss += mse(model(xb), yb).item()
    va_loss /= len(va_loader)
    sch.step()

    if va_loss < best_va:
        best_va, best_ep = va_loss, ep
        torch.save(model.state_dict(), 'spin_mlp_best.pt')

    if ep % 20 == 0 or ep < 5:
        print(f"E{ep:3d}  tr={tr_loss:.4f}  va={va_loss:.4f}  best={best_va:.4f}@{best_ep}")

    if ep - best_ep > 50:
        print(f"Early stop at {ep}"); break

# ======== 4. 测试 ========
print(f"\n{'='*60}")
model.load_state_dict(torch.load('spin_mlp_best.pt', weights_only=True))
model.eval()
preds, ys = [], []
with torch.no_grad():
    for xb, yb in te_loader:
        xb = xb.to(DEVICE)
        p = model(xb).cpu().numpy()
        preds.extend((p * y_std + y_mean).flatten().tolist())
        ys.extend((yb.numpy() * y_std + y_mean).flatten().tolist())

preds = np.array(preds); ys = np.array(ys)
mae = np.mean(np.abs(preds - ys))
rmse = np.sqrt(np.mean((preds - ys)**2))
r2 = np.corrcoef(preds, ys)[0, 1]**2

print(f"Pure Spin MLP (NO GNN):")
print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R^2={r2:.4f}")
print(f"\nComparison:")
print(f"  v5 (spin + GNN):     MAE 2.78  R^2 0.9923")
print(f"  Spin-only MLP:       MAE {mae:.2f}  R^2 {r2:.4f}")
print(f"  v5-no-spin (zeroed): MAE 11.18 R^2 0.3328")

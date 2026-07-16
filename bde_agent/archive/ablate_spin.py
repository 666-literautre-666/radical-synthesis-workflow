"""
Ablation: v5 with vs without spin features on scaffold split
直接把 spin_compact 输出置零，其他权重不变，同一个 test set 跑两遍
"""
import torch, numpy as np, os, sys, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.loader import DataLoader
from gnn_data_utils import mol_to_data
from spin_pretrain import SpinPretrainNN
from train_v5 import BDEGNNv5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz'

# ======== 1. 加载 scaffold split 的 test 数据（复用之前的划分逻辑）=======
print("Loading 50K + scaffold split...")
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

df = pd.read_csv(DATA_PATH, nrows=50000)
scaffold_groups = defaultdict(list)
idx_to_scaff = {}
for i, (_, row) in enumerate(df.iterrows()):
    try:
        mol = Chem.MolFromSmiles(row['molecule'])
        if mol is None: continue
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaff) if scaff.GetNumAtoms() > 0 else 'NO_SCAFFOLD'
        scaffold_groups[s].append(i)
        idx_to_scaff[i] = s
    except: pass

scaff_list = list(scaffold_groups.keys())
np.random.seed(42)
np.random.shuffle(scaff_list)
n_scaff = len(scaff_list)
n_train = int(n_scaff * 0.7)
n_val = int(n_scaff * 0.15)
test_scaff_names = set(scaff_list[n_train + n_val:])
test_idx = [i for s in test_scaff_names for i in scaffold_groups[s]]
print(f"Test: {len(test_idx)} mols, {len(test_scaff_names)} novel scaffolds")

# 构建 test graphs
bde_mean = float(df['bde'].mean())
bde_std = float(df['bde'].std())
test_data, test_orig = [], []
for i in test_idx:
    row = df.iloc[i]
    bde_norm = (float(row['bde']) - bde_mean) / (bde_std + 1e-8)
    d = mol_to_data(smiles=row['molecule'], frag1_smi=str(row.get('fragment1', '')),
                     frag2_smi=str(row.get('fragment2', '')),
                     bond_idx=int(row['bond_index']), bde_value=bde_norm)
    if d is not None:
        test_data.append(d)
        test_orig.append(i)
print(f"Test graphs: {len(test_data)}")

# ======== 2. 加载 v5 checkpoint ========
print("\nLoading v5 model...")
spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(DEVICE)
spin_pt = 'spin_pretrain_frozen.pt'
if not os.path.exists(spin_pt):
    spin_pt = os.path.join(os.path.expanduser('~'), 'spin_pretrain_frozen.pt')
spin.load_state_dict(torch.load(spin_pt, map_location=DEVICE, weights_only=True)['backbone'], strict=False)
spin.eval()

ckpt = torch.load('gnn_bde_v5_best.pt', map_location=DEVICE, weights_only=True)
model = BDEGNNv5(spin, node_dim=10, hidden=256, n_layers=4, dropout=0.0).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

# ======== 3. Evaluate: with spin vs without spin ========
def evaluate_with_flag(use_spin):
    """use_spin=True: 正常 v5; use_spin=False: spin_compact 输出置零"""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in DataLoader(test_data, batch_size=128):
            b = b.to(DEVICE)
            spin_pseudo, charge_pseudo, emb = model.spin(b)

            if use_spin:
                spin_c = model.spin_compact(emb)
            else:
                spin_c = torch.zeros(emb.shape[0], 64, device=DEVICE)

            x_aug = torch.cat([b.x, spin_c], dim=-1)
            h = model.input_proj(x_aug)
            for conv, norm in zip(model.convs, model.norms):
                h_new = conv(h, b.edge_index, b.edge_attr)
                h = h + model.dropout(norm(h_new).relu())

            spin_r = model.spin_residual(emb)
            gate = torch.sigmoid(model.residual_gate) if use_spin else torch.tensor(0.0, device=DEVICE)
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
    r2 = np.corrcoef(preds, ys)[0, 1] ** 2 if len(preds) > 2 else 0
    return mae, rmse, r2

print(f"\n{'='*60}")
print(f"ABLATION: spin features contribution on scaffold split")
print(f"{'='*60}")

mae_with, rmse_with, r2_with = evaluate_with_flag(True)
print(f"With spin:     MAE={mae_with:.2f}  RMSE={rmse_with:.2f}  R^2={r2_with:.4f}")

mae_without, rmse_without, r2_without = evaluate_with_flag(False)
print(f"Without spin:  MAE={mae_without:.2f}  RMSE={rmse_without:.2f}  R^2={r2_without:.4f}")

delta = mae_without - mae_with
print(f"\nSpin contribution: {delta:+.2f} kcal MAE ({delta/mae_without*100:+.1f}%)")
print(f"R^2 delta:         {r2_with - r2_without:+.4f}")

"""验证 ablation: v5 with spin vs without spin 的真实结果"""
import torch, numpy as np, os, sys, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.logger().setLevel(RDLogger.ERROR)
import pandas as pd
from torch_geometric.loader import DataLoader
from gnn_data_utils import mol_to_data
from spin_pretrain import SpinPretrainNN
from train_v5 import BDEGNNv5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz'

print("Loading 50K + scaffold split...")
df = pd.read_csv(DATA_PATH, nrows=50000)
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
test_scaff_names = set(scaff_list[n_train + n_val:])
test_idx = [i for s in test_scaff_names for i in scaffold_groups[s]]

bde_mean = float(df['bde'].mean()); bde_std = float(df['bde'].std())
test_data = []
for i in test_idx:
    row = df.iloc[i]
    bde_norm = (float(row['bde']) - bde_mean) / (bde_std + 1e-8)
    d = mol_to_data(smiles=row['molecule'], frag1_smi=str(row.get('fragment1', '')),
                     frag2_smi=str(row.get('fragment2', '')),
                     bond_idx=int(row['bond_index']), bde_value=bde_norm)
    if d is not None: test_data.append(d)
print(f"Test graphs: {len(test_data)}")

print("Loading v5 checkpoint...")
spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(DEVICE)
spin_pt = 'spin_pretrain_frozen.pt'
if not os.path.exists(spin_pt):
    spin_pt = os.path.join(os.path.expanduser('~'), 'spin_pretrain_frozen.pt')
spin.load_state_dict(torch.load(spin_pt, map_location=DEVICE, weights_only=True)['backbone'], strict=False)
spin.eval()

ckpt = torch.load('gnn_bde_v5_best.pt', map_location=DEVICE, weights_only=True)
model = BDEGNNv5(spin, node_dim=10, edge_dim=4, hidden=256, n_layers=4,
                 dropout=0.0, spin_compact_dim=64, spin_dropout_prob=0.0).to(DEVICE)
model_state = {k: v for k, v in ckpt['model'].items() if not k.startswith('spin.')}
model.load_state_dict(model_state, strict=False)
model.eval()

def run_eval(use_spin):
    preds, ys = [], []
    with torch.no_grad():
        for b in DataLoader(test_data, batch_size=128):
            b = b.to(DEVICE)
            out = model.spin(b); emb = out[-1]
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
print(f"ABLATION VERIFICATION")
print(f"{'='*60}")

mae_w, rmse_w, r2_w = run_eval(True)
print(f"With spin:     MAE={mae_w:.2f}  RMSE={rmse_w:.2f}  R^2={r2_w:.4f}")

mae_wo, rmse_wo, r2_wo = run_eval(False)
print(f"Without spin:  MAE={mae_wo:.2f}  RMSE={rmse_wo:.2f}  R^2={r2_wo:.4f}")

print(f"\nPreviously reported: With=2.78, Without=11.18")
print(f"Verified now:        With={mae_w:.2f}, Without={mae_wo:.2f}")
print(f"Match: {'YES' if abs(mae_w-2.78)<0.1 and abs(mae_wo-11.18)<0.1 else 'CHECK DIFF'}")

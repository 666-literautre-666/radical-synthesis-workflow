"""Quick evaluation of saved best model"""
import torch, numpy as np, os, warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from gnn_train import BDEGNNv3
from gnn_data_utils import load_gnn_data
from config import CONFIG
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ckpt = torch.load('gnn_bde_best_v2.pt', map_location=device)
print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Best val loss: {ckpt.get('best_val', 'N/A'):.4f}")
print(f"Best epoch: {ckpt.get('best_epoch', 'N/A')}")

bde_mean = ckpt['bde_mean']
bde_std = ckpt['bde_std']
print(f"BDE mean={bde_mean:.1f}, std={bde_std:.1f}")

model = BDEGNNv3(
    node_dim=10, edge_dim=4, phys_dim=12,
    hidden=CONFIG['gnn_hidden'],
    n_layers=CONFIG['gnn_layers'],
    dropout=CONFIG['dropout'],
).to(device)
model.load_state_dict(ckpt['model'])

# Load test data — use 10K for fast eval (same split as training)
nrows = 10000
print(f"Loading {nrows} molecules for test evaluation...")
data_list, _, _ = load_gnn_data(CONFIG['data_path'], nrows=nrows)
idx = list(range(len(data_list)))
_, temp_idx = train_test_split(idx, test_size=0.2, random_state=42)
_, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
test_data = [data_list[i] for i in test_idx]
print(f"Test samples: {len(test_data)}")

test_loader = DataLoader(test_data, batch_size=CONFIG['batch_size'])

model.eval()
all_preds, all_y = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        all_preds.extend(pred.cpu().squeeze().tolist())
        all_y.extend(batch.y.cpu().tolist())

preds_arr = np.array([p * bde_std + bde_mean for p in all_preds])
y_arr = np.array([y * bde_std + bde_mean for y in all_y])

mae = np.mean(np.abs(preds_arr - y_arr))
rmse = np.sqrt(np.mean((preds_arr - y_arr) ** 2))
r2 = np.corrcoef(preds_arr, y_arr)[0, 1] ** 2

print(f"\n=== EVALUATION RESULTS ===")
print(f"MAE:  {mae:.2f} kcal/mol")
print(f"RMSE: {rmse:.2f} kcal/mol")
print(f"R2:   {r2:.4f}")
print(f"BDE range: {y_arr.min():.0f}-{y_arr.max():.0f} kcal/mol")

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from config import CONFIG
from data_utils import load_data, N_BOND_TYPES

# ======== 1. 载入数据 ========
print("Loading data...")
nrows = CONFIG.get('nrows', 800000)
X, y = load_data(CONFIG['data_path'], nBits=CONFIG['nBits'], nrows=nrows)
INPUT_DIM = CONFIG['nBits'] + N_BOND_TYPES  # 指纹 + 键型编码
print(f"Features: {CONFIG['nBits']} (fp) + {N_BOND_TYPES} (bond) = {INPUT_DIM}")
print(f"Loaded {len(X)} rows")

# ======== 2. 划分 train/val/test ========
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ======== 3. 转 tensor + 归一化 ========
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

mu = X_train.mean(0); std = X_train.std(0) + 1e-8
X_train = (X_train - mu) / std
X_val = (X_val - mu) / std
X_test = (X_test - mu) / std

# ======== 4. 定义模型 ========
layers = []
in_dim = INPUT_DIM
for h in CONFIG['hidden_layers']:
    layers.append(nn.Linear(in_dim, h))
    layers.append(nn.Dropout(CONFIG['dropout']))
    layers.append(nn.ReLU())
    in_dim = h
layers.append(nn.Linear(in_dim, 1))

model = nn.Sequential(*layers)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

# ======== 5. 训练 ========
print("Training...")
for epoch in range(CONFIG['epochs']):
    model.train()
    pred = model(X_train)
    train_loss = loss_fn(pred, y_train)
    opt.zero_grad(); train_loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d}  Train: {train_loss.item():.2f}  Val: {val_loss.item():.2f}")

# ======== 6. 测试 + 保存 ========
model.eval()
with torch.no_grad():
    test_loss = loss_fn(model(X_test), y_test)
print(f"\nTest Loss: {test_loss.item():.2f}  MAE: {test_loss.item()**0.5:.1f} kcal/mol")

torch.save(model.state_dict(), 'bde_model.pt')
print("Model saved: bde_model.pt")
import torch, torch.nn as nn
import numpy as np
from config import CONFIG
from data_utils import fp

# ======== 1. 重建模型结构 ========
layers = []
in_dim = CONFIG['nBits']
for h in CONFIG['hidden_layers']:
    layers.append(nn.Linear(in_dim, h))
    layers.append(nn.Dropout(CONFIG['dropout']))
    layers.append(nn.ReLU())
    in_dim = h
layers.append(nn.Linear(in_dim, 1))
model = nn.Sequential(*layers)

# ======== 2. 加载训练好的权重 ========
model.load_state_dict(torch.load('bde_model.pt'))

# ======== 3. 预测函数 ========
def predict_bde(smiles):
    mol_fp = fp(smiles, CONFIG['nBits'])
    X = torch.tensor([mol_fp], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(X)
    return pred.item()

# ======== 4. 测试 ========
if __name__ == '__main__':
    test_smiles = [
        'Cc1ccccc1',       # 甲苯 benzylic ~88
        'C=CC',             # 丙烯 allylic ~88
        'CC=O',             # 乙醛 aldehyde ~87
        'BrCc1ccccc1',      # 溴化苄 C-Br ~68
        'CC(C)C',           # 异丁烷 secondary ~99
    ]
    print("=== BDE Prediction ===\n")
    for s in test_smiles:
        print(f'{s:20s}  BDE = {predict_bde(s):6.1f} kcal/mol')

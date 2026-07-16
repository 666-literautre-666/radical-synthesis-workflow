import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 常见键型——固定顺序，编码时用
BOND_TYPES = ['C-H','C-C','C=C','C-Br','C-I','C-Cl','C-S','C-F',
              'N-O','O-H','N-H','C-O','C-N','C-P']
BOND_ENCODER = {b: i for i, b in enumerate(BOND_TYPES)}
N_BOND_TYPES = len(BOND_TYPES)


def fp(smi, nBits=256):
    """分子 → Morgan 指纹 → NumPy 数组"""
    mol = Chem.MolFromSmiles(smi)
    f = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    a = np.zeros(nBits)
    AllChem.DataStructs.ConvertToNumpyArray(f, a)
    return a


def load_data(csv_path, nBits=256, nrows=None):
    """
    读 CSV,返回特征矩阵 X 和 BDE 向量 y。

    特征 = Morgan 指纹 (nBits 位)
          + 键型 one-hot 编码 (N_BOND_TYPES 位)
    共 nBits + N_BOND_TYPES 位
    """
    df = pd.read_csv(csv_path, nrows=nrows)

    X_list = []
    for _, row in df.iterrows():
        # 分子指纹
        fp_vec = fp(row['molecule'], nBits=nBits)
        # 键型 one-hot
        bt_onehot = np.zeros(N_BOND_TYPES)
        bt = row.get('bond_type', '')
        if bt in BOND_ENCODER:
            bt_onehot[BOND_ENCODER[bt]] = 1.0
        # 拼接
        X_list.append(np.concatenate([fp_vec, bt_onehot]))

    X = np.array(X_list, dtype=np.float32)
    y = df['bde'].values.reshape(-1, 1).astype(np.float32)
    return X, y

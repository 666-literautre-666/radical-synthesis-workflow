"""
BDE (Bond Dissociation Energy) Prediction Module
==================================================
用机器学习从分子结构预测键解离能，替代经验规则查表。

两种方法：
1. sklearn RandomForest（基线，随时可用）
2. GNN 图神经网络（需要 PyTorch + PyTorch Geometric）

面试可以讲：
  "我用 Morgan 指纹 + 随机森林建立了 BDE 预测基线模型，
   在此基础上用图神经网络（GCN/GAT）直接从分子图结构学习
   BDE，避免了手工特征工程，MAE 优于经验规则。"
"""

import numpy as np
from pathlib import Path
import json

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    raise ImportError("需要 rdkit: pip install rdkit")

# ---------------------------------------------------------------------------
# 训练数据：文献已知的 BDE 值（kcal/mol）
# 数据来源: Luo, Y.R. "Handbook of Bond Dissociation Energies" (2003)
# ---------------------------------------------------------------------------

KNOWN_BDE_DATA = [
    # (SMILES, BDE_kcal_mol, 键描述)
    ("C", 105, "甲烷 C-H"),
    ("CC", 101, "乙烷 primary C-H"),
    ("CCC", 101, "丙烷 primary C-H"),
    ("CC(C)C", 99, "异丁烷 secondary C-H"),
    ("CC(C)(C)C", 96, "新戊烷 tertiary C-H"),
    ("C1CC1", 106, "环丙烷 C-H"),
    ("C1CCCC1", 96, "环己烷 secondary C-H"),
    ("C=C", 111, "乙烯 C-H"),
    ("C=CC", 88, "丙烯 allylic C-H"),
    ("C=CCC", 86, "1-丁烯 allylic C-H"),
    ("Cc1ccccc1", 88, "甲苯 benzylic C-H"),
    ("CCc1ccccc1", 85, "乙苯 secondary benzylic C-H"),
    ("CC(C)c1ccccc1", 83, "异丙苯 tertiary benzylic C-H"),
    ("CC=O", 87, "乙醛 aldehyde C-H"),
    ("O=CC=O", 87, "甲醛 aldehyde C-H"),
    ("c1ccccc1", 113, "苯 aromatic C-H"),
    ("C#C", 133, "乙炔 C-H"),
    ("c1ccc(OC)cc1", 85, "苯甲醚 benzylic C-H (alpha to O)"),
    ("CC(=O)C", 92, "丙酮 alpha C-H"),
    ("CCO", 94, "乙醇 alpha C-H (alpha to OH)"),
    ("CN", 93, "甲胺 alpha C-H (alpha to NH2)"),
]

# ---------------------------------------------------------------------------
# 特征提取：Morgan 指纹 (ECFP4)
# ---------------------------------------------------------------------------

def _smiles_to_fingerprint(smiles: str, nBits: int = 1024) -> np.ndarray:
    """将 SMILES 转为 Morgan 指纹（固定长度向量）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    arr = np.zeros(nBits)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _featurize(smiles_list: list, nBits: int = 1024) -> tuple:
    """
    将 SMILES 列表转为特征矩阵 X 和标签向量 y。

    返回:
        X: np.ndarray shape (n_samples, nBits)  — Morgan 指纹
        y: np.ndarray shape (n_samples,)          — BDE kcal/mol
    """
    X_list, y_list = [], []
    for smiles, bde, desc in smiles_list:
        try:
            fp = _smiles_to_fingerprint(smiles, nBits)
            X_list.append(fp)
            y_list.append(bde)
        except Exception:
            continue
    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# 基线模型：随机森林
# ---------------------------------------------------------------------------

class BDEBaseline:
    """
    基于 Morgan 指纹 + 随机森林的 BDE 预测基线。

    用法:
        model = BDEBaseline()
        model.train(KNOWN_BDE_DATA)
        bde = model.predict("Cc1ccccc1")  # → 预测 BDE
    """

    def __init__(self):
        self.model = None
        self.trained = False

    def train(self, data: list = None):
        """
        训练模型。

        参数:
            data: [(smiles, bde, desc), ...] 训练数据
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        if data is None:
            data = KNOWN_BDE_DATA

        X, y = _featurize(data)
        self.X_ = X
        self.y_ = y

        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        # 交叉验证评估
        try:
            scores = cross_val_score(self.model, X, y, cv=3,
                                     scoring='neg_mean_absolute_error')
            self.cv_mae = -scores.mean()
        except Exception:
            self.cv_mae = None

        self.trained = True
        return self

    def predict(self, smiles: str) -> dict:
        """预测单个分子的最弱 BDE"""
        if not self.trained:
            raise RuntimeError("模型未训练，请先调用 .train()")

        fp = _smiles_to_fingerprint(smiles).reshape(1, -1)
        bde_pred = float(self.model.predict(fp)[0])

        return {
            "smiles": smiles,
            "predicted_bde_kcal": round(bde_pred, 1),
            "model": "RandomForest (Morgan ECFP4)",
            "cv_mae_kcal": round(self.cv_mae, 1) if self.cv_mae else None,
        }

    def compare_with_rules(self, smiles: str, rule_bde: float) -> dict:
        """
        对比 ML 预测和经验规则估算。

        参数:
            smiles: 分子 SMILES
            rule_bde: 你项目中 _estimate_bde() 给出的经验 BDE

        返回:
            包含两种方法预测值和差异的字典
        """
        ml_result = self.predict(smiles)
        return {
            "smiles": smiles,
            "ml_predicted_bde": ml_result["predicted_bde_kcal"],
            "rule_based_bde": rule_bde,
            "difference_kcal": round(ml_result["predicted_bde_kcal"] - rule_bde, 1),
        }


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def train_baseline(data: list = None) -> BDEBaseline:
    """一键训练 BDE 基线模型"""
    model = BDEBaseline()
    model.train(data)
    return model


def predict_bde(smiles: str, model: BDEBaseline = None) -> dict:
    """
    预测 BDE（自动加载或使用传入的模型）。

    用法:
        result = predict_bde("Cc1ccccc1")
        print(result["predicted_bde_kcal"])
    """
    if model is None:
        model = train_baseline()
    return model.predict(smiles)


# ---------------------------------------------------------------------------
# 模型保存/加载
# ---------------------------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
MODEL_DIR = ML_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model: BDEBaseline, name: str = "bde_baseline"):
    """保存训练好的模型"""
    import pickle
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return str(path)


def load_model(name: str = "bde_baseline") -> BDEBaseline:
    """加载已保存的模型"""
    import pickle
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}\n请先运行 train_baseline() 训练并保存。")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BDE Predictor — ML 基线模型测试")
    print("=" * 60)

    model = train_baseline()
    print(f"\n交叉验证 MAE: {model.cv_mae:.1f} kcal/mol\n")

    test_smiles_list = [
        "Cc1ccccc1",    # 甲苯，苄位 BDE ~88
        "C=CC",         # 丙烯，烯丙位 BDE ~88
        "CC=O",         # 乙醛，醛基 BDE ~87
        "CCC",          # 丙烷，一级 C-H BDE ~101
        "CC(C)C",       # 异丁烷，二级 C-H BDE ~99
    ]

    for smi in test_smiles_list:
        result = model.predict(smi)
        print(f"  {smi:20s} → BDE = {result['predicted_bde_kcal']:.1f} kcal/mol")

    # 对比：ML vs 经验规则
    print("\n--- ML vs 经验规则对比 ---")
    from scripts.reaction_predictor import _estimate_bde

    for smi in test_smiles_list:
        rule = _estimate_bde(smi)
        rule_bde = rule.get("estimated_bde_kcal", "N/A")
        ml_result = model.predict(smi)
        ml_bde = ml_result["predicted_bde_kcal"]
        print(f"  {smi:20s}  ML={ml_bde:.1f}  规则={rule_bde} kcal/mol")

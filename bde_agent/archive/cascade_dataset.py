"""
级联图网络数据集 — 分子图 + 全局物理特征 + 环境张量 + 多监督标签

每个 Data 对象包含:
  - 图特征: x, edge_index, edge_attr
  - 全局物理: num_resonance, conj_length, dynamic_shielding_score
  - 环境张量: env_tensor [T, epsilon, viscosity, p_O2]
  - 目标边索引: target_edge_pair  (两个方向的索引)
  - 标签: y_bde, y_spin (per-atom), y_delta_est (可选)
"""
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, ResonanceMolSupplier


# ---- 环境归一化参数 (离线统计) ----
ENV_STATS = {
    'T':       {'mean': 350.0, 'std': 75.0},    # 开尔文
    'epsilon': {'mean': 20.0,  'std': 15.0},    # 介电常数
    'viscosity':{'mean': 0.5,   'std': 0.4},    # 粘度 mPa·s
    'p_O2':    {'mean': 0.15,  'std': 0.1},     # 氧气分压 atm
}


def _compute_num_resonance(mol):
    """计算共振式数目"""
    try:
        res = ResonanceMolSupplier(mol)
        return float(min(len(res), 100))  # cap at 100
    except Exception:
        return 0.0


def _compute_conj_length(mol):
    """计算最长共轭链长度 (交替单双键路径)"""
    max_len = 0
    n = mol.GetNumAtoms()
    # 建邻接表 (只保留 conjugated bonds)
    conj = {i: set() for i in range(n)}
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            conj[i].add(j); conj[j].add(i)

    visited = set()
    def dfs(u, depth):
        visited.add(u)
        cur = depth
        for v in conj[u]:
            if v not in visited:
                cur = max(cur, dfs(v, depth + 1))
        visited.remove(u)
        return cur

    for start in range(n):
        max_len = max(max_len, dfs(start, 1))
    return float(max_len)


def _compute_shielding_score(mol):
    """动态遮蔽分数代理: 基于立体位阻估算 (0-1 之间)"""
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return 0.0
    # 用环内原子占比 + 季碳占比 作为位阻代理
    ring_atoms = sum(1 for a in mol.GetAtoms() if a.IsInRing())
    quaternary = sum(1 for a in mol.GetAtoms() if a.GetDegree() >= 4 and a.GetAtomicNum() == 6)
    score = (ring_atoms + quaternary) / n_atoms
    return float(np.clip(score, 0.0, 1.0))


def normalize_env(env_raw):
    """Z-Score 归一化环境张量, 附带温度开尔文防御"""
    T, eps, visc, po2 = env_raw
    assert T >= 100, f"温度必须为开尔文! 当前 T={T} < 100K"
    return torch.tensor([
        (T - ENV_STATS['T']['mean']) / ENV_STATS['T']['std'],
        (eps - ENV_STATS['epsilon']['mean']) / ENV_STATS['epsilon']['std'],
        (visc - ENV_STATS['viscosity']['mean']) / ENV_STATS['viscosity']['std'],
        (po2 - ENV_STATS['p_O2']['mean']) / ENV_STATS['p_O2']['std'],
    ], dtype=torch.float32)


def mol_to_cascade_data(smiles, bond_idx, bde_value, env_raw=None):
    """
    构建级联 Data 对象.

    Args:
        smiles: 分子 SMILES
        bond_idx: 目标键索引 (RDKit bond index)
        bde_value: BDE 标签 (kcal/mol)
        env_raw: [T, epsilon, visc, p_O2], 默认标准条件

    Returns:
        PyG Data 或 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    # ---- 节点特征 (10维, 同 v2) ----
    node_feats = []
    for atom in mol.GetAtoms():
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
            0.0,  # fragment label (暂不用)
        ])

    x = torch.tensor(node_feats, dtype=torch.float32)

    # ---- 边特征 (4维) + 目标边标记 ----
    ei0, ei1, ea = [], [], []
    is_target_edge = []  # 每条有向边是否为我们要预测的目标边
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bid = bond.GetIdx()

        ei0.extend([i, j]); ei1.extend([j, i])
        bt = bond.GetBondType()
        f = [float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE),
             float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC)]
        ea.extend([f, f])

        is_target = (bid == bond_idx)
        is_target_edge.extend([is_target, is_target])

    edge_index = torch.tensor([ei0, ei1], dtype=torch.long)
    edge_attr = torch.tensor(ea, dtype=torch.float32)
    target_mask = torch.tensor(is_target_edge, dtype=torch.bool)

    if not any(is_target_edge):
        return None

    # ---- 全局物理特征 ----
    num_res = _compute_num_resonance(mol)
    conj_len = _compute_conj_length(mol)
    shield = _compute_shielding_score(mol)

    # ---- 环境张量 ----
    if env_raw is None:
        env_raw = [298.0, 1.0, 0.0, 0.21]  # 标准气相条件
    env_tensor = normalize_env(env_raw)

    # ---- 标签 ----
    y_bde = torch.tensor([bde_value], dtype=torch.float32)

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        num_resonance=torch.tensor([num_res], dtype=torch.float32),
        conj_length=torch.tensor([conj_len], dtype=torch.float32),
        shielding=torch.tensor([shield], dtype=torch.float32),
        env_tensor=env_tensor,
        target_mask=target_mask,
        y_bde=y_bde,
    )


if __name__ == '__main__':
    # 冒烟测试
    d = mol_to_cascade_data("Cc1ccccc1", bond_idx=6, bde_value=85.0)
    if d:
        print(f"atoms={d.x.shape[0]}, edges={d.edge_index.shape[1]//2}")
        print(f"target_edge_pair={d.target_edge.tolist()}")
        print(f"num_res={d.num_resonance.item():.0f}, conj={d.conj_length.item():.0f}, shield={d.shielding.item():.3f}")
        print(f"env={d.env_tensor.tolist()}")
        print("冒烟通过")
    else:
        print("FAILED")

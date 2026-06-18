"""GNN v2 — 显式加氢 + 子结构匹配 + 边界原子标记 + 边特征"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem

N_BOND_TYPES = 14


def mol_to_data(smiles, frag1_smi, frag2_smi, bond_idx, bde_value):
    """显式加氢建图，含碎片标签、边界原子标记、边特征。"""
    mol_implicit = Chem.MolFromSmiles(smiles)
    if mol_implicit is None:
        return None

    # 显式加氢（C-H 键才能被 RDKit 索引）
    mol = Chem.AddHs(mol_implicit)
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol_implicit.GetNumAtoms()

    # 获取目标键
    try:
        bond = mol.GetBondWithIdx(bond_idx)
        target_atoms = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    except (IndexError, AttributeError):
        return None

    # ---- 碎片匹配（在隐式氢分子上做，碎片没有显式 H）----
    atom_labels = [0.0] * n_atoms
    frag1 = Chem.MolFromSmiles(frag1_smi) if frag1_smi and frag1_smi != 'nan' else None
    frag2 = Chem.MolFromSmiles(frag2_smi) if frag2_smi and frag2_smi != 'nan' else None

    if frag1 is not None and frag2 is not None:
        # 给碎片也加氢
        frag1_h = Chem.AddHs(frag1)
        frag2_h = Chem.AddHs(frag2)
        matches1 = mol.GetSubstructMatches(frag1_h)
        matches2 = mol.GetSubstructMatches(frag2_h)
        f1_set = set(matches1[0]) if matches1 else set()
        f2_set = set(matches2[0]) if matches2 else set()

        both = f1_set & f2_set
        f1_set -= both
        f2_set -= both

        for i in f1_set:
            atom_labels[i] = 1.0
        for i in f2_set:
            atom_labels[i] = 2.0

    # 目标键端点标记为 3.0（覆盖碎片标记）
    for i in target_atoms:
        atom_labels[i] = 3.0

    # ---- 节点特征 (10维) ----
    node_feats = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        f = [
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.GetFormalCharge()),
            float(atom.GetHybridization() == Chem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            float(atom.IsInRing()),
            atom_labels[idx],
        ]
        node_feats.append(f)

    x = torch.tensor(node_feats, dtype=torch.float32)

    # ---- 边特征 (4维) ----
    edge_idx = [[], []]
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_idx[0].extend([i, j])
        edge_idx[1].extend([j, i])
        bt = b.GetBondType()
        feats = [
            float(bt == Chem.BondType.SINGLE),
            float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE),
            float(bt == Chem.BondType.AROMATIC),
        ]
        edge_attr.extend([feats, feats])

    edge_index = torch.tensor(edge_idx, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor([bde_value], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t, y=y)


def load_gnn_data(csv_path, nrows=None):
    """读取 CSV → PyG Data list + 归一化参数"""
    df = pd.read_csv(csv_path, nrows=nrows)
    bde_all = df['bde'].values.astype(float)
    bde_mean = float(np.mean(bde_all))
    bde_std = float(np.std(bde_all))

    data_list = []
    skipped = 0
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  Loading: {i}/{total} molecules...")
        bde_norm = (float(row['bde']) - bde_mean) / (bde_std + 1e-8)
        bond_idx = int(row['bond_index'])

        d = mol_to_data(
            smiles=row['molecule'],
            frag1_smi=str(row.get('fragment1', '')),
            frag2_smi=str(row.get('fragment2', '')),
            bond_idx=bond_idx,
            bde_value=bde_norm,
        )
        if d is not None:
            data_list.append(d)
        else:
            skipped += 1

    print(f"Loaded {len(data_list)} graphs (skipped {skipped})")
    return data_list, bde_mean, bde_std


if __name__ == '__main__':
    data_list, mean, std = load_gnn_data(
        'C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz',
        nrows=5)
    for i, d in enumerate(data_list):
        n_target = int((d.x[:, -1] == 3.0).sum())
        print(f"Mol {i}: atoms={d.x.shape[0]}, edges={d.edge_index.shape[1]//2}, "
              f"target_atoms={n_target}, BDE={d.y.item():.1f}")

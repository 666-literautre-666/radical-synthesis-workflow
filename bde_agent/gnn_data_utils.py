"""GNN v2 — 显式加氢 + 子结构匹配 + 边界原子标记 + 边特征 + 物理描述符"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Crippen, rdMolChemicalFeatures

N_BOND_TYPES = 14

# Pauling electronegativity (常见元素)
ELECTRONEG = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
# 共价半径 (pm)
COV_RADII = {1: 31, 6: 76, 7: 71, 8: 66, 9: 57, 15: 107, 16: 105, 17: 102, 35: 120, 53: 139}


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
    if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
        return None
    try:
        bond = mol.GetBondWithIdx(bond_idx)
        target_atoms = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    except (IndexError, AttributeError, RuntimeError):
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

    # ---- 全局物理特征 (12维，每分子一组) ----
    # Gasteiger 电荷
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(a.GetDoubleProp('_GasteigerCharge')) for a in mol.GetAtoms()]

    # 自由基稳定化位点计数
    benzylic_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[c][CH2,CH]')))
    allylic_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]=[C][CH2,CH]')))
    carbonyl_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[CH2,CH]')))

    # 未成对电子代理: fragment1 vs fragment2 电荷差
    # （简化: 有 Br/I/Cl 时可能产生自由基 → 1 或 2 个未成对电子）
    radical_proxy = 1 if any(smi in Chem.MolToSmiles(mol) for smi in ['Br','I','Cl','O=','N=']) else 0

    # 共轭长度代理: 最长连续 sp2 链
    conj_length = 0
    visited = set()
    for atom in mol.GetAtoms():
        if atom.GetIdx() in visited or not atom.GetIsAromatic():
            continue
        stack = [atom.GetIdx()]
        local_len = 0
        while stack:
            aidx = stack.pop()
            if aidx in visited:
                continue
            visited.add(aidx)
            local_len += 1
            for nb in mol.GetAtomWithIdx(aidx).GetNeighbors():
                if nb.GetIsAromatic() and nb.GetIdx() not in visited:
                    stack.append(nb.GetIdx())
        conj_length = max(conj_length, local_len)

    phys_feats = torch.tensor([
        float(len(mol.GetAtoms())),       # 1. 总原子数
        float(max(charges)),               # 2. 最大正电荷
        float(min(charges)),               # 3. 最小负电荷
        float(np.mean(charges)),           # 4. 平均电荷
        float(benzylic_hit),               # 5. 苄基位点数
        float(allylic_hit),                # 6. 烯丙基位点数
        float(carbonyl_hit),               # 7. α-羰基位点数
        float(conj_length),                # 8. 共轭长度
        float(radical_proxy),              # 9. 未成对电子代理
        float(rdMolDescriptors.CalcNumAromaticRings(mol_implicit)),  # 10. 芳香环数
        float(rdMolDescriptors.CalcNumRings(mol_implicit)), # 11. 总环数
        float(n_heavy),                    # 12. 重原子数
    ], dtype=torch.float32)

    # ---- 节点特征: 基础10维 (spin兼容) ----
    # ---- 物理描述符 x_phys (额外 ~10维, GNN专用) ----
    node_feats = []
    node_phys = []  # 新增物理特征
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        an = atom.GetAtomicNum()
        # 基础10维 (不动, spin模型依赖)
        f = [
            float(an),
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

        # 物理描述符 (廉价, RDKit直接获取)
        # 电负性、共价半径、vdW半径、质量、价电子、自由基电子、电荷
        en = ELECTRONEG.get(an, 2.5)
        cr = COV_RADII.get(an, 100.0)
        phys = [
            float(en),                                      # 电负性
            float(cr) / 100.0,                               # 共价半径 归一化
            float(atom.GetMass()) / 100.0,                   # 原子质量 归一化
            float(atom.GetTotalValence()),                   # 总价电子
            float(atom.GetNumRadicalElectrons()),            # 自由基电子数
            float(charges[idx]),                              # Gasteiger电荷
            float(atom.GetIsAromatic() and an in [6,7,8]),    # 芳香杂原子
            # H键供体/受体 (化学直觉)
            float(an in [7,8] and atom.GetTotalNumHs() > 0),  # H键供体 (-OH/-NH)
            float(an in [7,8] and atom.GetTotalNumHs() == 0), # H键受体 (=O, -N=)
            # 是否为卤素
            float(an in [9, 17, 35, 53]),
        ]
        node_phys.append(phys)

    x = torch.tensor(node_feats, dtype=torch.float32)
    x_phys = torch.tensor(node_phys, dtype=torch.float32)

    # ---- 边特征 (4维 spin兼容) + edge_phys (2维 GNN专用) ----
    edge_idx = [[], []]
    edge_attr = []
    edge_phys = []  # 新增: 共轭、环内键
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
        # 边物理特征
        ephys = [
            float(b.GetIsConjugated()),
            float(b.IsInRing()),
        ]
        edge_phys.extend([ephys, ephys])

    edge_index = torch.tensor(edge_idx, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
    edge_phys_t = torch.tensor(edge_phys, dtype=torch.float32)
    y = torch.tensor([bde_value], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t,
                phys=phys_feats, y=y,
                x_phys=x_phys, edge_phys=edge_phys_t)


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

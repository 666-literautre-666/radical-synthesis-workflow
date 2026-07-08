"""
Δ-learning 规则 BDE 估算器 — 覆盖 24 种键型的教科书 BDE 值。

精度要求不高（±10 kcal 够用），关键是覆盖主要趋势。
GNN 学的是 "真实BDE - 规则BDE" 这个残差。
"""
from rdkit import Chem


# ---- 键型 → 近似 BDE (kcal/mol) ----
# 按 (atom1_symbol, atom2_symbol, bond_order) + 环境 分类

def estimate_bond_bde(mol, bond_idx):
    """
    给定 RDKit 分子和目标键索引，返回规则估算的 BDE 值。

    Args:
        mol: RDKit Mol (隐式氢)
        bond_idx: int

    Returns:
        float, BDE in kcal/mol (近似值)
    """
    if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
        return 90.0  # 无效键索引, 返回兜底值
    bond = mol.GetBondWithIdx(bond_idx)
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    s1, s2 = a1.GetSymbol(), a2.GetSymbol()
    order = bond.GetBondTypeAsDouble()

    # 排序: 按原子序数排，保证 C-H 和 H-C 一致
    if a1.GetAtomicNum() > a2.GetAtomicNum():
        s1, s2 = s2, s1
        a1, a2 = a2, a1

    is_aromatic = bond.GetIsAromatic()
    in_ring = bond.IsInRing()

    # 环境判断: 显式氢分子需要用隐式氢副本做环境检测 (GetDegree/GetNeighbors)
    has_explicit_H = any(at.GetAtomicNum() == 1 for at in mol.GetAtoms())
    if has_explicit_H:
        mol_env = Chem.RemoveHs(mol)
        # 隐式氢下重原子 idx 不变, H 原子不存在
        a1_env = mol_env.GetAtomWithIdx(a1.GetIdx()) if a1.GetAtomicNum() != 1 else None
        a2_env = mol_env.GetAtomWithIdx(a2.GetIdx()) if a2.GetAtomicNum() != 1 else None
    else:
        mol_env = mol
        a1_env, a2_env = a1, a2

    a1_chk = a1_env if a1_env is not None else a1
    a2_chk = a2_env if a2_env is not None else a2
    is_benzylic = _is_benzylic(a1_chk, a2_chk, mol_env)
    is_allylic = _is_allylic(a1_chk, a2_chk, mol_env)
    alpha_carbonyl = _alpha_to_carbonyl(a1_chk, a2_chk, mol_env)
    alpha_ether = _alpha_to_ether(a1_chk, a2_chk, mol_env)
    is_vinylic = _is_vinylic(a1_chk, a2_chk, mol_env)

    # ==================== C-H 键 ====================
    if {s1, s2} == {'C', 'H'}:
        c_atom = a1 if s1 == 'C' else a2
        # 用隐式氢原子做度判断 (显式氢下 GetDegree 会包含 H)
        c_env = a1_env if s1 == 'C' else a2_env
        if c_env is None:
            c_env = c_atom

        if is_aromatic:
            return 112.0  # Ph-H
        if is_vinylic:
            return 110.0  # C=C-H
        if c_env.GetHybridization() == Chem.HybridizationType.SP:
            return 130.0  # alkyne C-H
        if is_benzylic:
            deg = c_env.GetDegree()
            return 85.0 if deg <= 3 else 88.0
        if is_allylic:
            deg = c_env.GetDegree()
            return 83.0 if deg <= 3 else 86.0
        if alpha_carbonyl:
            return 93.0  # α-羰基 C-H
        if alpha_ether:
            return 92.0  # α-醚 C-H

        # 饱和 C-H: 按取代度
        deg = c_env.GetDegree()
        if deg == 1:    return 101.0  # primary
        elif deg == 2:  return 98.0   # secondary
        elif deg == 3:  return 96.0   # tertiary
        else:           return 95.0

    # ==================== C-C 键 ====================
    if {s1, s2} == {'C', 'C'}:
        if is_aromatic:
            return 115.0  # aromatic C-C (比单键强, 比双键弱)
        if order >= 2.0:
            return 145.0  # C=C
        if order >= 3.0:
            return 200.0  # C#C

        # 单键
        if is_benzylic:
            return 72.0
        if is_allylic:
            return 70.0
        if alpha_carbonyl:
            return 80.0
        if in_ring:
            return 82.0  # 环内C-C, 有环张力平均
        # 饱和 C-C
        deg1 = a1_env.GetDegree() if a1_env is not None else a1.GetDegree()
        deg2 = a2_env.GetDegree() if a2_env is not None else a2.GetDegree()
        if deg1 >= 3 or deg2 >= 3:
            return 83.0  # 有位阻, 略弱
        return 85.0

    # ==================== C-O 键 ====================
    if {s1, s2} == {'C', 'O'}:
        o_atom = a1 if s1 == 'O' else a2
        o_env = a1_env if s1 == 'O' else a2_env
        if o_env is None:
            o_env = o_atom
        if order >= 2.0:
            return 175.0  # C=O
        # C-O 单键
        if o_env.GetDegree() == 2:
            return 84.0  # 醚
        if o_env.GetDegree() >= 3:
            return 95.0  # 酯/酸 (与羰基共轭)
        return 96.0  # 醇

    # ==================== C-N 键 ====================
    if {s1, s2} == {'C', 'N'}:
        if order >= 2.0:
            return 140.0  # C=N
        n_atom = a1 if s1 == 'N' else a2
        n_env = a1_env if s1 == 'N' else a2_env
        if n_env is None:
            n_env = n_atom
        # 检查是否酰胺 (N 旁边有 C=O) — 用隐式氢做邻居搜索
        for nb in n_env.GetNeighbors():
            for nb_bond in nb.GetBonds():
                if nb_bond.GetBondTypeAsDouble() >= 2.0 and 8 in [nb.GetAtomicNum() for nb in [nb_bond.GetBeginAtom(), nb_bond.GetEndAtom()]]:
                    return 95.0  # 酰胺 C-N
        if n_env.GetDegree() >= 3:
            return 78.0  # 叔胺
        return 80.0  # 伯/仲胺

    # ==================== C-X 键 ====================
    if {s1, s2} == {'C', 'F'}:
        return 126.0
    if {s1, s2} == {'C', 'Cl'}:
        return 82.0
    if {s1, s2} == {'Br', 'C'}:
        if is_benzylic:
            return 55.0  # 苄基溴特别弱
        return 68.0
    if {s1, s2} == {'C', 'I'}:
        return 55.0
    if {s1, s2} == {'C', 'S'}:
        return 72.0

    # ==================== H-X 键 ====================
    if {s1, s2} == {'H', 'O'}:
        return 110.0
    if {s1, s2} == {'H', 'N'}:
        return 105.0
    if {s1, s2} == {'H', 'S'}:
        return 89.0

    # ==================== 杂原子-杂原子键 ====================
    if {s1, s2} == {'N', 'N'}:
        return 55.0
    if {s1, s2} == {'N', 'O'}:
        return 50.0
    if {s1, s2} == {'O', 'O'}:
        return 38.0
    if {s1, s2} == {'O', 'S'}:
        return 60.0
    if {s1, s2} == {'N', 'S'}:
        return 58.0
    if {s1, s2} == {'O', 'P'}:
        return 85.0
    if {s1, s2} == {'C', 'P'}:
        return 65.0
    if {s1, s2} == {'H', 'P'}:
        return 80.0
    if {s1, s2} == {'Cl', 'N'}:
        return 55.0
    if {s1, s2} == {'F', 'N'}:
        return 65.0
    if {s1, s2} == {'F', 'S'}:
        return 90.0

    # 兜底
    return 90.0


# ---- 环境判断辅助函数 ----
def _is_benzylic(a1, a2, mol):
    """至少一个原子直接连在芳香环上"""
    for atom in [a1, a2]:
        for nb in atom.GetNeighbors():
            if nb.GetIsAromatic():
                return True
    return False


def _is_allylic(a1, a2, mol):
    """至少一个原子邻接 C=C 双键（非芳香）"""
    for atom in [a1, a2]:
        for nb in atom.GetNeighbors():
            if not nb.GetIsAromatic():
                for bond in nb.GetBonds():
                    b = bond
                    if b.GetBondTypeAsDouble() == 2.0 and not b.GetIsAromatic():
                        oa = b.GetBeginAtom(); ob = b.GetEndAtom()
                        if oa.GetIdx() == nb.GetIdx() or ob.GetIdx() == nb.GetIdx():
                            return True
    return False


def _alpha_to_carbonyl(a1, a2, mol):
    """至少一个原子邻接 C=O"""
    for atom in [a1, a2]:
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 6:
                for nb_bond in nb.GetBonds():
                    if nb_bond.GetBondTypeAsDouble() >= 2.0:
                        oa = nb_bond.GetBeginAtom(); ob = nb_bond.GetEndAtom()
                        if 8 in [oa.GetAtomicNum(), ob.GetAtomicNum()]:
                            return True
    return False


def _alpha_to_ether(a1, a2, mol):
    """至少一个原子邻接 O-C 单键"""
    for atom in [a1, a2]:
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 8 and nb.GetDegree() == 2:
                return True
    return False


def _is_vinylic(a1, a2, mol):
    """至少一个原子直接连在 C=C 双键碳上"""
    for atom in [a1, a2]:
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                oa = bond.GetBeginAtom(); ob = bond.GetEndAtom()
                if oa.GetAtomicNum() == 6 and ob.GetAtomicNum() == 6:
                    if oa.GetIdx() == atom.GetIdx() or ob.GetIdx() == atom.GetIdx():
                        return True
    return False


# ---- 批量标注函数 (用于Δ-learning数据预处理) ----
def add_rule_bde_to_csv(csv_path, nrows=None):
    """读取 CSV, 为每行计算 rule_bde, 保存新列"""
    import pandas as pd
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)

    df = pd.read_csv(csv_path, nrows=nrows)
    rule_bdes = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100000 == 0:
            print(f"  Computing rule BDE: {i}/{len(df)}...")
        try:
            mol = Chem.MolFromSmiles(row['molecule'])
            if mol is None:
                rule_bdes.append(float('nan'))
                skipped += 1
                continue
            rule_bde = estimate_bond_bde(mol, int(row['bond_index']))
            rule_bdes.append(rule_bde)
        except Exception:
            rule_bdes.append(float('nan'))
            skipped += 1

    df['rule_bde'] = rule_bdes
    print(f"Done. Computed {len(df)-skipped} rule BDEs (skipped {skipped})")
    return df


if __name__ == '__main__':
    from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

    # 冒烟测试: 几个代表性键型
    test_cases = [
        ("Cc1ccccc1", 0, "苄位 C-H"),           # C-C (ring-methyl), 实际上是C-C单键在苄位
        ("CCO", 1, "乙醇 C-O"),
        ("c1ccccc1", 0, "苯 C=C (aromatic)"),
        ("C=CC", 0, "烯丙位 C=C"),
        ("CC(C)(C)O", 3, "叔丁醇 C-O"),
        ("BrCc1ccccc1", 1, "苄基溴 C-Br"),
        ("CC(C)=O", 0, "丙酮 C-C (alpha carbonyl)"),
    ]
    for smi, bid, desc in test_cases:
        mol = Chem.MolFromSmiles(smi)
        bde = estimate_bond_bde(mol, bid)
        # 获取键的实际原子对
        bond = mol.GetBondWithIdx(bid)
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        print(f"  {desc:25s} {a1}-{a2} bond_idx={bid} → rule BDE = {bde:.0f} kcal/mol")

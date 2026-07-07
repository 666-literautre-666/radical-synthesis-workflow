"""GNN BDE 推理模块 — 加载训练好的模型，对任意 SMILES 预测 BDE"""
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


# ======== 与 checkpoint 匹配的模型定义 ========
class BDEGNNv2(nn.Module):
    """纯 GINEConv 模型（无 phys_net），匹配 gnn_bde_best_v2.pt"""

    def __init__(self, node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0):
        super().__init__()
        self.hidden = hidden

        self.input_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden))

        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new).relu()
            h_new = self.dropout(h_new)
            h = h + h_new

        is_target = (x[:, -1] == 3.0)
        target_embs = h[is_target]
        target_embs = target_embs.view(-1, self.hidden * 2)
        return self.fc(target_embs)


# 全局缓存
_model = None
_bde_mean = None
_bde_std = None
_device = None
_loaded = False


def _ensure_model():
    """懒加载模型，首次调用时加载，后续复用"""
    global _model, _bde_mean, _bde_std, _device, _loaded
    if _loaded:
        return

    import os
    model_path = os.path.join(os.path.dirname(__file__), 'gnn_bde_best_v2.pt')
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(model_path, map_location=_device, weights_only=True)
    _bde_mean = ckpt['bde_mean']
    _bde_std = ckpt['bde_std']

    _model = BDEGNNv2(node_dim=10, edge_dim=4, hidden=256, n_layers=4).to(_device)
    _model.load_state_dict(ckpt['model'])
    _model.eval()
    _loaded = True


def _smiles_to_data_for_bond(mol, mol_implicit, bond_atom_idx1, bond_atom_idx2):
    """为指定键构建 PyG Data 对象。不依赖碎片信息。"""
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol_implicit.GetNumAtoms()

    # ---- atom_labels: 目标键原子=3.0, 其余=0.0 ----
    atom_labels = [0.0] * n_atoms
    atom_labels[bond_atom_idx1] = 3.0
    atom_labels[bond_atom_idx2] = 3.0

    # ---- 全局物理特征 (12维) ----
    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(a.GetDoubleProp('_GasteigerCharge')) for a in mol.GetAtoms()]

    benzylic_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[c][CH2,CH]')))
    allylic_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]=[C][CH2,CH]')))
    carbonyl_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[CH2,CH]')))

    # 共轭长度代理
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

    # 未成对电子代理（推断时无碎片，基于卤素/杂原子推断）
    smi_str = Chem.MolToSmiles(mol_implicit)
    radical_proxy = 1 if any(t in smi_str for t in ['Br', 'I', 'Cl']) else 0

    phys_feats = torch.tensor([
        float(n_atoms),
        float(max(charges)),
        float(min(charges)),
        float(np.mean(charges)),
        float(benzylic_hit),
        float(allylic_hit),
        float(carbonyl_hit),
        float(conj_length),
        float(radical_proxy),
        float(rdMolDescriptors.CalcNumAromaticRings(mol_implicit)),
        float(rdMolDescriptors.CalcNumRings(mol_implicit)),
        float(n_heavy),
    ], dtype=torch.float32)

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

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t, phys=phys_feats)


def _find_all_bonds(mol_implicit):
    """找出所有重原子间的键，返回 [(atom1_idx, atom2_idx, bond_type_str), ...]"""
    mol = Chem.AddHs(mol_implicit)
    bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        # 跳过 H-H 和与 H 的键（但这些在加氢分子中自动包含）
        # 我们保留至少一端是重原子的键
        if a1.GetAtomicNum() == 1 and a2.GetAtomicNum() == 1:
            continue
        bt = bond.GetBondType()
        bt_str = {Chem.BondType.SINGLE: '-', Chem.BondType.DOUBLE: '=',
                  Chem.BondType.TRIPLE: '#', Chem.BondType.AROMATIC: 'ar'}.get(bt, '?')
        # 键类型标签：重原子在前，C-H / C-C / C-Br 等
        sym1, sym2 = a1.GetSymbol(), a2.GetSymbol()
        n1, n2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        # H 永远放后面
        if n1 == 1:
            sym1, sym2 = sym2, sym1
        elif n2 == 1:
            pass  # 已经正确
        elif n1 < n2:
            sym1, sym2 = sym2, sym1
        label = f"{sym1}-{sym2}"
        bonds.append((a1.GetIdx(), a2.GetIdx(), label))
    return mol, bonds


def predict_all_bonds(smiles):
    """
    预测 SMILES 分子中所有键的 BDE。

    Returns:
        list[dict]: 按 BDE 升序，每项 {atom1, atom2, bond_type, bde_kcal}
    """
    _ensure_model()

    mol_implicit = Chem.MolFromSmiles(smiles)
    if mol_implicit is None:
        return []

    mol, bonds = _find_all_bonds(mol_implicit)
    if not bonds:
        return []

    results = []
    with torch.no_grad():
        for a1_idx, a2_idx, label in bonds:
            data = _smiles_to_data_for_bond(mol, mol_implicit, a1_idx, a2_idx)
            data = data.to(_device)
            pred_norm = _model(data).item()
            bde_kcal = round(pred_norm * _bde_std + _bde_mean, 1)
            results.append({
                "atom1": a1_idx,
                "atom2": a2_idx,
                "bond_type": label,
                "bde_kcal": bde_kcal,
            })

    results.sort(key=lambda x: x["bde_kcal"])
    return results


def predict_bde(smiles):
    """
    兼容旧接口：预测所有键 BDE。
    """
    return predict_all_bonds(smiles)


def predict_weakest_bde(smiles):
    """
    预测 SMILES 分子中最弱的键。

    Returns:
        dict: {"bond_type": "C-Br", "bde_kcal": 55.2, "source": "GNN"}
    """
    all_bonds = predict_all_bonds(smiles)
    if not all_bonds:
        return None

    w = all_bonds[0]
    return {
        "bond_type": w["bond_type"],
        "bde_kcal": w["bde_kcal"],
        "atom1": w["atom1"],
        "atom2": w["atom2"],
        "source": "GNN (GINEConv v2, MAE 1.46)",
    }


# ======== BDE 数据库查表 ========
_bde_db = None        # {smiles: weakest_bde_kcal}
_bde_db_full = None   # {smiles: [(bond_index, bond_type, bde), ...]}


def _load_bde_db():
    """加载 BDE 数据库（首次调用时初始化）"""
    global _bde_db, _bde_db_full
    if _bde_db is not None:
        return
    import os
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                            'bde_rdf_with_multi_halo_model_2.csv.gz')
    df = pd.read_csv(csv_path)
    _bde_db = df.groupby('molecule')['bde'].min().to_dict()
    # 全键存储：每个分子 → 所有键的 BDE 列表
    _bde_db_full = {}
    for _, row in df.iterrows():
        smi = row['molecule']
        if smi not in _bde_db_full:
            _bde_db_full[smi] = []
        _bde_db_full[smi].append({
            'bond_idx': int(row['bond_index']),
            'bond_type': str(row.get('bond_type', '?')),
            'bde_kcal': round(float(row['bde']), 1),
        })
    print(f"[BDE DB] Loaded {len(_bde_db):,} molecules for lookup")


def lookup_bde_db(smiles):
    """查 BDE 数据库，返回最弱键 BDE (kcal/mol)。"""
    _load_bde_db()
    return _bde_db.get(smiles)


def analyze_bde(smiles):
    """
    BDE Agent 核心 — 分析分子中所有键的 BDE。

    Returns:
        dict: {
            "smiles": str,
            "source": "database" | "GNN" | "mixed",
            "weakest": {"bond_type": "C-Br", "bde_kcal": 55.2},
            "all_bonds": [{"bond_type": ..., "bde_kcal": ..., "note": ...}, ...],
        }
    """
    _load_bde_db()
    _ensure_model()

    # ---- 第一步：数据库查表 ----
    if smiles in _bde_db_full:
        bonds = _bde_db_full[smiles]
        bonds_sorted = sorted(bonds, key=lambda x: x['bde_kcal'])
        return {
            "smiles": smiles,
            "source": "database",
            "confidence": "high",
            "n_bonds": len(bonds_sorted),
            "weakest": bonds_sorted[0],
            "all_bonds": bonds_sorted,
        }

    # ---- 第二步：GNN 预测 ----
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}

    gnn_bonds = predict_all_bonds(smiles)
    if not gnn_bonds:
        return {"error": "No bonds found"}

    # ---- 第三步：规则注释 ----
    for bond in gnn_bonds:
        bt = bond['bond_type']
        if bt == 'C-H':
            # 检查这个碳是否苄位（通过子结构匹配碳索引）
            a1, a2 = bond['atom1'], bond['atom2']
            # 取碳的索引（可能 atom1 或 atom2 是碳）
            note = ""
            benzylic_pat = Chem.MolFromSmarts('[c][CH2,CH3]')
            if mol.HasSubstructMatch(benzylic_pat):
                matches = mol.GetSubstructMatches(benzylic_pat)
                for m in matches:
                    if a1 in m or a2 in m:
                        note = "benzylic C-H"
                        break
            if note:
                bond['note'] = note

    return {
        "smiles": smiles,
        "source": "GNN",
        "confidence": "medium",
        "model": "GINEConv v2, MAE 1.46",
        "n_bonds": len(gnn_bonds),
        "weakest": gnn_bonds[0],
        "all_bonds": gnn_bonds,
    }


def draw_bde_map(smiles, output_path=None):
    """
    画出分子结构图，每根键标注 BDE 并颜色编码。
    红色=弱键(易断) 蓝色=强键(稳定)

    Args:
        smiles: 分子 SMILES
        output_path: 输出路径（默认 data/figures/{smiles}_bde_map.svg）
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    import os

    result = analyze_bde(smiles)
    if 'error' in result:
        print(f"Error: {result['error']}")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)

    all_bonds = result['all_bonds']
    min_bde = min(b['bde_kcal'] for b in all_bonds)
    max_bde = max(b['bde_kcal'] for b in all_bonds)

    # 构建键 BDE 映射（重原子间）
    n_heavy = mol.GetNumAtoms()
    bond_bde = {}
    bond_note = {}
    for b in all_bonds:
        a1, a2 = b['atom1'], b['atom2']
        # 映射到隐式氢原子索引（跳过 H）
        if a1 >= n_heavy or a2 >= n_heavy:
            # C-H 键：重原子索引保留，H 索引丢弃
            h_idx = min(a1, a2)
            bond_bde[h_idx] = b['bde_kcal']
            if b.get('note'):
                bond_note[h_idx] = b['note']
        else:
            bond_bde[(a1, a2)] = b['bde_kcal']

    # 颜色映射
    def bde_to_color(bde):
        ratio = (bde - min_bde) / (max_bde - min_bde + 1)
        r = 0.9 * (1 - ratio)
        g = 0.3
        b_col = 0.9 * ratio
        return (r, g, b_col)

    # 画图 — 给每根键单独上色
    d = rdMolDraw2D.MolDraw2DSVG(700, 500)
    opts = d.drawOptions()
    opts.bondLineWidth = 4
    opts.addAtomIndices = False
    opts.padding = 0.1

    # 收集需要高亮的键及其颜色
    highlight_bonds = []
    bond_colors_dict = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (i, j) if (i, j) in bond_bde else (j, i)
        bde = None
        if key in bond_bde:
            bde = bond_bde[key]
        elif i in bond_bde:
            bde = bond_bde[i]
        elif j in bond_bde:
            bde = bond_bde[j]
        if bde is not None:
            bidx = bond.GetIdx()
            highlight_bonds.append(bidx)
            bond_colors_dict[bidx] = bde_to_color(bde)

    legend = f'{smiles}  |  {result["weakest"]["bond_type"]}={result["weakest"]["bde_kcal"]} kcal  |  {result["source"]}'

    d.DrawMolecule(mol, legend=legend,
                   highlightBonds=highlight_bonds,
                   highlightBondColors=bond_colors_dict)
    d.FinishDrawing()
    svg = d.GetDrawingText()

    # 保存
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{smiles.replace("/","_")[:30]}_bde_map.svg')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f"Saved: {output_path}")

    # 文本图例
    print(f"\n{'='*50}")
    print(f"BDE Map: {smiles}")
    print(f"Source: {result['source']} ({result['confidence']})")
    print(f"{'='*50}")
    print(f"{'Bond':12s} {'BDE':>7s}  {'Note'}")
    print(f"{'-'*35}")
    for b in all_bonds:
        note = b.get('note', '')
        bar = '█' * max(1, int((max_bde - b['bde_kcal']) / (max_bde - min_bde + 1) * 20))
        print(f"{b['bond_type']:12s} {b['bde_kcal']:6.1f}  {bar} {note}")

    return output_path


if __name__ == '__main__':
    test_smiles = ["BrCc1ccccc1", "CC(C)=O", "c1ccccc1C"]
    for smi in test_smiles:
        print(f"\n=== {smi} ===")
        result = predict_weakest_bde(smi)
        if result:
            print(f"  Weakest: {result['bond_type']} = {result['bde_kcal']} kcal/mol")
        print(f"  All bonds:")
        for b in predict_all_bonds(smi)[:10]:
            print(f"    {b['bond_type']:10s}  {b['bde_kcal']:6.1f} kcal/mol")

    # 画图
    for smi in ['BrCc1ccccc1', 'c1ccccc1C']:
        draw_bde_map(smi)

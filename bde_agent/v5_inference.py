"""
BDEGNNv5 推理模块 — 基于 v5 双通道+多任务模型进行 BDE 预测
"""
import torch
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch_geometric.data import Data

from spin_pretrain import SpinPretrainNN
from train_v5 import BDEGNNv5


_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_model = None
_bde_mean = None
_bde_std = None
_loaded = False


def load_v5():
    global _model, _bde_mean, _bde_std, _loaded
    if _loaded:
        return

    base = os.path.dirname(__file__)

    spin = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0).to(_device)
    spin_pt = os.path.join(base, '..', 'spin_pretrain_frozen.pt')
    if not os.path.exists(spin_pt):
        spin_pt = os.path.join(os.path.expanduser('~'), 'spin_pretrain_frozen.pt')
    spin.load_state_dict(torch.load(spin_pt, map_location=_device, weights_only=True)['backbone'], strict=False)
    spin.eval()

    ckpt = torch.load(os.path.join(base, 'gnn_bde_v5_best.pt'),
                      map_location=_device, weights_only=True)
    _bde_mean = ckpt['bde_mean']
    _bde_std = ckpt['bde_std']

    _model = BDEGNNv5(spin, node_dim=10, hidden=256, n_layers=4, dropout=0.0).to(_device)
    _model.load_state_dict(ckpt['model'])
    _model.eval()
    _loaded = True
    print(f"v5 loaded: MAE 1.03, R2 0.9862, gate={torch.sigmoid(torch.tensor(ckpt.get('residual_gate',0))).item():.3f}")


def _mol_to_data_for_bond(smiles, bond_idx):
    """为指定键构建 PyG Data（与 gnn_data_utils.mol_to_data 格式一致）"""
    mol_imp = Chem.MolFromSmiles(smiles)
    if mol_imp is None:
        return None
    mol = Chem.AddHs(mol_imp)
    n_atoms = mol.GetNumAtoms()
    n_heavy = mol_imp.GetNumAtoms()

    if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
        return None
    try:
        bond = mol.GetBondWithIdx(bond_idx)
        target_atoms = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    except (IndexError, AttributeError, RuntimeError):
        return None

    atom_labels = [0.0] * n_atoms
    for i in target_atoms:
        atom_labels[i] = 3.0

    AllChem.ComputeGasteigerCharges(mol)
    charges = [float(a.GetDoubleProp('_GasteigerCharge')) for a in mol.GetAtoms()]

    benzylic_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[c][CH2,CH]')))
    allylic_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]=[C][CH2,CH]')))
    carbonyl_hit = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[CH2,CH]')))

    smi_str = Chem.MolToSmiles(mol_imp)
    radical_proxy = 1 if any(t in smi_str for t in ['Br', 'I', 'Cl']) else 0

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
        float(n_atoms), float(max(charges)), float(min(charges)),
        float(np.mean(charges)), float(benzylic_hit), float(allylic_hit),
        float(carbonyl_hit), float(conj_length), float(radical_proxy),
        float(rdMolDescriptors.CalcNumAromaticRings(mol_imp)),
        float(rdMolDescriptors.CalcNumRings(mol_imp)), float(n_heavy),
    ], dtype=torch.float32)

    node_feats = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        node_feats.append([
            float(atom.GetAtomicNum()), float(atom.GetDegree()),
            float(atom.GetTotalNumHs()), float(atom.GetIsAromatic()),
            float(atom.GetFormalCharge()),
            float(atom.GetHybridization() == Chem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            float(atom.IsInRing()), atom_labels[idx],
        ])

    x = torch.tensor(node_feats, dtype=torch.float32)
    edge_idx = [[], []]
    edge_attr = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edge_idx[0].extend([i, j])
        edge_idx[1].extend([j, i])
        bt = b.GetBondType()
        feats = [float(bt == Chem.BondType.SINGLE), float(bt == Chem.BondType.DOUBLE),
                 float(bt == Chem.BondType.TRIPLE), float(bt == Chem.BondType.AROMATIC)]
        edge_attr.extend([feats, feats])

    edge_index = torch.tensor(edge_idx, dtype=torch.long)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t, phys=phys_feats)


def predict_all_bonds(smiles):
    """预测 SMILES 分子中所有重原子间键的 BDE"""
    load_v5()

    mol_imp = Chem.MolFromSmiles(smiles)
    if mol_imp is None:
        return []

    mol = Chem.AddHs(mol_imp)
    bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 and a2.GetAtomicNum() == 1:
            continue
        bt = bond.GetBondType()
        bt_str = {Chem.BondType.SINGLE: '-', Chem.BondType.DOUBLE: '=',
                  Chem.BondType.TRIPLE: '#', Chem.BondType.AROMATIC: 'ar'}.get(bt, '?')
        sym1, sym2 = a1.GetSymbol(), a2.GetSymbol()
        n1, n2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        if n1 == 1:
            sym1, sym2 = sym2, sym1
        elif n2 == 1:
            pass
        elif n1 < n2:
            sym1, sym2 = sym2, sym1
        label = f"{sym1}-{sym2}"
        bonds.append((bond.GetIdx(), a1.GetIdx(), a2.GetIdx(), label))

    results = []
    with torch.no_grad():
        for bidx, a1_idx, a2_idx, label in bonds:
            data = _mol_to_data_for_bond(smiles, bidx)
            if data is None:
                continue
            data = data.to(_device)
            bde_pred, _, _, _, _ = _model(data)
            bde_kcal = round(bde_pred.item() * _bde_std + _bde_mean, 1)
            results.append({'atom1': a1_idx, 'atom2': a2_idx, 'bond_type': label, 'bde_kcal': bde_kcal})

    results.sort(key=lambda x: x['bde_kcal'])
    return results


def print_bde_table(smiles):
    """打印分子 BDE 表格"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    bonds = predict_all_bonds(smiles)
    if not bonds:
        print("No bonds found")
        return

    n_heavy = mol.GetNumAtoms()
    print(f"\n{'='*65}")
    print(f"v5 BDE: {smiles}  ({mol.GetNumAtoms()} heavy atoms, {len(bonds)} bonds)")
    print(f"{'='*65}")
    print(f"{'Bond':10s} {'Atoms':12s} {'BDE (kcal)':>10s}")
    print(f"{'-'*35}")

    for b in bonds:
        # Map explicit-H indices back to implicit-H notation
        a1, a2 = b['atom1'], b['atom2']
        if a1 >= n_heavy:
            a1_str = f"H(a1={a1})"
        else:
            a1_imp = mol.GetAtomWithIdx(a1)
            a1_str = f"{a1_imp.GetSymbol()}{a1}"
        if a2 >= n_heavy:
            a2_str = f"H(a2={a2})"
        else:
            a2_imp = mol.GetAtomWithIdx(a2)
            a2_str = f"{a2_imp.GetSymbol()}{a2}"

        print(f"{b['bond_type']:10s} {a1_str}-{a2_str:6s} {b['bde_kcal']:8.1f}")

    # Summary
    bdes = [b['bde_kcal'] for b in bonds]
    print(f"\nMin: {min(bdes):.1f}  Max: {max(bdes):.1f}  Mean: {np.mean(bdes):.1f}  Median: {np.median(bdes):.1f}")
    return bonds


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    test_mols = [
        'BrCc1ccccc1',           # benzyl bromide
        'CC(C)Cc1ccc(C(C)C(=O)O)cc1',  # ibuprofen
        'CC(C)=O',               # acetone
        'c1ccccc1C',             # toluene
        'C=CC',                  # propene
        'CCO',                   # ethanol
    ]
    for smi in test_mols:
        print_bde_table(smi)

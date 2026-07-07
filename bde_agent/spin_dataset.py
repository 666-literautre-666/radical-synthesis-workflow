"""
QM9star → PyG Dataset: CSV 流式读取, 按 mol_id 分组, 构建分子图
"""
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import os


class QM9starRadicalDataset(Dataset):
    """QM9star 自由基数据集 — 流式处理, 无需全量加载到内存"""

    def __init__(self, csv_path, max_molecules=None, root='data/qm9star_processed'):
        self.csv_path = csv_path
        self.max_molecules = max_molecules
        self.mol_groups = None  # {mol_id: [(atom_idx, elem, spin, charge, f_rad), ...]}
        super().__init__(root=root)

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_path)]

    @property
    def processed_file_names(self):
        n = f"_{self.max_molecules}" if self.max_molecules else ""
        return [f'data_list{n}.pt']

    def download(self):
        pass  # CSV 已存在

    def process(self):
        """CSV → PyG Data list, 分块读取，到了目标分子数就停"""
        chunk_size = 500000  # 每次读50万行（~25MB）
        mol_data = {}  # {mol_id: [(atom_idx, elem, spin, charge, f_rad, smiles), ...]}
        n_mols_wanted = self.max_molecules or 999999999
        total_rows = 0

        print(f"Scanning CSV (need {n_mols_wanted:,} molecules)...")
        for chunk in pd.read_csv(self.csv_path, chunksize=chunk_size):
            total_rows += len(chunk)
            for _, row in chunk.iterrows():
                mid = int(row['mol_id'])
                if mid not in mol_data:
                    if len(mol_data) >= n_mols_wanted:
                        break
                    mol_data[mid] = []
                mol_data[mid].append((
                    int(row['atom_idx']), row['element'], float(row['spin_density']),
                    float(row['mulliken_charge']), int(row['formal_radical']),
                    row['smiles']
                ))
            if len(mol_data) >= n_mols_wanted:
                break
            if len(mol_data) % 10000 == 0 or len(mol_data) < 10:
                print(f"  Scanned {total_rows:,} rows, collected {len(mol_data):,} molecules...")

        mol_ids = list(mol_data.keys())[:n_mols_wanted]

        print(f"Building graphs for {len(mol_ids)} molecules...")
        data_list = []
        skipped = 0

        for i, mid in enumerate(mol_ids):
            if i % 5000 == 0 and i > 0:
                print(f"  {i}/{len(mol_ids)} graphs built...")

            rows = mol_data[mid]
            smiles = rows[0][5]  # smiles 在第5个位置

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped += 1
                continue

            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()

            # 构建原子→数据行的映射 RDKit 原子索引 = CSV 行顺序
            # CSV 的 atom_idx 是按 RDKit AddHs 后的顺序
            spin_densities = np.zeros(n_atoms, dtype=np.float32)
            charges = np.zeros(n_atoms, dtype=np.float32)
            f_radicals = np.zeros(n_atoms, dtype=np.float32)

            for row in rows:
                idx = row[0]  # atom_idx
                if idx < n_atoms:
                    spin_densities[idx] = row[2]   # spin_density
                    charges[idx] = row[3]           # mulliken_charge
                    f_radicals[idx] = row[4]        # formal_radical

            # 节点特征 (10维)
            node_feats = []
            for atom in mol.GetAtoms():
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
                    f_radicals[atom.GetIdx()],  # 形式自由基标记
                ]
                node_feats.append(f)

            x = torch.tensor(node_feats, dtype=torch.float32)

            # 边特征 (4维)
            edge_idx = [[], []]
            edge_attr = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_idx[0].extend([i, j])
                edge_idx[1].extend([j, i])
                bt = bond.GetBondType()
                feats = [
                    float(bt == Chem.BondType.SINGLE),
                    float(bt == Chem.BondType.DOUBLE),
                    float(bt == Chem.BondType.TRIPLE),
                    float(bt == Chem.BondType.AROMATIC),
                ]
                edge_attr.extend([feats, feats])

            edge_index = torch.tensor(edge_idx, dtype=torch.long)
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

            # 标签
            y_spin = torch.tensor(spin_densities, dtype=torch.float32)
            y_charge = torch.tensor(charges, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_t,
                        y_spin=y_spin, y_charge=y_charge, smiles=smiles)
            data_list.append(data)

        print(f"Processed {len(data_list)} molecules (skipped {skipped})")

        # 保存
        torch.save(data_list, self.processed_paths[0])
        print(f"Saved to {self.processed_paths[0]}")

    def len(self):
        return len(torch.load(self.processed_paths[0], weights_only=False))

    def get(self, idx):
        data_list = torch.load(self.processed_paths[0], weights_only=False)
        return data_list[idx]


if __name__ == '__main__':
    # 测试: 加载100个分子
    ds = QM9starRadicalDataset(
        'E:/qm9star_radicals.csv',
        max_molecules=100,
        root='data/qm9star_processed'
    )
    print(f"Dataset size: {len(ds)}")
    d = ds[0]
    print(f"Sample: atoms={d.x.shape[0]}, edges={d.edge_index.shape[1]//2}, "
          f"spin_range=[{d.y_spin.min():.3f}, {d.y_spin.max():.3f}]")

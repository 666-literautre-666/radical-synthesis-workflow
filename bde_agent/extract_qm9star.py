"""
从 QM9star plain SQL (28GB) 提取自由基: SMILES + 自旋密度 + 电荷

策略: 流式读取, 先解析 molecule 表建索引, 再解析 snapshot 表输出匹配行
"""
import csv, os

SQL_PATH = "E:/qm9star_data_only.sql"
OUTPUT_CSV = "E:/qm9star_radicals.csv"


def extract():
    # ====== Pass 1: 建 molecule 索引 ======
    print("Pass 1: Indexing molecules...")
    mol_index = {}  # {molecule_id: (smiles, has_radical)}

    in_molecule_section = False
    line_count = 0

    with open(SQL_PATH, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')

            if line.startswith('COPY public.molecule'):
                in_molecule_section = True
                print(f"  Found molecule section")
                continue

            if in_molecule_section:
                if line == '\\.':
                    in_molecule_section = False
                    print(f"  Molecule section end. Indexed {len(mol_index):,} radicals")
                    break

                # TAB-separated: cols 0=smiles, 1=charge, 2=multiplicity, 9=id
                fields = line.split('\t')
                if len(fields) < 10:
                    continue

                try:
                    charge = int(fields[1])
                    multiplicity = int(fields[2])
                    mol_id = int(fields[9])
                    smiles = fields[0]

                    # 只保留自由基: charge=0, multiplicity=2 (doublet)
                    if charge == 0 and multiplicity == 2:
                        mol_index[mol_id] = smiles
                except (ValueError, IndexError):
                    continue

                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Scanned {line_count:,} molecule rows, indexed {len(mol_index):,} radicals")

    print(f"Pass 1 done: {len(mol_index):,} radical molecules indexed")

    # ====== Pass 2: 匹配 snapshot 数据 ======
    print("\nPass 2: Extracting spin densities...")
    in_snapshot_section = False
    output_count = 0

    with open(SQL_PATH, 'r', encoding='utf-8', errors='replace') as f_in, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:

        writer = csv.writer(f_out)
        writer.writerow(['smiles', 'mol_id', 'atom_idx', 'element',
                          'spin_density', 'mulliken_charge', 'formal_radical'])

        # Skip to snapshot section
        for line in f_in:
            line = line.rstrip('\n').rstrip('\r')

            if line.startswith('COPY public.snapshot'):
                in_snapshot_section = True
                print(f"  Found snapshot section")
                continue

            if in_snapshot_section:
                if line == '\\.':
                    break

                fields = line.split('\t')
                if len(fields) < 57:
                    continue

                # col 64 = molecule_id, col 3 = atoms[], col 6 = formal_num_radicals[]
                # col 16 = mulliken_charge[], col 17 = spin_densities[]
                try:
                    mol_id = int(fields[64])
                except (ValueError, IndexError):
                    continue

                if mol_id not in mol_index:
                    continue  # 不是自由基

                smiles = mol_index[mol_id]

                # 解析 PostgreSQL 数组: {val1,val2,val3}
                def parse_pg_array(s):
                    if s is None or s == '' or s == '\\N':
                        return []
                    s = s.strip('{}')
                    if not s:
                        return []
                    return [float(x) if '.' in x or 'e' in x.lower() else int(x)
                            for x in s.split(',')]

                atoms = parse_pg_array(fields[3])        # 原子序数
                spin = parse_pg_array(fields[17])         # 自旋密度
                charges = parse_pg_array(fields[16])      # Mulliken 电荷
                f_radicals = parse_pg_array(fields[6])    # 形式自由基计数

                # 元素符号映射
                elem_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
                            15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

                n_atoms = len(atoms)
                for i in range(n_atoms):
                    elem = elem_map.get(atoms[i] if isinstance(atoms[i], int) else int(atoms[i]),
                                        str(atoms[i]))
                    sd = spin[i] if i < len(spin) else 0.0
                    ch = charges[i] if i < len(charges) else 0.0
                    fr = f_radicals[i] if i < len(f_radicals) else 0

                    writer.writerow([smiles, mol_id, i, elem, sd, ch, fr])

                output_count += 1
                if output_count % 50000 == 0:
                    print(f"  Written {output_count:,} radical molecules")

    file_size = os.path.getsize(OUTPUT_CSV)
    print(f"\nDone! {output_count:,} radical molecules → {OUTPUT_CSV}")
    print(f"File size: {file_size / 1024 / 1024:.0f} MB")


if __name__ == '__main__':
    extract()

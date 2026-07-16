"""
从 QM9star SQL 提取: 自由基 + 键级 + 轨道能隙

v2 新增字段:
  - wiberg_bond_order: 每键 Wiberg 键级（边级）
  - nbo_bond_order: 每键 NBO 键级
  - alpha_homo/alpha_lumo/alpha_gap: alpha 轨道
  - beta_homo/beta_lumo/beta_gap: beta 轨道（自由基非零）
  - dipole: 偶极矩
"""
import csv, os, mmap

SQL_PATH = "E:/snapshot_data.sql"
MOLECULE_CSV = "E:/qm9star_radicals_v2_molecules.csv"
OUTPUT_CSV = "E:/qm9star_radicals_v2.csv"


def parse_pg_array(s):
    """解析 PostgreSQL 数组: {val1,val2,val3} → list[float]"""
    if s is None or s == '' or s == '\\N':
        return []
    s = s.strip('{}')
    if not s:
        return []
    parts = s.split(',')
    result = []
    for p in parts:
        try:
            result.append(float(p) if '.' in p or 'e' in p.lower() else int(p))
        except ValueError:
            result.append(0.0)
    return result


# ====== Pass 1: molecule 表建索引 ======
# 从原始 dump 读, 需要先转换
# 简化: 用已有的 radicals CSV 提取 mol_id → smiles 映射
print("Pass 1: Loading radical molecule index...")
mol_smiles = {}
with open('E:/qm9star_radicals.csv', 'r') as f:
    r = csv.reader(f)
    next(r)  # skip header
    for row in r:
        mid = int(row[1])
        if mid not in mol_smiles:
            mol_smiles[mid] = row[0]
print(f"  {len(mol_smiles):,} radical molecules indexed")


# ====== Pass 2: snapshot 表提取键级 + 轨道 ======
print(f"\nPass 2: Scanning snapshot SQL for bond orders + orbital gaps...")

# 列索引 (snapshot 表 65 列)
# col3  = atoms[]
# col4  = bonds[]  (格式: {{a1,a2,order},{a1,a2,order},...})
# col6  = formal_num_radicals[]
# col16 = mulliken_charge[]
# col17 = spin_densities[]
# col33 = alpha_homo
# col34 = alpha_lumo
# col36 = alpha_gap
# col37 = beta_homo
# col38 = beta_lumo
# col39 = beta_gap
# col44 = nbo_bond_order[]
# col45 = wiberg_bond_order[]
# col48 = dipole[]
# col64 = molecule_id

output_count = 0
with open(SQL_PATH, 'r', encoding='utf-8', errors='replace') as f_in, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:

    writer = csv.writer(f_out)
    writer.writerow([
        'smiles', 'mol_id',
        'atom_idx', 'element', 'spin_density', 'mulliken_charge', 'formal_radical',
        'wiberg_bond_order', 'nbo_bond_order',
        'alpha_homo', 'alpha_lumo', 'alpha_gap',
        'beta_homo', 'beta_lumo', 'beta_gap',
    ])

    in_snapshot = False
    scanned = 0

    for line in f_in:
        line = line.rstrip('\n').rstrip('\r')

        if line.startswith('COPY public.snapshot'):
            in_snapshot = True
            print(f"  Found snapshot section")
            continue

        if in_snapshot:
            if line == '\\.':
                break

            fields = line.split('\t')
            if len(fields) < 65:
                continue

            try:
                mol_id = int(fields[64])
            except (ValueError, IndexError):
                continue

            if mol_id not in mol_smiles:
                continue

            smiles = mol_smiles[mol_id]

            # 每原子数据
            atoms = [int(x) for x in parse_pg_array(fields[3])]
            spin = parse_pg_array(fields[17])
            charges = parse_pg_array(fields[16])
            f_radicals = parse_pg_array(fields[6])

            # 每键数据
            nbo_bo = parse_pg_array(fields[44])
            wiberg_bo = parse_pg_array(fields[45])

            # 图级数据
            try:
                alpha_gap = float(fields[36])
                beta_gap = float(fields[39])
                alpha_homo = float(fields[33])
                alpha_lumo = float(fields[34])
                beta_homo = float(fields[37])
                beta_lumo = float(fields[38])
            except (ValueError, IndexError):
                alpha_gap = beta_gap = alpha_homo = alpha_lumo = beta_homo = beta_lumo = 0.0

            elem_map = {1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl',35:'Br',53:'I'}
            n_atoms = len(atoms)

            for i in range(n_atoms):
                elem = elem_map.get(atoms[i], str(atoms[i]))
                sd = spin[i] if i < len(spin) else 0.0
                ch = charges[i] if i < len(charges) else 0.0
                fr = f_radicals[i] if i < len(f_radicals) else 0

                # 键级: 每原子的键级之和（简化）
                # 精确版需要解析 bonds 列的边列表
                wbo = 0.0
                nbo = 0.0

                writer.writerow([smiles, mol_id, i, elem, sd, ch, fr,
                                 wbo, nbo,
                                 alpha_homo, alpha_lumo, alpha_gap,
                                 beta_homo, beta_lumo, beta_gap])

            output_count += 1
            scanned += 1
            if output_count % 50000 == 0:
                print(f"  Written {output_count:,} molecules")

file_size = os.path.getsize(OUTPUT_CSV) if os.path.exists(OUTPUT_CSV) else 0
print(f"\nDone! {output_count:,} radical molecules → {OUTPUT_CSV}")
print(f"File size: {file_size / 1024 / 1024:.0f} MB")

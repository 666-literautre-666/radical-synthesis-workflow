"""
QM9star v2 全物理特征提取
列: smiles, mol_id, atom_idx, element,
     spin_density, mulliken_charge, npa_charge, formal_radical,
     wiberg_bo_sum, nbo_bo_sum,
     alpha_homo, alpha_lumo, alpha_gap, beta_homo, beta_lumo, beta_gap,
     dipole_mag
"""
import csv, os, math

SNAPSHOT_SQL = "E:/snapshot_v2.sql"
OUTPUT_CSV = "E:/qm9star_radicals_v2.csv"

# 列映射 (已手动验证):
# 3=atoms, 4=bonds, 6=formal_num_radicals
# 16=mulliken_charge, 17=spin_densities
# 33=alpha_homo, 34=alpha_lumo, 35=alpha_gap
# 36=beta_homo, 37=beta_lumo, 38=beta_gap
# 44=nbo_bond_order, 45=wiberg_bond_order
# 46=dipole, 51=npa_charges
# 64=molecule_id


def parse_pg_array(s):
    if s is None or s == '' or s == '\\N': return []
    s = s.strip('{}')
    if not s: return []
    # 处理嵌套数组 {{...},{...}} 格式: 按逗号分割后清理残余括号
    parts = [x.strip('{} ') for x in s.split(',')]
    result = []
    for p in parts:
        if not p: continue
        try:
            result.append(float(p) if ('.' in p or 'e' in p.lower()) else int(p))
        except ValueError:
            result.append(0.0)
    return result


# Pass 1: 建索引
print("Pass 1: Loading radical index from v1 CSV...")
mol_smiles = {}
with open('E:/qm9star_radicals.csv', 'r') as f:
    for i, row in enumerate(csv.reader(f)):
        if i == 0: continue
        mid = int(row[1])
        if mid not in mol_smiles: mol_smiles[mid] = row[0]
print(f"  {len(mol_smiles):,} radicals indexed")

# Pass 2: 提取
print(f"\nPass 2: Extracting from {SNAPSHOT_SQL}...")
elem_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

cnt = 0
with open(SNAPSHOT_SQL, 'r', encoding='utf-8', errors='replace') as fin, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fout:
    w = csv.writer(fout)
    # 10 个物理特征列 (全节点/图/边级)
    w.writerow(['smiles', 'mol_id', 'atom_idx', 'element',
                'spin_density', 'mulliken_charge', 'npa_charge', 'formal_radical',
                'wiberg_bo_sum', 'nbo_bo_sum',
                'alpha_homo', 'alpha_lumo', 'alpha_gap',
                'beta_homo', 'beta_lumo', 'beta_gap',
                'dipole_mag'])

    in_sec = False
    for line in fin:
        line = line.rstrip('\n').rstrip('\r')
        if line.startswith('COPY public.snapshot'):
            in_sec = True; continue
        if in_sec:
            if line == '\\.': break
            f = line.split('\t')
            if len(f) < 65: continue
            try:
                mid = int(f[64])
            except:
                continue
            if mid not in mol_smiles: continue
            smiles = mol_smiles[mid]

            atoms = [int(x) for x in parse_pg_array(f[3])]
            spin = parse_pg_array(f[17])
            mull_ch = parse_pg_array(f[16])
            npa_ch = parse_pg_array(f[51])
            f_rad = parse_pg_array(f[6])
            wiberg = parse_pg_array(f[45])
            nbo = parse_pg_array(f[44])
            n_atoms = len(atoms)

            # 图级标量
            def r(col):
                try: return float(f[col]) if f[col] != '\\N' else 0.0
                except: return 0.0

            ah = r(33); al = r(34); ag = r(35)
            bh = r(36); bl = r(37); bg = r(38)

            # 偶极矩大小
            dip = parse_pg_array(f[46])
            d_mag = math.sqrt(sum(x * x for x in dip[:3])) if len(dip) >= 3 else 0.0

            # 每原子键级和
            bonds_raw = parse_pg_array(f[4])  # [a1, a2, order, a1, a2, order, ...]
            atom_wb = [0.0] * n_atoms
            atom_nb = [0.0] * n_atoms
            nb_bonds = min(len(wiberg), len(nbo))
            for b in range(nb_bonds):
                bi = b * 3
                if bi + 2 < len(bonds_raw):
                    a1 = int(bonds_raw[bi]); a2 = int(bonds_raw[bi + 1])
                    if a1 < n_atoms: atom_wb[a1] += wiberg[b]; atom_nb[a1] += nbo[b]
                    if a2 < n_atoms: atom_wb[a2] += wiberg[b]; atom_nb[a2] += nbo[b]

            for i in range(n_atoms):
                w.writerow([smiles, mid, i,
                           elem_map.get(atoms[i], str(atoms[i])),
                           spin[i] if i < len(spin) else 0.0,
                           mull_ch[i] if i < len(mull_ch) else 0.0,
                           npa_ch[i] if i < len(npa_ch) else 0.0,
                           f_rad[i] if i < len(f_rad) else 0,
                           round(atom_wb[i], 6), round(atom_nb[i], 6),
                           round(ah, 6), round(al, 6), round(ag, 6),
                           round(bh, 6), round(bl, 6), round(bg, 6),
                           round(d_mag, 6)])
            cnt += 1
            if cnt % 100000 == 0: print(f"  {cnt:,} molecules")

print(f"\nDone! {cnt:,} molecules -> {OUTPUT_CSV}")
print(f"Size: {os.path.getsize(OUTPUT_CSV) / 1024 / 1024:.0f} MB")

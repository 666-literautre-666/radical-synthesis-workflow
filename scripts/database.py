"""
SQLite 数据库模块 — 存储实验数据、引发剂、溶剂、反应条件
零依赖，Python 内置 sqlite3
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "radical_db.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_db():
    """获取数据库连接（自动创建表和种子数据）"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_tables(conn)
    conn.commit()
    return conn


def _init_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        smiles TEXT NOT NULL,
        substrate_name TEXT,
        reaction_type TEXT,
        initiator TEXT,
        catalyst TEXT,
        solvent TEXT,
        temperature TEXT,
        yield_percent REAL,
        D_mhz REAL,
        E_mhz REAL,
        g_value REAL,
        notes TEXT,
        source TEXT,
        date_added TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS initiators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        structure_smiles TEXT,
        t_half_10h TEXT,
        working_range TEXT,
        solubility TEXT,
        byproduct TEXT,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS solvents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        boiling_point REAL,
        polarity TEXT,
        radical_suitable INTEGER DEFAULT 1,
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS reaction_types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        description TEXT,
        typical_conditions TEXT,
        substrates TEXT,
        key_literature TEXT
    );

    CREATE TABLE IF NOT EXISTS reaction_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        smarts TEXT UNIQUE,
        reaction_family TEXT,
        description TEXT
    );
    """)


def seed_initiators(conn):
    """导入已知引发剂数据"""
    initiators = [
        ("AIBN", "CC(C)(C#N)N=NC(C)(C)C#N", "65 °C", "50-80 °C",
         "organic solvents", "N2", "Azo initiator, most common"),
        ("BPO", "O=C(OOC(=O)c1ccccc1)c1ccccc1", "73 °C", "60-90 °C",
         "organic solvents", "CO2 + benzoic acid", "Peroxide initiator"),
        ("DTBP", "CC(C)(C)OOC(C)(C)C", "125 °C", "110-150 °C",
         "organic solvents", "acetone + ethane", "High-temperature initiator"),
        ("K2S2O8", "", "N/A", "60-80 °C",
         "water", "SO4²⁻", "Water-soluble inorganic initiator"),
        ("TBHP", "CC(C)(C)OO", "170 °C", "120-160 °C, or RT with Fe(II)",
         "organic/aqueous", "t-BuOH", "Fenton-type with reducing agent"),
        ("ACCN", "N#CC1(CCCCC1)N=NC1(CCCCC1)C#N", "88 °C", "80-100 °C",
         "organic solvents", "N2", "Higher-T alternative to AIBN"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO initiators (name,structure_smiles,t_half_10h,working_range,solubility,byproduct,notes) VALUES (?,?,?,?,?,?,?)",
        initiators)
    conn.commit()


def seed_reaction_types(conn):
    """导入自由基反应类型"""
    types = [
        ("atom_transfer", "ATRA / radical cyclization via halogen transfer",
         "Cu(I) cat., 60-80 °C, deoxygenated",
         "alkyl halide + alkene", "Curran, D.P. Synthesis 1988"),
        ("HAT", "Hydrogen atom transfer",
         "Peroxide initiator, light or heat",
         "C-H bond + radical acceptor", "Capaldo, L. Chem. Rev. 2022"),
        ("SET", "Single electron transfer (photoredox)",
         "Photoredox cat., visible light, RT",
         "electron-rich + electron-poor", "Prier, C.K. Chem. Rev. 2013"),
        ("radical_addition", "Radical addition to unsaturated bonds",
         "Initiator + alkene/alkyne, 60-100 °C",
         "R. + alkene/alkyne", "Zard, S.Z. 2003"),
        ("radical_cyclization", "Radical cyclization (5-exo-trig etc.)",
         "Bu3SnH/AIBN or photoredox",
         "unsaturated halide/selenide", "Giese, B. 1986"),
        ("HAS", "Homolytic aromatic substitution",
         "Peroxide, heat, or photoredox",
         "aryl diazonium/halide + nucleophile", "Studer, A. Angew. Chem. 2016"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO reaction_types (name,description,typical_conditions,substrates,key_literature) VALUES (?,?,?,?,?)",
        types)
    conn.commit()


def seed_solvents(conn):
    """导入常见自由基反应溶剂"""
    solvents = [
        ("MeCN", 82, "polar aprotic", 1, "Most common for ATRA/ATRP"),
        ("Toluene", 111, "nonpolar", 1, "Good for thermal initiators"),
        ("DCE", 84, "polar aprotic", 1, "1,2-dichloroethane"),
        ("THF", 66, "polar aprotic", 1, "Low BP, easy to remove"),
        ("DMSO", 189, "polar aprotic", 1, "High BP"),
        ("Benzene", 80, "nonpolar", 1, "Classic radical solvent"),
        ("DMF", 153, "polar aprotic", 1, "High BP, good solubility"),
        ("Water", 100, "polar protic", 1, "For KPS initiation"),
        ("DCM", 40, "polar aprotic", 1, "Low BP"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO solvents (name,boiling_point,polarity,radical_suitable,notes) VALUES (?,?,?,?,?)",
        solvents)
    conn.commit()


def seed_smarts_rules(conn):
    """导入自由基 SMARTS 识别规则"""
    rules = [
        ("C-Br_bond", "[CX4][Br]", "atom_transfer", "sp3 C-Br: excellent radical precursor"),
        ("C-I_bond", "[CX4][I]", "atom_transfer", "sp3 C-I: excellent radical precursor"),
        ("C-Cl_bond", "[CX4][Cl]", "atom_transfer", "sp3 C-Cl: moderate radical precursor"),
        ("benzylic_CH", "[c][CH2]", "HAT", "Benzylic C-H: BDE ~85-90 kcal/mol"),
        ("allylic_CH", "[C]=[C][CH2]", "HAT", "Allylic C-H: BDE ~86 kcal/mol"),
        ("aldehyde_CH", "[CX3H1](=O)", "HAT", "Aldehyde C-H: BDE ~87 kcal/mol"),
        ("alkene_C=C", "[C]=[C]", "radical_addition", "C=C: radical addition site"),
        ("alkyne_CC", "[C]#[C]", "radical_addition", "Alkyne: radical addition site"),
        ("aryl_Br", "[c][Br]", "HAS", "Aryl-Br: HAS precursor"),
        ("aryl_I", "[c][I]", "HAS", "Aryl-I: HAS precursor"),
        ("acrylate", "[C]=[C]C(=O)O", "radical_addition", "Acrylate: ATRP monomer"),
        ("styrene", "[C]=[C]c1ccccc1", "radical_addition", "Styrene: radical polymerization"),
        ("nitroxide", "N[O]", "spin_trap", "Nitroxide: spin trap or NMP mediator"),
        ("diazonium", "[c][N+]#N", "SET", "Diazonium: SET-active"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO reaction_rules (name,smarts,reaction_family,description) VALUES (?,?,?,?)",
        rules)
    conn.commit()


# ---------- 查询接口 ----------

def add_experiment(smiles: str, reaction_type: str = "", initiator: str = "",
                   catalyst: str = "", solvent: str = "", temperature: str = "",
                   yield_percent: float = None, D_mhz: float = None,
                   E_mhz: float = None, g_value: float = None,
                   notes: str = "", source: str = "experiment",
                   substrate_name: str = "") -> int:
    """添加一条实验记录"""
    conn = get_db()
    cur = conn.execute(
        """INSERT INTO experiments (smiles,substrate_name,reaction_type,initiator,catalyst,
           solvent,temperature,yield_percent,D_mhz,E_mhz,g_value,notes,source)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (smiles, substrate_name, reaction_type, initiator, catalyst,
         solvent, temperature, yield_percent, D_mhz, E_mhz, g_value, notes, source))
    conn.commit()
    return cur.lastrowid


def query_similar_substrates(smiles: str, limit: int = 5) -> list:
    """查询数据库中与输入底物最相似的历史记录

    使用 RDKit 子结构匹配 + Tanimoto 相似度排序
    """
    from rdkit import Chem
    from rdkit.Chem import DataStructs, AllChem

    conn = get_db()
    rows = conn.execute(
        "SELECT id, smiles, substrate_name, reaction_type, initiator, "
        "catalyst, solvent, temperature, yield_percent, D_mhz, E_mhz, "
        "g_value, notes, source FROM experiments ORDER BY date_added DESC LIMIT 500"
    ).fetchall()

    if not rows:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    fp_query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    scored = []
    for row in rows:
        mol_db = Chem.MolFromSmiles(row["smiles"])
        if not mol_db:
            continue
        fp_db = AllChem.GetMorganFingerprintAsBitVect(mol_db, 2, nBits=2048)
        tanimoto = DataStructs.TanimotoSimilarity(fp_query, fp_db)
        has_substruct = mol.HasSubstructMatch(mol_db) or mol_db.HasSubstructMatch(mol)
        score = tanimoto * 0.7 + (0.3 if has_substruct else 0)
        scored.append((score, dict(row)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def get_all_reaction_types() -> list:
    conn = get_db()
    return [dict(r) for r in conn.execute("SELECT * FROM reaction_types").fetchall()]


def get_initiators_for_condition(temp_range: str = "") -> list:
    conn = get_db()
    return [dict(r) for r in conn.execute("SELECT * FROM initiators").fetchall()]


def get_smarts_matches(smiles: str) -> list:
    """返回输入分子匹配的所有 SMARTS 规则"""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    conn = get_db()
    rules = conn.execute("SELECT * FROM reaction_rules").fetchall()
    matches = []
    for rule in rules:
        pattern = Chem.MolFromSmarts(rule["smarts"])
        if pattern and mol.HasSubstructMatch(pattern):
            matches.append(dict(rule))
    return matches


def init_database():
    """一键初始化：建表 + 种子数据"""
    conn = get_db()
    seed_initiators(conn)
    seed_solvents(conn)
    seed_reaction_types(conn)
    seed_smarts_rules(conn)
    print(f"[database] Initialized at {DB_PATH}")
    print(f"  {_count(conn, 'initiators')} initiators, "
          f"{_count(conn, 'solvents')} solvents, "
          f"{_count(conn, 'reaction_types')} reaction types, "
          f"{_count(conn, 'reaction_rules')} SMARTS rules, "
          f"{_count(conn, 'experiments')} experiments")


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


if __name__ == "__main__":
    init_database()

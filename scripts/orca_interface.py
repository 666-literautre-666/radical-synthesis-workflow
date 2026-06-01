"""
ORCA 6.x 量子化学计算接口 — 学术免费，无需服务器
====================================================
安装：去 https://orcaforum.kofo.mpg.de/ 注册（学术邮箱）→ 下载 Windows 版 → 解压到 C:\\orca\\
功能：NMR 化学位移 | EPR g张量+超精细耦合 | 结构优化 | 过渡态搜索
"""

import re
import subprocess
import shutil
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORCA_INPUT_DIR = PROJECT_ROOT / "data" / "orca_inputs"
ORCA_OUTPUT_DIR = PROJECT_ROOT / "data" / "orca_outputs"
ORCA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
ORCA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 常见 ORCA 路径
ORCA_PATHS = [
    r"C:\ORCA_6.1.1\orca.exe",
    "orca", r"C:\orca\orca.exe", r"C:\orca\orca.bat",
    r"C:\Program Files\orca\orca.exe", r"D:\orca\orca.exe",
]


def find_orca() -> str:
    """查找 ORCA 可执行文件"""
    for p in ORCA_PATHS:
        if shutil.which(p) or Path(p).exists():
            return p
    return ""


# ===========================================================================
# ORCA 输入文件生成
# ===========================================================================

# 可用泛函 + 基组组合
FUNCTIONALS = {
    "b3lyp": "B3LYP", "pbe0": "PBE0", "m062x": "M06-2X",
    "wb97x-d3": "wB97X-D3", "bp86": "BP86", "tpss": "TPSS",
    "r2scan": "r2SCAN", "b97-3c": "B97-3c",  # B97-3c 自带基组，极快
}

BASIS_SETS = {
    "quick": "def2-SVP",       # 快速预优化
    "standard": "def2-TZVP",  # 标准精度
    "accurate": "def2-TZVPP",  # 高精度
    "nmr": "pcSseg-2",         # NMR 专用基组（Jensen 优化）
    "epr": "EPR-II",           # EPR 专用基组
}

SOLVENTS = {
    "water": "Water", "acetonitrile": "Acetonitrile", "meCN": "Acetonitrile",
    "dmso": "DMSO", "chloroform": "Chloroform", "chcl3": "Chloroform",
    "dcm": "Dichloromethane", "thf": "THF", "toluene": "Toluene",
    "methanol": "Methanol", "benzene": "Benzene",
    "": "",  # 气相
}


def _get_coordinates(smiles: str) -> str:
    """SMILES → 3D 坐标（MMFF94 预优化）"""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"  {atom.GetSymbol():<3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
    return "\n".join(lines)


def build_nmr_input(smiles: str, name: str = "molecule",
                    functional: str = "B3LYP",
                    basis: str = "pcSseg-2",
                    solvent: str = "chloroform",
                    n_cores: int = 4) -> str:
    """
    生成 ORCA NMR 计算输入文件。

    NMR 推荐: B3LYP/pcSseg-2 或 PBE0/pcSseg-2
    """
    coords = _get_coordinates(smiles)
    func = FUNCTIONALS.get(functional.lower(), functional)
    bs = BASIS_SETS.get(basis.lower(), basis)
    solv = SOLVENTS.get(solvent.lower(), solvent)

    solv_line = f"CPCM({solv})" if solv else ""

    inp = f"""! {func} {bs} NMR {solv_line} TightSCF
%pal nprocs {n_cores} end
%maxcore 2000
* xyz 0 1
{coords}
*
"""
    return inp


def build_epr_input(smiles: str, name: str = "radical",
                    functional: str = "B3LYP",
                    basis: str = "EPR-II",
                    solvent: str = "",
                    multiplicity: int = 2,
                    n_cores: int = 4) -> str:
    """
    生成 ORCA EPR 计算输入文件（g-tensor + 超精细耦合）。

    multiplicity: 2 = doublet, 3 = triplet
    """
    coords = _get_coordinates(smiles)
    func = FUNCTIONALS.get(functional.lower(), functional)
    bs = BASIS_SETS.get(basis.lower(), basis)
    solv = SOLVENTS.get(solvent.lower(), solvent)
    solv_line = f"CPCM({solv})" if solv else ""

    inp = f"""! {func} {bs} {solv_line} TightSCF
%pal nprocs {n_cores} end
%maxcore 2000
* xyz 0 {multiplicity}
{coords}
*
%eprnmr
  Nuclei = all H {{ also C, N, O }} {{ also F, P, Br }}
  end
"""
    return inp


def build_opt_input(smiles: str, name: str = "molecule",
                    functional: str = "B3LYP",
                    basis: str = "def2-TZVP",
                    solvent: str = "",
                    n_cores: int = 4) -> str:
    """生成结构优化输入文件"""
    coords = _get_coordinates(smiles)
    func = FUNCTIONALS.get(functional.lower(), functional)
    bs = BASIS_SETS.get(basis.lower(), basis)
    solv = SOLVENTS.get(solvent.lower(), solvent)
    solv_line = f"CPCM({solv})" if solv else ""

    inp = f"""! {func} {bs} Opt Freq {solv_line} TightSCF
%pal nprocs {n_cores} end
%maxcore 2000
* xyz 0 1
{coords}
*
"""
    return inp


def save_input(inp_content: str, name: str) -> Path:
    """保存 .inp 文件"""
    path = ORCA_INPUT_DIR / f"{name}.inp"
    path.write_text(inp_content, encoding="utf-8")
    print(f"  [ORCA inp] {path}")
    return path


def run_orca(inp_path: Path, timeout_hours: float = 24) -> Path:
    """在本地运行 ORCA 计算"""
    orca_exe = find_orca()
    if not orca_exe:
        raise RuntimeError(
            "找不到 ORCA。请先:\n"
            "  1. 去 https://orcaforum.kofo.mpg.de/ 注册（学术邮箱）\n"
            "  2. 下载 Windows 版 → 解压到 C:\\orca\\\n"
            "  3. 把 C:\\orca\\ 加入系统 PATH 环境变量"
        )

    out_path = ORCA_OUTPUT_DIR / f"{inp_path.stem}.out"
    cmd = f'"{orca_exe}" "{inp_path}" > "{out_path}"'

    print(f"  [ORCA] 开始计算: {inp_path.name}")
    print(f"  [ORCA] 使用 {find_orca()}")
    print(f"  [ORCA] 预计时间: 分钟到小时级（取决于分子大小）")

    result = subprocess.run(cmd, shell=True, cwd=str(ORCA_OUTPUT_DIR),
                             timeout=timeout_hours * 3600)
    if result.returncode != 0:
        print(f"  [警告] ORCA 退出码: {result.returncode}")
    return out_path


# ===========================================================================
# ORCA 输出解析
# ===========================================================================

def parse_nmr_output(out_path: str) -> dict:
    """解析 ORCA NMR 输出，提取化学位移"""
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")

    result = {
        "h_shifts": {},
        "c_shifts": {},
        "n_shifts": {},
        "scf_energy": None,
        "method": None,
    }

    # SCF Energy
    m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", text)
    if not m:
        m = re.search(r"Total Energy\s+[.:]+\s+([-\d.]+)", text)
    if m:
        result["scf_energy"] = float(m.group(1))

    # Method
    m = re.search(r"! (.+)", text)
    if m:
        result["method"] = m.group(1).strip()

    # NMR Shielding Constants
    # ORCA outputs: "Nucleus  13C :  ...  isotropic = xxx.xx"
    # or in a table format
    shielding = {}

    # Pattern 1: detailed table
    for line in text.split("\n"):
        # "  1  H    28.345    5.123    4.987   ..."
        m = re.search(r"^\s*(\d+)\s+(\w+)\s+([-\d.]+)\s+", line)
        if m and "CHEMICAL SHIFTS" in text[:text.index(line)] if "CHEMICAL SHIFTS" in text else True:
            pass  # too ambiguous, need context

    # Pattern 2: ORCA 5+ uses clear headers
    in_shielding = False
    for line in text.split("\n"):
        if "CHEMICAL SHIELDING SUMMARY" in line or "NMR shielding constants" in line:
            in_shielding = True
            continue
        if in_shielding:
            if line.strip() == "" or "----" in line:
                continue
            if "Total" in line or "CHEMICAL SHIFTS" in line:
                break
            # "  0  H   31.1234   5.1234   5.1234..."
            m = re.search(r"^\s*(\d+)\s+(\w+)\s+([-\d.]+)", line)
            if m:
                idx = int(m.group(1))
                elem = m.group(2)
                if elem in ["H", "C", "N", "F", "P"]:
                    shielding[f"{elem}{idx}"] = float(m.group(3))

    # Pattern 3: per-nucleus output
    for match in re.finditer(
        r"Nucleus\s+(\d+)(\w+)\s*:.*?isotropic\s*=\s*([-\d.]+)",
        text, re.DOTALL
    ):
        idx = int(match.group(1))
        elem = match.group(2)
        iso = float(match.group(3))
        if elem in ["H", "C", "N", "F", "P"]:
            shielding[f"{elem}{idx}"] = iso

    # Convert shielding → chemical shift
    # TMS reference (B3LYP level typical values)
    TMS = {"H": 31.8, "C": 184.1, "N": 245.0, "F": 355.0, "P": 325.0}

    for label, sigma in shielding.items():
        elem = re.match(r"([A-Za-z]+)", label).group(1)
        ref = TMS.get(elem)
        if ref is None:
            continue
        shift = round(ref - sigma, 2)

        if elem == "H":
            result["h_shifts"][label] = shift
        elif elem == "C":
            result["c_shifts"][label] = round(shift, 1)
        elif elem == "N":
            result["n_shifts"][label] = round(shift, 1)

    nh = len(result["h_shifts"])
    nc = len(result["c_shifts"])
    if nh + nc == 0:
        print(f"  [警告] 未检测到 NMR 位移，可能 ORCA 输出格式不同。请检查 {out_path}")
    else:
        print(f"  [ORCA NMR] {Path(out_path).name}")
        print(f"    1H: {nh} signals  |  13C: {nc} signals")
        if result["scf_energy"]:
            print(f"    Energy: {result['scf_energy']:.6f} Eh")
        print(f"    1H shifts: {list(result['h_shifts'].values())}")
        print(f"    13C shifts: {list(result['c_shifts'].values())}")

    return result


def parse_epr_output(out_path: str) -> dict:
    """解析 ORCA EPR 输出，提取 g 张量和超精细耦合常数"""
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")

    result = {
        "g_tensor": None,
        "g_iso": None,
        "hfc": [],
    }

    # --- g-tensor ---
    # ORCA 输出: "g-tensor:" 后跟 3 行对角元
    g_block = re.search(
        r"g-tensor:.*?\n(.*?)\n(.*?)\n(.*?)\n",
        text, re.DOTALL
    )
    if not g_block:
        g_block = re.search(
            r"ELECTRONIC G-MATRIX.*?\n(.*?)\n(.*?)\n(.*?)\n",
            text, re.DOTALL
        )

    if g_block:
        all_text = g_block.group(0)
        g_vals = re.findall(r"([\d.]+)", all_text)
        g_floats = [float(v) for v in g_vals if 1.9 < float(v) < 3.0]
        if len(g_floats) >= 3:
            result["g_tensor"] = g_floats[:3]
            result["g_iso"] = round(sum(g_floats[:3]) / 3, 5)

    # Fallback: search for "g-iso" or "g_iso"
    if result["g_iso"] is None:
        m = re.search(r"g[-_]?iso\s*[=:]\s*([\d.]+)", text, re.IGNORECASE)
        if m:
            result["g_iso"] = float(m.group(1))

    # --- Hyperfine Couplings ---
    # ORCA outputs a table with columns:
    # Nucleus  Element  A_iso(MHz)  T_xx  T_yy  T_zz  ...
    hfc_section = False
    for line in text.split("\n"):
        if "HYPERFINE COUPLING" in line.upper() or "A- TENSOR" in line:
            hfc_section = True
            continue
        if hfc_section:
            if "----" in line or line.strip() == "":
                continue
            if "Total" in line or "Sum" in line or "---" in line:
                break

            # "  1  H  14.50  -1.23  -1.45  2.68"
            m = re.search(r"(\d+)\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                result["hfc"].append({
                    "atom_idx": int(m.group(1)),
                    "nucleus": m.group(2),
                    "A_iso_MHz": float(m.group(3)),
                    "A_iso_G": round(float(m.group(3)) / 2.8025, 2),
                })
                continue

            # Simpler: "  1  H  14.50  1.23  1.45  2.68"
            m = re.search(r"(\d+)\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m and float(m.group(3)) < 1000:  # reasonable A value range
                result["hfc"].append({
                    "atom_idx": int(m.group(1)),
                    "nucleus": m.group(2),
                    "A_iso_MHz": float(m.group(3)),
                    "A_iso_G": round(float(m.group(3)) / 2.8025, 2),
                })

    n_hfc = len(result["hfc"])
    print(f"  [ORCA EPR] {Path(out_path).name}")
    print(f"    g_iso = {result['g_iso']}")
    print(f"    {n_hfc} hyperfine coupling constants")
    for h in result["hfc"]:
        print(f"      {h['nucleus']}{h['atom_idx']}: A_iso = {h['A_iso_G']:.2f} G ({h['A_iso_MHz']:.2f} MHz)")

    return result


# ===========================================================================
# 便捷全流程
# ===========================================================================

def setup_guide():
    """打印 ORCA 安装指南"""
    guide = """
    === ORCA 安装指南（5 步，10 分钟）===

    1. 打开浏览器，访问: https://orcaforum.kofo.mpg.de/
    2. 点击右上角 Register → 用学校邮箱注册（@xxx.edu.cn）
    3. 登录后进入 Downloads → 下载 Windows 版本:
       orca_6.x.x_win64.zip (约 6 GB)
    4. 解压到 C:\\orca\\
    5. 添加环境变量:
       - 按 Win+R → 输入 sysdm.cpl → 高级 → 环境变量
       - 在 Path 中添加 C:\\orca\\
       - 确定保存

    验证: 打开新终端，输入 orca --version，显示版本号即成功。

    ORCA 对学术用户完全免费，引用:
    Neese, F. WIREs Comput. Mol. Sci. 2022, 12, e1606.
    """
    print(guide)


def check_installation() -> bool:
    """检查 ORCA 是否已正确安装"""
    orca = find_orca()
    if orca:
        print(f"  [OK] ORCA 已安装: {orca}")
        try:
            r = subprocess.run([orca, "--version"], capture_output=True, text=True, timeout=10)
            print(f"  [OK] 版本: {r.stdout.strip()}")
            return True
        except Exception:
            pass
    else:
        print("  [未安装] ORCA 未找到")
        setup_guide()
        return False


print("[orca_interface] Ready.")
print("  - build_nmr_input(smiles) → .inp NMR 计算")
print("  - build_epr_input(smiles)  → .inp EPR 计算")
print("  - build_opt_input(smiles)  → .inp 结构优化")
print("  - parse_nmr_output(.out)   → 解析化学位移")
print("  - parse_epr_output(.out)   → 解析 g-tensor + HFC")
print("  - check_installation()     → 检查 ORCA 是否装好")

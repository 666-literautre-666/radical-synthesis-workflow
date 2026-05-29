"""
Gaussian 09/16 DFT 计算接口
============================
功能：
  1. 生成 Gaussian 输入文件 (.gjf) — NMR / EPR / 结构优化
  2. 解析 Gaussian 输出文件 (.log) — 提取 NMR 位移、g张量、超精细耦合
  3. 支持本地 Gaussian 或远程集群提交

使用场景：
  - 本地有 g09/g16 → 自动运行
  - 只有 GaussView → 生成 .gjf 拖入 GaussView 提交
  - 远程集群 → 生成 .gjf 上传，下载 .log 拖回来解析
"""

import re
import subprocess
import tempfile
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAUSSIAN_INPUT_DIR = PROJECT_ROOT / "data" / "gaussian_inputs"
GAUSSIAN_OUTPUT_DIR = PROJECT_ROOT / "data" / "gaussian_outputs"
GAUSSIAN_INPUT_DIR.mkdir(parents=True, exist_ok=True)
GAUSSIAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 常见 Gaussian 可执行文件路径
GAUSSIAN_PATHS = [
    "g16", "g09", "g16w", "g09w",
    r"C:\G16W\g16.exe", r"C:\G09W\g09w.exe",
    "/usr/local/g16/g16", "/opt/gaussian/g16/g16",
]

# TMS 参考值（B3LYP/6-311+G(2d,p) 级别的各原子 NMR 屏蔽常数）
# 来自文献：J. Org. Chem. 2015, 80, 4583-4591
TMS_REFERENCE = {
    "H": 31.8,   # 1H reference shielding (ppm)
    "C": 184.1,  # 13C reference shielding (ppm)
    "N": 245.0,  # 15N reference shielding (ppm)
    "F": 355.0,  # 19F reference shielding (ppm)
}

# 常见溶剂对应的 Gaussian SCRF 关键词
SOLVENT_KEYWORDS = {
    "water": "scrf=(smd,solvent=water)",
    "acetonitrile": "scrf=(smd,solvent=acetonitrile)",
    "meCN": "scrf=(smd,solvent=acetonitrile)",
    "dmso": "scrf=(smd,solvent=dmso)",
    "chloroform": "scrf=(smd,solvent=chloroform)",
    "chcl3": "scrf=(smd,solvent=chloroform)",
    "dcm": "scrf=(smd,solvent=dichloromethane)",
    "thf": "scrf=(smd,solvent=tetrahydrofuran)",
    "toluene": "scrf=(smd,solvent=toluene)",
    "methanol": "scrf=(smd,solvent=methanol)",
    "benzene": "scrf=(smd,solvent=benzene)",
    "": "",  # 气相
}


def find_gaussian() -> str:
    """查找系统中的 Gaussian 可执行文件，找不到返回空字符串"""
    import shutil
    for path in GAUSSIAN_PATHS:
        if shutil.which(path) or Path(path).exists():
            return path
    return ""


# ===========================================================================
# Part 1: 输入文件生成器
# ===========================================================================

def build_nmr_input(smiles: str, name: str = "molecule",
                    functional: str = "B3LYP",
                    basis_set: str = "6-311+G(2d,p)",
                    solvent: str = "chloroform",
                    n_procs: int = 4, memory: str = "4GB",
                    charge: int = 0, multiplicity: int = 1) -> str:
    """
    生成 NMR 计算的 Gaussian 输入文件 (GIAO).

    Parameters
    ----------
    smiles : 分子 SMILES
    functional : 泛函 (B3LYP, M06-2X, wB97XD, PBE0...)
    basis_set : 基组 (6-311+G(2d,p) 推荐用于 NMR)
    solvent : 溶剂名 (chloroform, dmso, water, meCN...)
    n_procs : CPU 核心数
    memory : 内存

    Returns
    -------
    gjf_content : 完整的 .gjf 文件内容
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    # 3D 坐标生成（MMFF94 力场预优化）
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    # 获取原子坐标
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append(f" {atom.GetSymbol():<3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    charge_mult = f"{charge} {multiplicity}"
    solvent_kw = SOLVENT_KEYWORDS.get(solvent.lower(), "")
    scf_line = f"#p opt freq {functional}/{basis_set} {solvent_kw}".strip()

    header = f"""%chk={name}.chk
%nprocshared={n_procs}
%mem={memory}
#p nmr giao {functional}/{basis_set} {solvent_kw}
nosymm

{name} — NMR prediction

{charge_mult}
""" + "\n".join(coords) + "\n\n"

    return header


def build_epr_input(smiles: str, name: str = "radical",
                    functional: str = "UB3LYP",
                    basis_set: str = "EPR-II",
                    solvent: str = "",
                    n_procs: int = 4, memory: str = "4GB",
                    charge: int = 0, multiplicity: int = 2) -> str:
    """
    生成 EPR 参数计算的 Gaussian 输入文件。

    计算内容：
      - g-tensor (电子 g 张量)
      - Hyperfine coupling constants (超精细耦合常数 A)
      - Spin density distribution (自旋密度分布)

    multiplicity 必须 > 1 (至少是 doublet = 2)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if multiplicity < 2:
        raise ValueError("EPR needs multiplicity >= 2 (radical has unpaired electron)")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append(f" {atom.GetSymbol():<3s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")

    charge_mult = f"{charge} {multiplicity}"
    solvent_kw = SOLVENT_KEYWORDS.get(solvent.lower(), "")

    header = f"""%chk={name}.chk
%nprocshared={n_procs}
%mem={memory}
#p opt freq {functional}/{basis_set} {solvent_kw} nosymm

{name} — geometry optimization (UDFT)

{charge_mult}
""" + "\n".join(coords) + "\n\n"

    header += f"""--Link1--
%chk={name}.chk
%nprocshared={n_procs}
%mem={memory}
#p {functional}/{basis_set} prop=(epr,nmr) {solvent_kw} nosymm geom=allcheck guess=read

{name} — EPR/NMR properties (g-tensor, hyperfine)

{charge_mult}

"""

    return header


def build_ts_input(smiles_reactant: str, smiles_product: str,
                   name: str = "ts_search",
                   functional: str = "B3LYP", basis_set: str = "6-31G(d)",
                   solvent: str = "", n_procs: int = 4, memory: str = "4GB"):
    """
    生成过渡态搜索输入文件。
    需要提供反应物和产物的 SMILES，生成 Opt=TS 计算。

    Note: 过渡态搜索需要好的初始猜测，通常先用 GaussView 手动建模。
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    solvent_kw = SOLVENT_KEYWORDS.get(solvent.lower(), "")

    header = f"""%chk={name}.chk
%nprocshared={n_procs}
%mem={memory}
#p opt=(ts,calcfc,noeigen) freq {functional}/{basis_set} {solvent_kw} nosymm

{name} — Transition State Search
NOTE: You need to provide a good TS guess geometry below!

0 1
"""
    return header


def save_gjf(content: str, name: str):
    """保存 .gjf 文件到输入目录"""
    path = GAUSSIAN_INPUT_DIR / f"{name}.gjf"
    path.write_text(content, encoding="utf-8")
    print(f"  [GJF 已保存] {path}")
    return path


def run_gaussian(gjf_path: Path, timeout_hours: float = 24) -> Path:
    """
    在本地运行 Gaussian 计算。

    Returns
    -------
    output_path : .log 文件的路径
    """
    gaussian_exe = find_gaussian()
    if not gaussian_exe:
        raise RuntimeError(
            "找不到 Gaussian。请:\n"
            "  1) 把 .gjf 文件拖入 GaussView 提交\n"
            "  2) 或上传到课题组服务器计算\n"
            "  3) 计算完成后把 .log 文件拖回来，用 parse_nmr_output() / parse_epr_output() 解析"
        )

    output_path = GAUSSIAN_OUTPUT_DIR / f"{gjf_path.stem}.log"
    cmd = f'"{gaussian_exe}" < "{gjf_path}" > "{output_path}"'

    print(f"  [Gaussian] 开始计算: {gjf_path.name}")
    print(f"  [Gaussian] 预计时间: 数小时 (取决于分子大小和基组)")
    print(f"  [Gaussian] 输出: {output_path}")

    result = subprocess.run(cmd, shell=True, cwd=str(GAUSSIAN_OUTPUT_DIR),
                             timeout=timeout_hours * 3600)

    if result.returncode != 0:
        print(f"  [警告] Gaussian 退出码: {result.returncode}")

    return output_path


# ===========================================================================
# Part 2: 输出文件解析器
# ===========================================================================

def parse_nmr_output(log_path: str, reference: str = "TMS") -> dict:
    """
    解析 Gaussian NMR 输出文件，提取各原子的化学位移。

    Parameters
    ----------
    log_path : Gaussian .log 文件路径
    reference : 参考标准 ("TMS" 使用内置 TMS 参考值)

    Returns
    -------
    dict with:
      - h_shifts: {atom_label: chemical_shift_ppm}
      - c_shifts: {atom_label: chemical_shift_ppm}
      - isotropic_shielding: raw shielding constants
      - scf_energy: 总能量 (Hartree)
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

    result = {
        "h_shifts": {},
        "c_shifts": {},
        "n_shifts": {},
        "f_shifts": {},
        "isotropic_shielding": {},
        "scf_energy": None,
        "method": None,
    }

    # 提取 SCF 能量
    m = re.search(r"SCF Done:\s*E\([\w+]+\)\s*=\s*([-\d.]+)", text)
    if m:
        result["scf_energy"] = float(m.group(1))

    # 提取方法/基组
    m = re.search(r"#p?\s+(.+)", text)
    if m:
        result["method"] = m.group(1).strip()

    # 提取 NMR 各向同性屏蔽值
    # Gaussian 格式: "Atom  Element  Isotropic  ..."
    # 位于 "SCF GIAO Magnetic shielding tensor" 之后
    shielding_section = False
    for line in text.split("\n"):
        if "SCF GIAO Magnetic shielding tensor" in line:
            shielding_section = True
            continue

        if shielding_section:
            # 检测表格结束
            if line.strip().startswith("Eigenvalues") or "----" in line or not line.strip():
                continue
            if "End of Minotr" in line or "Save" in line:
                break

            # 解析数据行: "  1  H   Isotropic =  xx.xxxx   Anisotropy =  xx.xxxx"
            # 或: "  2  C   Isotropic =  xx.xxxx"
            m = re.search(r"(\d+)\s+(\w+)\s+Isotropic\s*=\s*([-\d.]+)", line)
            if m:
                idx = int(m.group(1))
                element = m.group(2)
                iso = float(m.group(3))
                label = f"{element}{idx}"
                result["isotropic_shielding"][label] = iso

    # 屏蔽常数 → 化学位移: δ = σ_ref - σ_calc
    for label, sigma in result["isotropic_shielding"].items():
        element = re.match(r"([A-Za-z]+)", label).group(1)
        sigma_ref = TMS_REFERENCE.get(element)
        if sigma_ref is None:
            continue
        shift = sigma_ref - sigma

        if element == "H":
            result["h_shifts"][label] = round(shift, 2)
        elif element == "C":
            result["c_shifts"][label] = round(shift, 1)
        elif element == "N":
            result["n_shifts"][label] = round(shift, 1)
        elif element == "F":
            result["f_shifts"][label] = round(shift, 1)

    n_h = len(result["h_shifts"])
    n_c = len(result["c_shifts"])
    print(f"  [NMR 解析] {Path(log_path).name}")
    print(f"    1H signals: {n_h}  |  13C signals: {n_c}")
    print(f"    SCF Energy: {result['scf_energy']:.6f} Hartree" if result['scf_energy'] else "")

    return result


def parse_epr_output(log_path: str) -> dict:
    """
    解析 Gaussian EPR 输出文件，提取 g 张量和超精细耦合常数。

    Returns
    -------
    dict with:
      - g_tensor: 3x3 or diagonal g values
      - g_iso: isotropic g value
      - hfc: [{"nucleus": ..., "isotropic_G": ..., "anisotropic_G": ...}, ...]
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

    result = {
        "g_tensor": None,
        "g_iso": None,
        "g_anisotropy": None,
        "hfc": [],
        "spin_density": {},
    }

    # --- g-tensor ---
    # 查找 "Electron spin g-tensor" 或 "g tensor"
    g_section = False
    g_lines = []
    for line in text.split("\n"):
        if "g tensor" in line.lower() or "electron spin g-tensor" in line.lower():
            g_section = True
            g_lines = []
            continue
        if g_section:
            if line.strip() == "" or "---" in line:
                g_section = False
                continue
            g_lines.append(line.strip())

    # 尝试从 g_lines 提取数值
    g_values = []
    for line in g_lines:
        nums = re.findall(r"([-\d.]+)", line)
        g_values.extend([float(n) for n in nums if 1.9 < float(n) < 3.0])

    if len(g_values) >= 3:
        result["g_tensor"] = g_values[:9] if len(g_values) >= 9 else g_values[:3]
        result["g_iso"] = round(sum(g_values[:3]) / 3, 5)
        result["g_anisotropy"] = round(max(g_values[:3]) - min(g_values[:3]), 5)

    # 备选：从 "Isotropic Fermi Contact Couplings" 附近的 g 值提取
    if result["g_iso"] is None:
        m = re.search(r"isotropic\s+g(?:-|\s*)value\s*[=:]\s*([\d.]+)", text, re.IGNORECASE)
        if m:
            result["g_iso"] = float(m.group(1))

    # --- 超精细耦合常数 ---
    # Gaussian 输出格式:
    # "Atom  Element   Fermi Contact   Spin Density"
    # "  1    H(1)      0.023456        0.001234"
    hfc_section = False
    for line in text.split("\n"):
        if "Fermi Contact" in line and "Spin Density" in line:
            hfc_section = True
            continue
        if hfc_section:
            if line.strip() == "" or "---" in line:
                continue
            if "Sum of" in line or "Eigenvalues" in line:
                break
            m = re.search(r"(\d+)\s+(\w+)\(\d+\)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                result["hfc"].append({
                    "atom_idx": int(m.group(1)),
                    "nucleus": m.group(2),
                    "fermi_contact": float(m.group(3)),
                    "spin_density": float(m.group(4)),
                })

    # 费米接触 → 各向同性超精细耦合常数 A_iso (Gauss)
    # A_iso = (8π/3) * g_e * μ_B * |Ψ(0)|² * ρ
    # Gaussian 输出的 Fermi contact 直接就是 A_iso / g_e 的某种形式
    # 实际使用: A_iso_in_G ≈ Fermi_Contact * g_iso
    for h in result["hfc"]:
        fc = h["fermi_contact"]
        g = result["g_iso"] or 2.0023
        h["A_iso_G"] = round(fc * g, 2)
        h["A_iso_MHz"] = round(fc * g * 2.8025, 2)  # 1 G = 2.8025 MHz for electron

    n_hfc = len(result["hfc"])
    print(f"  [EPR 解析] {Path(log_path).name}")
    print(f"    g_iso = {result['g_iso']}")
    print(f"    {n_hfc} hyperfine coupling constants found")
    for h in result["hfc"]:
        print(f"      {h['nucleus']}{h['atom_idx']}: A_iso = {h['A_iso_G']:.2f} G")

    return result


# ===========================================================================
# Part 3: 便捷工作流
# ===========================================================================

def predict_nmr_dft(smiles: str, **kwargs) -> dict:
    """
    一键生成 NMR 输入文件 + 尝试本地运行 + 解析结果。

    如果找不到 Gaussian，会生成并保存 .gjf 文件，打印提交指南。
    """
    name = kwargs.pop("name", smiles.replace(" ", "_")[:30])
    gjf_content = build_nmr_input(smiles, name=name, **kwargs)
    gjf_path = save_gjf(gjf_content, name)

    gaussian = find_gaussian()
    if gaussian:
        try:
            log_path = run_gaussian(gjf_path)
            return parse_nmr_output(str(log_path))
        except Exception as e:
            print(f"  [Gaussian 运行失败] {e}")
            print(f"  [手动] 请将 {gjf_path} 提交到集群，.log 文件拖回来解析")

    print(f"\n  === 手动提交流程 ===")
    print(f"  1. 打开 GaussView: C:\\G09W\\gview.exe")
    print(f"  2. 拖入: {gjf_path}")
    print(f"  3. Calculate → Gaussian → 设置队列/节点 → Submit")
    print(f"  4. 计算完成后，.log 文件拖回，调用 parse_nmr_output() 解析")
    return {"gjf_path": str(gjf_path), "status": "input_generated"}


print("[gaussian_interface] Ready.")
print("  - build_nmr_input(smiles) → .gjf 带 NMR GIAO 关键词")
print("  - build_epr_input(smiles)  → .gjf 带 EPR prop 关键词")
print("  - parse_nmr_output(log)    → 解析 NMR 化学位移")
print("  - parse_epr_output(log)    → 解析 g-tensor + HFC")
print(f"  - 输入文件目录: {GAUSSIAN_INPUT_DIR}")
print(f"  - 输出文件目录: {GAUSSIAN_OUTPUT_DIR}")

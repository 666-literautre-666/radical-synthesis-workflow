"""
UV-Vis Spectroscopy Prediction
================================
预测紫外-可见吸收光谱。

方法：
1. 经验规则 — 基于生色团（chromophore）加和规则快速估算 λ_max
2. TD-DFT — 生成 ORCA 输入文件，精确计算激发态

参考：
  - Woodward-Fieser 规则（共轭烯烃/烯酮）
  - Scott 规则（芳香羰基化合物）
  - 常见自由基生色团的实验 λ_max 范围
"""

import json
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    raise ImportError("需要 rdkit: pip install rdkit")

try:
    from scripts.plot_utils import ChemFigure, PREDICTIONS_DIR, FIGURES_DIR
except ModuleNotFoundError:
    from plot_utils import ChemFigure, PREDICTIONS_DIR, FIGURES_DIR

# ---------------------------------------------------------------------------
# 生色团数据库：常见发色团的 λ_max 范围
# ---------------------------------------------------------------------------

CHROMOPHORES = {
    "carbonyl": {
        "name": "羰基 n→π*",
        "smarts": "[CX3](=O)",
        "lambda_range": (270, 300),
        "epsilon": (10, 100),
        "note": "醛/酮的弱吸收带",
    },
    "alpha_beta_unsaturated_carbonyl": {
        "name": "α,β-不饱和羰基 π→π*",
        "smarts": "[C]=[C][CX3](=O)",
        "lambda_range": (210, 250),
        "epsilon": (10000, 20000),
        "note": "烯酮类强吸收",
    },
    "diene": {
        "name": "共轭二烯 π→π*",
        "smarts": "[C]=[C][C]=[C]",
        "lambda_range": (215, 230),
        "epsilon": (15000, 25000),
        "note": "丁二烯类，Woodward-Fieser 计算更准",
    },
    "benzene": {
        "name": "苯环 π→π* (B band)",
        "smarts": "c1ccccc1",
        "lambda_range": (250, 260),
        "epsilon": (100, 300),
        "note": "苯的特征弱吸收",
    },
    "nitro": {
        "name": "硝基 n→π*",
        "smarts": "[N+](=O)[O-]",
        "lambda_range": (270, 280),
        "epsilon": (10, 50),
        "note": "脂肪族硝基化合物",
    },
    "azo": {
        "name": "偶氮 n→π*",
        "smarts": "[N]=[N]",
        "lambda_range": (340, 360),
        "epsilon": (10, 50),
        "note": "偶氮化合物，自由基引发剂常见结构",
    },
    "nitroxide": {
        "name": "氮氧自由基 n→π*",
        "smarts": "N[O]",
        "lambda_range": (420, 460),
        "epsilon": (5, 20),
        "note": "TEMPO类弱可见光吸收（橙色）",
    },
    "quinone": {
        "name": "醌类 π→π* + n→π*",
        "smarts": "O=C1C=CC(=O)C=C1",
        "lambda_range": (240, 290),
        "epsilon": (15000, 30000),
        "note": "苯醌类",
    },
    "triaryl_carbon": {
        "name": "三芳甲基自由基",
        "smarts": "[C](c)(c)(c)",
        "lambda_range": (350, 550),
        "epsilon": (100, 5000),
        "note": "三苯甲基自由基可呈黄色到紫色",
    },
    "benzophenone": {
        "name": "二苯甲酮类 n→π*",
        "smarts": "O=C(c1ccccc1)c1ccccc1",
        "lambda_range": (330, 350),
        "epsilon": (100, 200),
        "note": "光引发剂常见结构",
    },
}

RADICAL_CHROMOPHORES = {
    "alkyl_radical": {
        "lambda_range": (200, 280),
        "color": "无色",
        "note": "烷基自由基无可见光吸收",
    },
    "benzyl_radical": {
        "lambda_range": (300, 320),
        "color": "无色到淡黄",
        "note": "苄基自由基 π→π*",
    },
    "phenoxyl_radical": {
        "lambda_range": (380, 420),
        "color": "黄色",
        "note": "苯氧自由基",
    },
    "nitroxide": {
        "lambda_range": (420, 460),
        "color": "橙色",
        "note": "TEMPO类",
    },
    "ketyl": {
        "lambda_range": (300, 350),
        "color": "无色到淡蓝",
        "note": "羰基阴离子自由基",
    },
    "semiquinone": {
        "lambda_range": (400, 500),
        "color": "红到紫色",
        "note": "半醌自由基阴离子",
    },
    "viologen": {
        "lambda_range": (600, 610),
        "color": "深蓝",
        "note": "紫精自由基阳离子",
    },
}


# ---------------------------------------------------------------------------
# 预测函数
# ---------------------------------------------------------------------------

def identify_chromophores(smiles: str) -> list[dict]:
    """
    识别分子中含有的生色团。

    返回:
        [{"name": "羰基 n→π*", "lambda_range": (270,300), "n_matches": 1}, ...]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    found = []
    for key, info in CHROMOPHORES.items():
        pattern = Chem.MolFromSmarts(info["smarts"])
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            found.append({
                "key": key,
                "name": info["name"],
                "lambda_range": info["lambda_range"],
                "epsilon": info["epsilon"],
                "n_matches": len(matches),
                "note": info["note"],
            })
    return found


def predict_uvvis(smiles: str) -> dict:
    """
    预测 UV-Vis 吸收。

    返回:
        {
            "smiles": ...,
            "chromophores": [...],
            "estimated_lambda_max_nm": (low, high),
            "likely_color": "无色" / "黄色" / "橙色" 等,
        }
    """
    chromophores = identify_chromophores(smiles)

    # 根据生色团估算 λ_max
    if not chromophores:
        lambda_est = (200, 220)
        color = "无色（真空紫外区）"
    else:
        # 取最长波长生色团
        max_lambda = max(c["lambda_range"][1] for c in chromophores)
        min_lambda = max(c["lambda_range"][0] for c in chromophores
                         if c["lambda_range"][1] >= max_lambda - 20)
        lambda_est = (min_lambda, max_lambda)

        # 根据 λ_max 推断颜色
        if max_lambda < 300:
            color = "无色"
        elif max_lambda < 400:
            color = "黄色"
        elif max_lambda < 500:
            color = "橙色/红色"
        elif max_lambda < 600:
            color = "紫色/蓝色"
        else:
            color = "绿色/深色"

    result = {
        "smiles": smiles,
        "chromophores": chromophores,
        "n_chromophores": len(chromophores),
        "estimated_lambda_max_nm": lambda_est,
        "likely_color": color,
        "method": "经验规则（生色团加和）",
        "note_for_accurate": "需要精确 λ_max 请用 TD-DFT: build_tddft_input()",
    }

    return result


# ---------------------------------------------------------------------------
# TD-DFT 接口：生成 ORCA 输入文件
# ---------------------------------------------------------------------------

def build_tddft_input(
    smiles: str,
    name: str = "uvvis_calc",
    functional: str = "B3LYP",
    basis: str = "def2-SVP",
    n_roots: int = 10,
    solvent: str = "ethanol",
    n_cores: int = 4,
) -> str:
    """
    生成 ORCA TD-DFT 输入文件内容，用于精确计算 UV-Vis。

    参数:
        smiles: 分子 SMILES
        name: 文件名前缀
        functional: 泛函（推荐 B3LYP 或 CAM-B3LYP）
        basis: 基组（def2-SVP 快速，def2-TZVP 精确）
        n_roots: 计算的激发态数量
        solvent: 溶剂（影响 λ_max 位移）
        n_cores: CPU 核心数

    返回:
        ORCA 输入文件内容（字符串）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(f"  {atom.GetSymbol():3s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")

    xyz_block = "\n".join(atoms)

    inp = f"""! {functional} {basis} TightSCF
! CPCM({solvent})
%tddft
  nroots {n_roots}
  maxdim 5
  tda false
end
%pal nprocs {n_cores} end
%maxcore 2000

* xyz 0 1
{xyz_block}
*"""
    return inp


def save_uvvis_inp(smiles: str, name: str = "uvvis_calc", **kwargs) -> Path:
    """
    生成并保存 ORCA UV-Vis 计算输入文件。

    返回: 输入文件路径
    """
    inp_content = build_tddft_input(smiles, name, **kwargs)
    inp_dir = Path(__file__).resolve().parent.parent / "data" / "uvvis_inputs"
    inp_dir.mkdir(parents=True, exist_ok=True)
    inp_path = inp_dir / f"{name}.inp"
    inp_path.write_text(inp_content)
    return inp_path


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------

def plot_uvvis_prediction(smiles: str, journal: str = "jacs"):
    """
    生成 UV-Vis 预测图表。

    输出: 预期吸收带的模拟谱图（棒状图或高斯展宽）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    result = predict_uvvis(smiles)

    safe_name = smiles[:30].replace(" ", "_").replace("/", "_")

    with ChemFigure(f"uvvis_{safe_name}", journal=journal, width="single") as cf:
        ax = cf.ax

        # 模拟谱：每个生色团一个高斯峰
        x = np.linspace(200, 800, 600)
        y_total = np.zeros_like(x)

        colors_cycle = plt.cm.Set2(np.linspace(0, 1, max(len(result["chromophores"]), 1)))

        for i, chrom in enumerate(result["chromophores"]):
            center = np.mean(chrom["lambda_range"])
            width = (chrom["lambda_range"][1] - chrom["lambda_range"][0]) / 2
            epsilon_mid = np.mean(chrom["epsilon"]) if chrom["epsilon"] else 100
            peak = epsilon_mid * np.exp(-((x - center) / (width + 1)) ** 2)
            y_total += peak

            ax.plot(x, peak, '--', color=colors_cycle[i % len(colors_cycle)],
                    alpha=0.5, linewidth=0.8,
                    label=f"{chrom['name']} (~{int(center)} nm)")

        # 总谱
        if len(result["chromophores"]) > 1:
            ax.plot(x, y_total, 'k-', linewidth=1.2, label='总预测谱')

        ax.fill_between(x, 0, y_total, alpha=0.15, color='steelblue')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Extinction Coefficient ε (L·mol⁻¹·cm⁻¹)")
        ax.set_title(f"UV-Vis Prediction: {smiles}")
        ax.legend(fontsize=6, loc='upper right')
        ax.set_xlim(200, 800)

        cf.data["wavelength"] = x.tolist()
        cf.data["absorption"] = y_total.tolist()
        cf.data["smiles"] = smiles
        cf.data["chromophores"] = result["chromophores"]


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("UV-Vis Predictor 测试")
    print("=" * 60)

    test_molecules = [
        ("C1=CC=C(C=C1)C=O", "苯甲醛"),
        ("CC(=O)C=C", "甲基乙烯基酮"),
        ("C1=CC=C(C=C1)N=NC1=CC=CC=C1", "偶氮苯"),
        ("CC1(C)CCCC(C)(C)N1[O]", "TEMPO"),
    ]

    for smiles, name in test_molecules:
        result = predict_uvvis(smiles)
        print(f"\n{name} ({smiles}):")
        print(f"  生色团: {[c['name'] for c in result['chromophores']]}")
        print(f"  λ_max 估计: {result['estimated_lambda_max_nm']} nm")
        print(f"  可能颜色: {result['likely_color']}")

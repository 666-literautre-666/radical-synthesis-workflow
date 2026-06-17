"""
Experimental Lab Guide — 实验操作助手
=======================================
输入产物 SMILES + 反应类型 → 输出完整实验流程建议

覆盖：
1. 反应条件提醒
2. 淬灭与后处理
3. 萃取溶剂推荐
4. 纯化方法（柱层析/重结晶/打浆/蒸馏）
5. 产物干燥
6. NMR/MS 送样建议
7. 安全警告

核心思想：
- 用 RDKit 计算分子属性（logP, MW, HBD/HBA）
- 用 SMARTS 检测官能团
- 用反应类型匹配后处理模板
- 所有输出都是"建议"，需要实验员自己判断

使用：
from scripts.lab_guide import lab_guide
guide = lab_guide("C1=CC=C(C=C1)C#N", reaction_type="ATRP")
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Crippen

# 接入项目现有模块
try:
    from scripts.reaction_predictor import (
        analyze_substrate, suggest_reaction_routes, predict_conditions,
        _estimate_bde, RADICAL_INITIATORS, RADICAL_REACTION_TYPES,
        CATALYSTS_MEDIATORS
    )
    from scripts.database import get_smarts_matches, query_similar_substrates
except ModuleNotFoundError:
    from reaction_predictor import (
        analyze_substrate, suggest_reaction_routes, predict_conditions,
        _estimate_bde, RADICAL_INITIATORS, RADICAL_REACTION_TYPES,
        CATALYSTS_MEDIATORS
    )
    from database import get_smarts_matches, query_similar_substrates

# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------

# 反应类型 → 后处理模板
WORKUP_TEMPLATES = {
    "ATRP": {
        "quench": "加入等体积溶剂稀释，暴露空气终止（O2 淬灭 Cu(I)）",
        "metal_removal": "过中性氧化铝短柱去除铜盐，或 5% EDTA 水溶液洗涤 3 次",
        "notes": "若产物颜色偏绿说明铜残留，需再过一次中性 Al2O3 柱",
    },
    "atom_transfer": {
        "quench": "加入等体积溶剂稀释，暴露空气或加少量水",
        "metal_removal": "若含金属催化剂：短硅胶柱过滤 / EDTA 洗涤",
    },
    "HAT": {
        "quench": "加水或饱和 NH4Cl 淬灭",
        "metal_removal": "通常不含金属，无需特殊处理",
        "notes": "若使用过氧化物引发剂，淬灭前确认无残留过氧化物（KI-淀粉试纸）",
    },
    "SET": {
        "quench": "加水淬灭，暴空终止",
        "metal_removal": "若含 Ru/Ir 光催化剂：旋蒸回收，或短硅胶柱分离",
        "notes": "光催化反应需用不透明容器收集，避免光照副反应",
    },
    "radical_addition": {
        "quench": "加饱和 NH4Cl 或水淬灭",
        "metal_removal": "若使用 Bu3SnH：氟化钾水溶液洗涤除锡",
    },
    "radical_cyclization": {
        "quench": "加水淬灭",
        "metal_removal": "若使用 Bu3SnH/AIBN：KF 洗涤 + 硅胶柱",
    },
    "HAS": {
        "quench": "加饱和 NaHCO3 淬灭（若使用 BF4 重氮盐）",
        "metal_removal": "通常无需特殊金属去除",
    },
}

# 溶剂极性表（用于萃取和柱层析）
EXTRACTION_GUIDE = {
    # logP 范围: (推荐萃取溶剂, 备用溶剂)
    "very_polar": (-99, 0.5, "水溶性产物", "正丁醇萃取 或 直接浓缩后柱层析"),
    "polar": (0.5, 1.5, "乙酸乙酯", "DCM:IPA=9:1"),
    "medium": (1.5, 3.0, "二氯甲烷 (DCM)", "乙酸乙酯"),
    "nonpolar": (3.0, 5.0, "二氯甲烷 (DCM)", "石油醚"),
    "very_nonpolar": (5.0, 99, "二氯甲烷 (DCM) 或 石油醚", "正己烷"),
}

# 官能团安全警告
DANGER_GROUPS = {
    "[N+]#[N-]": "重氮基：禁止蒸干！浓缩至小体积直接用于下一步",
    "[N-]=[N+]=N": "叠氮基：微量操作！禁止金属刮刀接触，禁止加热",
    "O-O": "过氧键：避免高温、撞击、摩擦，TLC 确认无过氧化物残留",
    "[CX3H1](=O)": "醛基：易氧化，N2 保护，尽快投下一步或冰箱储存",
    "C#N": "腈基：注意通风，避免与强酸混合（可能产生 HCN）",
    "[SH]": "巯基：易氧化成二硫键，需要时加 TCEP 或 DTT",
    "I": "碘代物：避光保存，见光缓慢分解",
    "[N+](=O)[O-]": "硝基：放大反应注意安全，可能剧烈分解",
}

# NMR 溶剂推荐
NMR_SOLVENT_GUIDE = [
    # (条件, 溶剂, 残留峰 ppm)
    ("default", "CDCl3 (氯仿-d)", "7.26 (1H), 77.16 (13C)"),
    ("含 -OH/-NH/-COOH", "DMSO-d6", "2.50 (1H), 39.52 (13C)"),
    ("含 -OH 但 DMSO 难除", "Acetone-d6", "2.05 (1H), 29.84 (13C)"),
    ("含芳香杂环、含 F", "CDCl3 或 Acetone-d6", "—"),
    ("仅溶于水", "D2O", "4.79 (1H)"),
    ("logP > 5 非极性", "CDCl3 或 Benzene-d6", "—"),
]


# ---------------------------------------------------------------------------
# 溶剂相关过柱技巧
# ---------------------------------------------------------------------------

SOLVENT_COLUMN_TIPS = {
    "DMF": {
        "issue": "DMF 沸点高 (153°C)、极性大，残留会干扰柱层析",
        "fix": [
            "柱层析前彻底除 DMF：水洗 3 次 + 饱和食盐水洗 1 次",
            "旋蒸后用 EtOAc 共沸带出残留 DMF（×2~3 次）",
            "若含胺类产物：硅胶柱用 1% Et3N 预活化（冲 2 CV 含 1% Et3N 的洗脱剂，再用中性洗脱剂冲 2 CV）",
            "原因：DMF 常含微量二甲胺（碱性），硅胶酸性位点会强烈吸附碱性产物导致拖尾",
        ],
    },
    "DMSO": {
        "issue": "DMSO 沸点 189°C，极难除净",
        "fix": [
            "水洗 5 次以上除去 DMSO（DMSO 与水互溶）",
            "若产物水溶性好：不萃取，直接浓缩后柱层析，用 EtOAc 共沸带 DMSO",
            "残留 DMSO 在柱子上会拖慢所有组分，需先用低极性洗脱剂冲干净 DMSO",
        ],
    },
    "DCM": {"issue": "", "fix": []},  # 无特殊问题
    "THF": {
        "issue": "THF 含 BHT 稳定剂（抗氧化剂），过柱后会留在产物里",
        "fix": ["THF 需新鲜蒸馏或用无稳定剂 THF", "若有 BHT 残留：柱层析可分离（BHT 极性低先出）"],
    },
    "Toluene": {"issue": "", "fix": []},
    "MeCN": {"issue": "", "fix": []},
    "Water": {"issue": "", "fix": []},
    "Benzene": {"issue": "苯剧毒", "fix": ["尽量避免使用苯，用甲苯替代"]},
}


def _get_column_modifier_tips(smiles: str) -> list[str]:
    """
    根据分子结构判断是否需要 Et3N/AcOH 等柱层析改性剂。
    """
    tips = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return tips

    # 检测碱性官能团（含 N 的胺、含氮杂环）
    amine = Chem.MolFromSmarts("[NX3;H2,H1,H0]")  # 胺
    pyridine = Chem.MolFromSmarts("c1ncccc1")       # 吡啶类
    imidazole = Chem.MolFromSmarts("c1nccn1")       # 咪唑类

    has_basic = False
    if amine and mol.HasSubstructMatch(amine):
        has_basic = True
    if pyridine and mol.HasSubstructMatch(pyridine):
        has_basic = True
    if imidazole and mol.HasSubstructMatch(imidazole):
        has_basic = True

    if has_basic:
        tips.append(
            "产物含碱性基团（胺/含氮杂环）："
            "洗脱剂中加 0.5-1% Et3N 防拖尾，"
            "或先用含 1% Et3N 的洗脱剂冲 2 CV 预活化硅胶柱"
        )

    # 检测酸性官能团
    carboxylic = Chem.MolFromSmarts("C(=O)O")       # 羧酸
    phenol = Chem.MolFromSmarts("cO")                # 酚羟基
    sulfonic = Chem.MolFromSmarts("S(=O)(=O)O")      # 磺酸

    has_acidic = False
    if carboxylic and mol.HasSubstructMatch(carboxylic):
        has_acidic = True
    if phenol and mol.HasSubstructMatch(phenol):
        has_acidic = True
    if sulfonic and mol.HasSubstructMatch(sulfonic):
        has_acidic = True

    if has_acidic:
        tips.append(
            "产物含酸性基团（羧酸/酚）："
            "洗脱剂中加 1-2% AcOH 防拖尾，"
            "或用 DCM-MeOH 体系替代 PE-EtOAc"
        )

    return tips


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def _get_mol(smiles: str):
    """RDKit 分子对象 + 属性计算"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 加氢后算 3D 构象
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_h)
    return mol_h


def _estimate_physical_state(mol) -> str:
    """估算产物物理状态"""
    mw = Descriptors.MolWt(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # 简单经验规则
    if mw < 300 and rot_bonds <= 3 and hbd == 0 and hba <= 2:
        return "liquid"
    elif mw > 400 or (hbd + hba) >= 4:
        return "solid"
    elif mw < 300 and rot_bonds <= 5:
        return "likely_liquid"
    else:
        return "likely_solid"


def _recommend_extraction(logp: float) -> dict:
    """根据 logP 推荐萃取溶剂"""
    for key, (lo, hi, solvent, backup) in EXTRACTION_GUIDE.items():
        if lo <= logp <= hi:
            return {
                "logP": round(logp, 2),
                "primary": solvent,
                "backup": backup,
            }
    return {"logP": round(logp, 2), "primary": "乙酸乙酯", "backup": "DCM"}


def _recommend_purification(mol, logp: float, physical_state: str,
                            reaction_type: str, contains_metal: bool,
                            is_radical: bool = True) -> dict:
    """推荐纯化方法"""
    mw = Descriptors.MolWt(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    steps = []
    method = ""

    # 含金属催化剂 → 先除金属
    if contains_metal:
        steps.append("1. 先过短硅胶柱或中性 Al2O3 柱除去金属")

    # 纯化方法推荐（自由基化学优先柱层析，避免重结晶加热）
    mol_hbd = Descriptors.NumHDonors(mol)
    mol_hba = Descriptors.NumHAcceptors(mol)
    if "solid" in physical_state and mw < 600 and (mol_hbd + mol_hba) <= 2 and rot_bonds <= 3:
        # 只有简单小分子固体才推荐重结晶
        method = "重结晶（或柱层析）"
        steps.append("2. 可尝试重结晶: DCM 溶解 → 缓慢加石油醚至浑浊 → 冰箱静置")
        steps.append("   若重结晶不析出 → 改用柱层析（见下）")
        steps.append("   注意：避免加热溶解（自由基产物可能热分解）")
    elif "solid" in physical_state and mw >= 600:
        method = "柱层析"
        steps.append("2. 硅胶柱层析 (200-300 目硅胶)")
        if logp < 1.5:
            steps.append("   洗脱剂: 乙酸乙酯:石油醚 = 1:3 → 1:1 梯度")
        elif logp < 3:
            steps.append("   洗脱剂: 乙酸乙酯:石油醚 = 1:10 → 1:5 梯度")
        else:
            steps.append("   洗脱剂: 纯石油醚 → 石油醚:乙酸乙酯 = 20:1 梯度")
        steps.append("   若 RF 值 < 0.2: 加大乙酸乙酯比例，或加 1% Et3N")
        steps.append("   若拖尾严重: 加 1% AcOH 或更换为 DCM-MeOH 体系")
    # 默认柱层析（芳香/多官能团产物过柱最可靠）
    if steps:  # 已有金属去除步骤
        steps.append("")
    else:
        # 无金属去除
        pass

    if is_radical:
        method = method if method else "柱层析（不推荐蒸馏，自由基产物热不稳定）"
    else:
        method = method if method else "柱层析"

    steps.append(f"{'2' if not contains_metal else ''} 硅胶柱层析 (200-300 目硅胶)".lstrip())
    if logp < 1.5:
        steps.append("   洗脱剂: PE:DCM=3:1 → DCM 梯度 或 EtOAc:PE=1:3 → 1:1")
    elif logp < 3:
        steps.append("   洗脱剂: PE:DCM=5:1 → 3:1 梯度 或 PE:EtOAc=10:1 → 5:1")
    else:
        steps.append("   洗脱剂: PE:DCM=10:1 → 5:1 梯度 或 纯PE → PE:EtOAc=20:1")
    steps.append("   若含碱性基团: 加 0.5-1% Et3N 防拖尾")
    steps.append("   TLC 监控，目标 Rf=0.2-0.3，合并纯品旋蒸")
    if is_radical:
        steps.append("   产物旋蒸时水浴温度 ≤30°C，避免长时间加热")
    else:
        steps.append("   旋蒸温度常规操作即可")

    return {"method": method, "steps": steps}


def _recommend_nmr_solvent(mol, logp: float) -> dict:
    """推荐 NMR 溶剂"""
    mol_unh = Chem.RemoveHs(mol)
    hbd = Descriptors.NumHDonors(mol_unh)
    mw = Descriptors.MolWt(mol_unh)

    # 检查特定官能团
    patterns = {
        "OH_NH": "[OH,NH]",
        "COOH": "C(=O)O",
        "heteroaromatic": "c1ncccc1",
    }

    has_polar_H = hbd > 0

    if has_polar_H and mw < 400:
        # 有活泼氢，用 DMSO
        return {"solvent": "DMSO-d6", "residual_peak": "2.50 (1H), 39.52 (13C)",
                "reason": "含活泼氢，DMSO-d6 不会交换"}
    elif logp > 5:
        return {"solvent": "CDCl3", "residual_peak": "7.26 (1H), 77.16 (13C)",
                "reason": "非极性分子，氯仿溶解性好"}
    elif logp < 0:
        return {"solvent": "D2O 或 DMSO-d6", "residual_peak": "4.79 (D2O) / 2.50 (DMSO)",
                "reason": "水溶性分子"}
    else:
        return {"solvent": "CDCl3", "residual_peak": "7.26 (1H), 77.16 (13C)",
                "reason": "默认，溶解性通常好"}


def _detect_dangers(mol) -> list[dict]:
    """检测危险官能团"""
    warnings = []
    mol_unh = Chem.RemoveHs(mol)
    for smarts, warning in DANGER_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol_unh.HasSubstructMatch(pattern):
            warnings.append({"smarts": smarts, "warning": warning})
    return warnings


def _estimate_scale_notes(reaction_type: str) -> list[str]:
    """根据反应类型给出放大注意事项"""
    notes = {
        "ATRP": [
            "Cu(I) 对氧敏感：严格除氧（冻融脱气 ×3 或 N2 sparge ≥30 min）",
            "引发剂 (如 EBiB) 需精确称量，建议配成甲苯溶液再注射",
            "温度控制关键：反应温度 ±2°C 内波动（过高导致终止反应）",
            "建议 mol 比：Substrate:Initiator:CuBr:PMDETA = 100:1:1:1",
        ],
        "HAT": [
            "过氧化物引发剂需注意安全：反应结束后用 KI-淀粉试纸确认无残留",
            "若使用 TBHP (70%水溶液)：折算有效浓度",
            "自由基 HAT 反应通常需要底物:自由基前体 = 1:2~5",
            "光照 HAT：避免用 UV 透过率低的玻璃容器",
        ],
        "SET": [
            "光催化剂用量通常 1-2 mol%，过量反而降低效率",
            "LED 光源需确认波长匹配光催化剂吸收峰",
            "反应管离光源距离控制在 1-3 cm",
            "若使用家用灯泡：功率 ≥ 20W，蓝光 LED 460nm 最优",
        ],
        "radical_addition": [
            "烯烃/炔烃需新鲜蒸馏（避免过氧化物积累）",
            "自由基加成反应速率取决于双键缺电子程度",
            "缺电子烯烃（丙烯酸酯类）反应快，富电子烯烃需较长时间",
        ],
        "radical_cyclization": [
            "Bu3SnH 有剧毒：手套箱操作，废物单独收集",
            "现代替代：Tris(TMS)3SiH（毒性低，价格高）",
            "5-exo-trig 优于 6-endo-trig：不需特别调控即优先关五元环",
            "稀释条件 (0.01-0.05 M) 有利分子内环化，抑制分子间反应",
            "AIBN 需缓慢滴加 (syringe pump, 4-8 h)，维持自由基低浓度",
        ],
        "HAS": [
            "芳基重氮盐不稳定：现制现用，低温保存 (0-5°C)",
            "重氮化反应在 0-5°C 进行：NaNO2 水溶液缓慢滴加",
            "重氮盐禁止蒸干：浓缩后直接下一步",
        ],
    }
    return notes.get(reaction_type, ["注意除氧除水，严格无水无氧操作"])


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def lab_guide(product_smiles: str, reaction_type: str = "HAT",
              scale: str = "0.5 mmol", include_orca: bool = False,
              is_radical: bool = True) -> dict:
    """
    输入产物 SMILES + 反应类型 → 输出完整实验流程建议。

    参数:
        product_smiles: 目标产物 SMILES
        reaction_type: 反应类型 (ATRP/atom_transfer/HAT/SET/
                       radical_addition/radical_cyclization/HAS)
        scale: 反应规模 ("0.1 mmol" / "1 mmol" / "10 mmol" 等)

    返回:
        {
            "product_info": {...},
            "reaction_notes": [...],
            "workup": {...},
            "extraction": {...},
            "purification": {...},
            "drying": "...",
            "characterization": {...},
            "safety_warnings": [...],
        }
    """
    mol = _get_mol(product_smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {product_smiles}"}

    mol_unh = Chem.RemoveHs(mol)

    # 1. 产物基本信息
    mw = round(Descriptors.MolWt(mol_unh), 2)
    logp = round(Crippen.MolLogP(mol_unh), 2)
    rot_bonds = Descriptors.NumRotatableBonds(mol_unh)
    hbd = Descriptors.NumHDonors(mol_unh)
    hba = Descriptors.NumHAcceptors(mol_unh)
    physical_state = _estimate_physical_state(mol_unh)

    product_info = {
        "smiles": product_smiles,
        "molecular_weight": mw,
        "logP": logp,
        "rotatable_bonds": rot_bonds,
        "H_donors": hbd,
        "H_acceptors": hba,
        "estimated_state": physical_state,
    }

    # 2. 反应注意事项
    reaction_notes = _estimate_scale_notes(reaction_type)

    # 3. 后处理
    workup = WORKUP_TEMPLATES.get(reaction_type, WORKUP_TEMPLATES["HAT"]).copy()

    # 判断是否含金属
    metal_containing_types = ["ATRP", "atom_transfer", "SET"]
    contains_metal = reaction_type in metal_containing_types
    workup["contains_metal"] = contains_metal

    # 4. 萃取推荐
    extraction = _recommend_extraction(logp)

    # 5. 是否需要萃取？
    if logp < 0.5:
        extraction["needed"] = False
        extraction["suggestion"] = "产物水溶性高，不建议萃取。直接浓缩后柱层析或重结晶。"
    else:
        extraction["needed"] = True
        extraction["suggestion"] = (
            f"用{extraction['primary']}萃取 3 次 → 合并有机相 → "
            f"饱和食盐水洗 1 次 → 无水 Na2SO4 或 MgSO4 干燥"
        )

    # 6. 纯化
    purification = _recommend_purification(
        mol_unh, logp, physical_state, reaction_type, contains_metal, is_radical
    )

    # 如果产物极性特别大且不是固体，建议打浆
    if logp < 0.5 and "solid" not in physical_state:
        purification["alternative"] = "可尝试打浆（加不良溶剂研磨）: 加少量乙醚或正己烷研磨产物"

    # 6b. 柱层析改性剂建议（检测碱性/酸性官能团）
    modifier_tips = _get_column_modifier_tips(product_smiles)
    if modifier_tips:
        purification["modifier_tips"] = modifier_tips

    # 7. 干燥
    if physical_state in ("liquid", "likely_liquid"):
        if is_radical:
            drying = (
                "旋蒸浓缩时水浴 ≤30°C → 减压油泵抽干 (0.1-1 mmHg, RT, 1-2 h) "
                "→ N2 回填 → 称重。注意：自由基产物避免长时间加热！"
            )
        else:
            drying = (
                "旋蒸浓缩 (水浴 ≤40°C) → 减压油泵抽干 (0.1-1 mmHg, RT, 1-2 h) "
                "→ 称重。非自由基产物正常操作即可。"
            )
    else:
        if is_radical:
            drying = (
                "旋蒸浓缩时水浴 ≤30°C → 油泵抽干至恒重 (0.1-1 mmHg, RT, 2-4 h) "
                "→ 称重 → 计算收率。固体产物可用冷正己烷洗涤除杂。"
            )
        else:
            drying = (
                "旋蒸浓缩 → 油泵抽干至恒重 (0.1-1 mmHg, RT, 2-4 h) "
                "→ 称重 → 计算收率。可真空干燥箱 40°C 过夜。"
            )

    # 8. 表征
    nmr_solvent = _recommend_nmr_solvent(mol, logp)
    characterization = {
        "nmr_1H": f"取 5-10 mg 产物溶于 0.6 mL {nmr_solvent['solvent']}，测 1H NMR",
        "nmr_13C": f"同一样品测 13C NMR（扫描 ≥64 次提高信噪比）",
        "nmr_solvent_reason": nmr_solvent.get("reason", ""),
        "ms": "ESI-MS (若含可电离基团) 或 EI-MS (若 MW < 400 且不含可电离基团)",
        "ir": "若产物为固体且有 C=O/OH/NH 官能团，测 IR 确认官能团",
    }

    # ORCA DFT 预测（可选，耗时较长）
    orca_info = {}
    if include_orca:
        try:
            from scripts.orca_interface import build_nmr_input, build_epr_input
            nmr_inp = build_nmr_input(product_smiles)
            epr_inp = build_epr_input(product_smiles)
            orca_info = {
                "nmr_input": "NMR 输入文件已生成，运行 ORCA 后可得精确化学位移",
                "epr_input": "EPR 输入文件已生成，运行 ORCA 后可得 g 张量 + 超精细耦合",
                "note": ("ORCA DFT 计算耗时较长（数小时），建议后台运行。\n"
                         f"  NMR: C:/ORCA_6.1.1/orca.exe {product_smiles[:20]}_nmr.inp > {product_smiles[:20]}_nmr.out\n"
                         "  EPR: 同上，替换为 _epr 文件"),
                "functional": "建议泛函: B3LYP (NMR), PBE0 (EPR)",
                "basis": "建议基组: pcSseg-2 (NMR), def2-TZVP (EPR)",
            }
        except Exception as e:
            orca_info = {"error": f"ORCA 输入生成失败: {e}"}

    characterization["orca"] = orca_info

    # 9. 安全警告
    safety_warnings = _detect_dangers(mol)

    # 10. 组装
    guide = {
        "product_info": product_info,
        "reaction_type": reaction_type,
        "scale": scale,
        "reaction_notes": reaction_notes,
        "workup": workup,
        "extraction": extraction,
        "purification": purification,
        "drying": drying,
        "characterization": characterization,
        "safety_warnings": safety_warnings,
    }

    return guide


# ---------------------------------------------------------------------------
# 格式化输出（方便打印阅读）
# ---------------------------------------------------------------------------

def print_guide(guide: dict):
    """美化打印实验指南"""
    if "error" in guide:
        print(f"[错误] {guide['error']}")
        return

    pi = guide["product_info"]
    print("=" * 60)
    print(f"  实验操作指南：{pi['smiles']}")
    print("=" * 60)

    print(f"\n【产物信息】")
    print(f"  MW: {pi['molecular_weight']} | logP: {pi['logP']} | "
          f"估计状态: {pi['estimated_state']}")

    print(f"\n【反应类型】{guide['reaction_type']}")
    print(f"  规模: {guide['scale']}")
    for note in guide["reaction_notes"]:
        print(f"  → {note}")

    wu = guide["workup"]
    print(f"\n【后处理】")
    print(f"  淬灭: {wu.get('quench', '加水淬灭')}")
    if wu.get("contains_metal"):
        print(f"  除金属: {wu.get('metal_removal', '')}")

    ex = guide["extraction"]
    print(f"\n【萃取】")
    if ex.get("needed"):
        print(f"  推荐溶剂: {ex['primary']} (logP={ex['logP']})")
        print(f"  备用: {ex['backup']}")
        print(f"  操作: {ex['suggestion']}")
    else:
        print(f"  {ex.get('suggestion', '无需萃取')}")

    pur = guide["purification"]
    print(f"\n【纯化】推荐方法: {pur['method']}")
    for step in pur.get("steps", []):
        print(f"  {step}")
    if "alternative" in pur:
        print(f"  备用方案: {pur['alternative']}")

    print(f"\n【干燥】")
    print(f"  {guide['drying']}")

    char = guide["characterization"]
    print(f"\n【表征】")
    print(f"  NMR: {char['nmr_1H']}")
    print(f"  MS: {char['ms']}")

    sw = guide["safety_warnings"]
    if sw:
        print(f"\n[!! 安全警告]")
        for w in sw:
            print(f"  {w['warning']}")

    print("\n" + "=" * 60)
    print("  以上为建议性操作指南，请根据实际情况调整。")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 统一入口：链接 lab_guide + reaction_predictor + database
# ---------------------------------------------------------------------------

def _auto_detect_reaction_type(smarts_hits: list) -> str:
    """根据 SMARTS 匹配结果自动判断主要反应类型"""
    families = [h["reaction_family"] for h in smarts_hits]
    if not families:
        return "HAT"  # 默认

    from collections import Counter
    count = Counter(families)

    # 按优先级：atom_transfer（最确定）> HAS > radical_addition > HAT > SET
    priority = ["atom_transfer", "HAS", "radical_addition", "radical_cyclization", "HAT", "SET", "spin_trap"]
    for p in priority:
        if p in count:
            return p
    return families[0]


def full_guide(product_smiles: str, substrate_smiles: str = None,
                reaction_type: str = None, scale: str = "0.5 mmol",
                include_orca: bool = False) -> dict:
    """
    一站式实验指南：预测 + 操作 + 安全全覆盖。

    输入产物 SMILES → 自动分析底物 → 自动识别反应类型 →
    匹配规则 → 推荐条件 → 给出实验操作。

    参数:
        product_smiles: 目标产物 SMILES
        substrate_smiles: 底物 SMILES，不提供则用 product_smiles 作为底物
        reaction_type: 手动指定反应类型。不指定则自动识别
        scale: 反应规模

    返回:
        完整实验指南，包含预测模块的所有输出 + 实验操作步骤
    """
    sub_smiles = substrate_smiles or product_smiles

    # --- 阶段一：底物分析 ---
    substrate = analyze_substrate(sub_smiles)

    # --- 阶段二：SMARTS 规则匹配 ---
    smarts_hits = get_smarts_matches(sub_smiles)

    # --- 阶段三：自动识别反应类型 ---
    if reaction_type is None:
        reaction_type = _auto_detect_reaction_type(smarts_hits)

    result = {
        "product_smiles": product_smiles,
        "substrate_smiles": sub_smiles,
        "reaction_type": reaction_type,
        "auto_detected": (reaction_type is None or reaction_type == _auto_detect_reaction_type(smarts_hits)),
        "scale": scale,
    }

    result["substrate_analysis"] = substrate
    result["smarts_matches"] = smarts_hits
    result["reaction_families"] = list(set(h["reaction_family"] for h in smarts_hits))

    # --- 阶段四：BDE 估算 ---
    bde_info = _estimate_bde(sub_smiles)
    result["bde_estimate"] = bde_info

    # --- 阶段五：数据库相似底物 ---
    try:
        similar = query_similar_substrates(sub_smiles, limit=5)
    except Exception:
        similar = []
    result["similar_in_db"] = similar

    # --- 阶段六：反应条件推荐 ---
    route = suggest_reaction_routes(sub_smiles, reaction_type)
    result["recommended_initiators"] = route.get("recommended_initiators", [])
    result["recommended_catalysts"] = route.get("recommended_catalysts", [])
    result["suggested_conditions"] = route.get("suggested_conditions", {})

    # --- 阶段七：实验操作指南 ---
    is_radical = len(smarts_hits) > 0
    lab = lab_guide(product_smiles, reaction_type=reaction_type, scale=scale,
                     include_orca=include_orca, is_radical=is_radical)
    result["lab_guide"] = lab

    # --- 结论：值不值得做 ---
    bde_kcal = bde_info.get("estimated_bde_kcal", 100)
    n_smarts = len(smarts_hits)
    if bde_kcal < 90 and n_smarts >= 2:
        verdict = "推荐合成：BDE 低 + 有明确自由基反应位点"
    elif bde_kcal < 95 or n_smarts >= 1:
        verdict = "可以尝试：反应有一定可行性，建议查阅类似底物文献确认"
    else:
        verdict = "风险较高：BDE 较高且无明确自由基位点，建议先做 DFT 计算确认"

    if similar:
        best = similar[0]
        verdict += f"。数据库中有类似底物：{best.get('substrate_name', best.get('smiles', ''))} (收率 {best.get('yield_percent', '?')}%)"
    else:
        verdict += "。数据库中暂无类似底物记录。"

    result["verdict"] = verdict

    return result


def quick_guide(product_smiles: str, substrate_smiles: str = None,
                initiator: str = "", catalyst: str = "",
                solvent: str = "", temperature: str = "",
                scale: str = "0.5 mmol", show_details: bool = False) -> dict:
    """
    精简版：你做实验前快速查"值不值得做 + 怎么后处理"。

    你已经知道要用什么引发剂/催化剂/溶剂/温度了，
    不需要程序推荐。只需要：

    1. 分析底物（反应位点在哪里）
    2. 估 BDE（键好不好断）
    3. 判断可行性（值得做吗？）
    4. 给后处理方案（怎么萃取？怎么纯化？怎么表征？）
    5. 安全警告

    参数:
        product_smiles: 产物 SMILES
        substrate_smiles: 底物 SMILES
        initiator: 你要用的引发剂（AIBN/BPO/...）
        catalyst: 你要用的催化剂（Cu/PMDETA/...）
        solvent: 你要用的溶剂（Toluene/MeCN/...）
        temperature: 反应温度（"80°C"/"RT"/...）
        scale: 反应规模
    """
    sub_smiles = substrate_smiles or product_smiles

    # --- 1. 底物分析 ---
    substrate = analyze_substrate(sub_smiles)

    # --- 2. SMARTS 匹配 ---
    smarts_hits = get_smarts_matches(sub_smiles)

    # --- 3. BDE ---
    bde_info = _estimate_bde(sub_smiles)
    bde_kcal = bde_info.get("estimated_bde_kcal", 100)

    # --- 4. 数据库 ---
    try:
        similar = query_similar_substrates(sub_smiles, limit=3)
    except Exception:
        similar = []

    # --- 5. 可行性判断 ---
    reasons = []
    if bde_kcal < 90:
        reasons.append(f"BDE≈{bde_kcal} kcal/mol，键容易断裂")
    elif bde_kcal < 96:
        reasons.append(f"BDE≈{bde_kcal} kcal/mol，中等强度")
    else:
        reasons.append(f"BDE≈{bde_kcal} kcal/mol，键较强，可能需要高温或强引发剂")

    n_smarts = len(smarts_hits)
    if n_smarts >= 2:
        reasons.append(f"匹配 {n_smarts} 条 SMARTS 规则，有明确反应位点")
    elif n_smarts >= 1:
        reasons.append(f"匹配 {n_smarts} 条规则: {smarts_hits[0]['name']}")
    else:
        reasons.append("未匹配自由基反应规则，反应位点不明确")

    if similar:
        reasons.append(f"数据库有类似底物: {similar[0].get('substrate_name', similar[0]['smiles'])}")

    if bde_kcal < 90 and n_smarts >= 1:
        verdict = "推荐尝试"
        confidence = "高"
    elif bde_kcal < 96 or n_smarts >= 1:
        verdict = "可以尝试"
        confidence = "中"
    else:
        verdict = "风险较高，建议先做 DFT 计算"
        confidence = "低"

    # --- 6. 溶剂专属提醒 ---
    solvent_tips = []
    if solvent.upper() in SOLVENT_COLUMN_TIPS:
        tips = SOLVENT_COLUMN_TIPS[solvent.upper()]
        if tips.get("issue"):
            solvent_tips = [f"溶剂提醒: {tips['issue']}"] + tips["fix"]

    # --- 7. 实验操作指南 ---
    reaction_family = _auto_detect_reaction_type(smarts_hits)
    is_radical = len(smarts_hits) > 0  # SMARTS 匹配到自由基规则 = 自由基反应
    lab = lab_guide(product_smiles, reaction_type=reaction_family, scale=scale,
                     is_radical=is_radical)

    return {
        "product_smiles": product_smiles,
        "substrate_smiles": sub_smiles,
        "your_conditions": {
            "initiator": initiator or "未指定",
            "catalyst": catalyst or "未指定",
            "solvent": solvent or "未指定",
            "temperature": temperature or "未指定",
            "scale": scale,
        },
        "substrate_analysis": substrate,
        "bde_estimate": bde_info,
        "smarts_matches": smarts_hits,
        "similar_in_db": similar,
        "feasibility": {
            "verdict": verdict,
            "confidence": confidence,
            "reasons": reasons,
        },
        "solvent_tips": solvent_tips if show_details else [],
        "lab_guide": lab,
        "show_details": show_details,
    }


def print_quick_guide(result: dict):
    """美化打印快速实验指南"""
    print("=" * 60)
    print(f"  快速实验评估: {result.get('product_smiles','')}")
    print("=" * 60)

    sub = result["substrate_analysis"]
    print(f"\n  【底物】{result['substrate_smiles']}")
    if "error" not in sub:
        print(f"  MW: {sub.get('molecular_weight','?')} | 分子式: {sub.get('formula','?')}")
        for s in sub.get("reactive_sites", [])[:3]:
            print(f"  反应位点: {s['site']} — {s['type']} ({s['reactivity']})")

    bde = result["bde_estimate"]
    print(f"\n  【BDE】{bde.get('estimated_bde_kcal','?')} kcal/mol — {bde.get('label','?')}")

    sm = result["smarts_matches"]
    print(f"  【SMARTS】匹配 {len(sm)} 条: {', '.join(h['name'] for h in sm[:4])}")

    yc = result["your_conditions"]
    print(f"\n  【你的条件】")
    print(f"  引发剂: {yc['initiator']}  |  催化剂: {yc['catalyst']}")
    print(f"  溶剂: {yc['solvent']}  |  温度: {yc['temperature']}")

    feas = result["feasibility"]
    print(f"\n  【可行性】{feas['verdict']} (置信度: {feas['confidence']})")
    for r in feas["reasons"]:
        print(f"  + {r}")

    lab = result["lab_guide"]
    if "error" not in lab:
        ex = lab["extraction"]
        print(f"\n  【后处理】")
        print(f"  萃取: {ex.get('primary','?')} (logP={ex.get('logP','?')})")
        print(f"  纯化: {lab['purification']['method']}")
        char = lab["characterization"]
        print(f"  NMR: {char['nmr_1H']}")

    sw = lab.get("safety_warnings", [])
    if sw:
        print(f"\n  [!! 安全] {', '.join(w['warning'] for w in sw)}")

    # 溶剂和改性剂建议（仅在 show_details=True 时显示）
    if result.get("show_details"):
        st = result.get("solvent_tips", [])
        if st:
            print(f"\n  [溶剂注意事项]")
            for t in st:
                print(f"  {t}")
        mod = lab.get("purification", {}).get("modifier_tips", [])
        if mod:
            print(f"\n  [柱层析改性剂]")
            for m in mod:
                print(f"  {m}")

    print("=" * 60)


def print_full_guide(result: dict):
    """美化打印一站式实验指南"""
    print("=" * 65)
    print("  自由基合成一站式实验指南")
    print("=" * 65)

    print(f"\n  【底物】{result['substrate_smiles']}")
    print(f"  【产物】{result['product_smiles']}")
    auto_tag = " (自动识别)" if result.get("auto_detected") else ""
    print(f"  【反应类型】{result['reaction_type']}{auto_tag}")

    # 底物分析
    sub = result["substrate_analysis"]
    if "error" not in sub:
        print(f"\n  ┌── 底物分析 ──")
        print(f"  │ 分子式: {sub.get('formula', '?')}  MW: {sub.get('molecular_weight', '?')}")
        sites = sub.get("reactive_sites", [])
        for s in sites[:5]:
            print(f"  │ 反应位点: {s['site']} — {s['type']} ({s['reactivity']})")

    # BDE
    bde = result["bde_estimate"]
    print(f"\n  ┌── BDE 估算 ──")
    print(f"  │ 最弱键 BDE ≈ {bde.get('estimated_bde_kcal', '?')} kcal/mol — {bde.get('label', '')}")

    # SMARTS
    sm = result["smarts_matches"]
    print(f"\n  ┌── SMARTS 匹配 ── {len(sm)} 条规则")
    for m in sm[:5]:
        print(f"  │ {m['name']}: {m['description']}")

    # 推荐条件
    print(f"\n  ┌── 推荐条件 ──")
    print(f"  │ 引发剂: {', '.join(result['recommended_initiators']) if result['recommended_initiators'] else 'AIBN (默认)'}")
    print(f"  │ 催化剂: {', '.join(result['recommended_catalysts']) if result['recommended_catalysts'] else '无特定推荐'}")
    cond = result["suggested_conditions"]
    if cond:
        print(f"  │ 温度: {cond.get('temperature', '?')}")
        print(f"  │ 溶剂: {cond.get('solvent', '?')}")

    # 实验操作
    lab = result["lab_guide"]
    if "error" not in lab:
        print(f"\n  ┌── 萃取 ──")
        ex = lab["extraction"]
        print(f"  │ 溶剂: {ex.get('primary', '?')} (logP={ex.get('logP', '?')})")
        print(f"  │ {ex.get('suggestion', '')}")
        print(f"\n  ┌── 纯化 ──")
        print(f"  │ 方法: {lab['purification']['method']}")
        for s in lab["purification"].get("steps", []):
            print(f"  │ {s}")
        print(f"\n  ┌── 干燥 ──")
        print(f"  │ {lab['drying']}")
        print(f"\n  ┌── 表征 ──")
        print(f"  │ {lab['characterization']['nmr_1H']}")
        print(f"  │ {lab['characterization']['ms']}")

    # 安全
    sw = lab.get("safety_warnings", [])
    if sw:
        print(f"\n  ┌── [!! 安全] ──")
        for w in sw:
            print(f"  │ {w['warning']}")

    # 结论
    print(f"\n  ┌── 综合判断 ──")
    print(f"  │ {result['verdict']}")
    print(f"\n{'='*65}")
    print("  以上为建议性指南，请根据实际实验情况调整。")
    print("=" * 65)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        ("C1=CC=C(C=C1)C#N", "ATRP", "苄腈"),
        ("CC(=O)Oc1ccccc1C(=O)O", "HAT", "阿司匹林"),
        ("Cc1ccccc1", "radical_cyclization", "甲苯（做环化底物）"),
        ("CC(C)(C)N1[O]C(C)(C)CCCC1(C)C", "SET", "TEMPO 类似物"),
    ]

    for smiles, rtype, name in test_cases:
        print(f"\n{'='*60}")
        print(f"  {name} ({rtype})")
        guide = lab_guide(smiles, reaction_type=rtype)
        print_guide(guide)
        print()

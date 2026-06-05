# Radical Synthesis Semi-Automated Research Workflow

## Project overview

This is a semi-automated research workflow for a graduate student (研一) in radical synthesis (自由基合成). The system assists with: suggesting reaction routes, predicting spectra (NMR/MS/ESR), processing experimental data, and generating publication-quality editable figures.

## Research workflow (完整流程)

### 阶段一：反应预测（输入 SMILES → 告诉我值不值得做）

1. **底物分析** `analyze_substrate(smiles)` — RDKit 解析分子，识别反应位点：C-X 键、C=C、苄位/烯丙位/醛基 C-H，输出反应活性评分
2. **SMARTS 规则匹配** — 14 条内置规则，自动识别原子转移/HAT/SET/自由基加成/环化/HAS 等反应类型
3. **数据库相似底物查询** `query_similar_substrates(smiles)` — Morgan 指纹 + Tanimoto 相似度 + 子结构匹配，找库中最接近的已知底物
4. **BDE 估算** `_estimate_bde(smiles)` — 经验规则估算最弱 C-H 键键解离能
5. **合成可行性决策** `_assess_synthesizability()` — 综合打分（数据库匹配 +30，弱键 BDE<90 +20，SMARTS 匹配 +20，高收率 +20），输出 worth_synthesizing + confidence
6. **一键预测** `predict_conditions(smiles)` — 以上全部整合，输出推荐引发剂/催化剂/溶剂/温度 + 是否值得合成

### 阶段二：DFT 模拟（双引擎 ORCA + Gaussian）

- **ORCA 6.1.1**（学术免费，推荐）：`orca_interface.py` — `build_epr_input()`, `build_nmr_input()`, `build_opt_input()`, `parse_epr_output()`, `parse_nmr_output()`
- **Gaussian 09/16**：`gaussian_interface.py` — `build_nmr_input()`, `build_epr_input()`, `build_ts_input()`, `parse_nmr_output()`, `parse_epr_output()`
- 预测内容：NMR 化学位移（GIAO）、EPR g 张量 + 超精细耦合、几何优化、过渡态搜索
- 支持泛函：B3LYP/PBE0/M06-2X/wB97X-D3/r2SCAN 等；支持溶剂模型（CPCM/SCRF）

### 阶段三：实验产物表征

- **NMR** — 预测 vs 实验对比；`ExperimentalNMRPipeline` 全自动处理（相位校正→基线校正→峰解卷积→积分归一化→多重峰标注→结构指派）
- **MS** — 同位素分布模拟 + 碎片预测 + 实验对比叠加图
- **ESR/EPR** — g 值预测（21 种自由基类型）+ 超精细耦合模式模拟 + 自旋捕获分析（DMPO/PBN/DEPMPO 加合物识别）
- **ZFS 拟合** — EasySpin + Octave 拟合零场分裂参数 D/E/g（`fit_zfs_from_csv()`），支持 pepper/garlic/chili 三种 EPR 模式
- **UV-Vis** — 待开发

### 阶段四：出图 → 直接投文章

- **ChemFigure 上下文管理器** — 自动保存 SVG + PDF + PNG + pickle + CSV 五种格式
- **期刊样式**：jacs（默认）/ angewandte / nature_chem / acs / rsc
- **高级绘图**：分子结构插图、多重峰模拟（J 耦合）、积分曲线、多面板图
- **自学习**：每次出图记录到 `data/plot_experience.jsonl`，系统持续优化绘图参数

### 阶段五：数据库自进化

- **双端分离**：`add_my_experiment()` 自己的实验 + `add_literature()` 文献数据
- **持续积累**：每做一次实验、每读一篇文献都录入，数据库越大预测越准
- **反馈闭环**：实验结果反哺预测引擎，score 越来越可信

## Key modules

| 阶段 | Module | 核心 API |
|------|--------|---------|
| 🔮 预测 | `scripts/reaction_predictor.py` | `analyze_substrate(smiles)`, `suggest_reaction_routes(smiles)`, `predict_conditions(smiles)` |
| 💾 数据库 | `scripts/database.py` | `add_my_experiment(smiles, ...)`, `add_literature(smiles, ...)`, `query_similar_substrates(smiles)`, `get_smarts_matches(smiles)` |
| 🖥️ DFT | `scripts/orca_interface.py` | `build_epr_input(smiles)`, `build_nmr_input(smiles)`, `build_opt_input(smiles)`, `parse_epr_output()`, `run_orca()` |
| 🖥️ DFT | `scripts/gaussian_interface.py` | `build_nmr_input(smiles)`, `build_epr_input(smiles)`, `build_ts_input(smiles)`, `parse_nmr_output()`, `predict_nmr_dft(smiles)` |
| 🧲 NMR | `scripts/nmr_processor.py` | `predict_nmr_from_smiles(smiles)`, `predict_and_compare_nmr(smiles, exp_data)`, `read_bruker_nmr(path)` |
| 🧲 NMR | `scripts/nmr_experimental.py` | `ExperimentalNMRPipeline` — 自动相位/基线/解卷积/积分/多重峰/结构指派 |
| ⚡ MS | `scripts/ms_processor.py` | `predict_ms(smiles)`, `predict_isotopic_pattern(smiles)`, `plot_ms_prediction(smiles)`, `plot_ms_comparison(smiles, exp)` |
| 🌀 ESR | `scripts/esr_processor.py` | `predict_g_value(radical_type)`, `predict_hyperfine_pattern(nuclei)`, `predict_and_compare_esr(exp, ...)`, `analyze_spin_trap(exp, ...)` |
| 📐 ZFS | `scripts/zfs_fitter.py` | `fit_zfs_from_csv(csv, S=1, mw_freq=9.5)`, `simulate_zfs(S, D, E, g)`, `check_environment()` |
| 🎨 出图 | `scripts/plot_utils.py` | `ChemFigure(name, journal, width)` 上下文管理器, `save_figure(fig, name)`, `get_plot_history(n)` |
| 🎨 出图 | `scripts/journal_plot.py` | `draw_molecule_inset(ax, smiles)`, `simulate_multiplet(...)`, `nmr_multipanel(...)` |

## Plotting conventions (CRITICAL)

- **Always use `ChemFigure` context manager** for figures intended for publication
- **Always save as SVG + PDF** — the user edits figures in Illustrator/Inkscape
- **Always save raw data as CSV** alongside figures for Origin/GraphPad
- **Save pickle** for re-editing within Python
- Default journal style: JACS. Also available: `angewandte`, `nature_chem`, `acs`, `rsc`
- Journal style sheets are in `styles/` with consistent typography and sizing
- Single column: 3.3 in, Double: 7.0 in, aspect ratio ~golden ratio
- All plotting experiences are logged to `data/plot_experience.jsonl` for self-evolution

## Self-evolution (plotting memory)

After each plotting session, review `data/plot_experience.jsonl` and `scripts/plot_utils.py:get_plot_history()` to learn from past plots. The goal is to generate increasingly accurate, publication-ready figures that match top journal requirements. When the user gives feedback on a figure (e.g., "font too small", "wrong axis range"), update the plotting approach and document the lesson.

## Python environment

- Python 3.14 at `python3` (python3.exe)
- Default `python` command is Python 2.7 — always use `python3`
- pip is associated with Python 3
- Key packages: rdkit, nmrglue, matplotlib, seaborn, pyteomics, pandas, scipy, scikit-learn, openpyxl

## EasySpin + Octave EPR/ZFS fitting

The user has GNU Octave 10.2.0 + EasySpin 6.x running on this computer for EPR simulation and ZFS fitting:

- **Octave CLI**: `C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe`
- **EasySpin**: `C:\Users\xushaobo\easyspin\EasySpin-main\easyspin\`
- **EasySpin private**: `C:\Users\xushaobo\easyspin\EasySpin-main\easyspin\private\`
- **Octave startup**: `C:\Users\xushaobo\.octaverc` (adds EasySpin to path)
- **Compatibility patches applied**: datetime.m, verLessThan.m, split.m, pad.m in private/; source edits to pepper.m, garlic.m, saffron.m, easyspin_compile.m
- **8 MEX files compiled** for Octave: cubicsolve, lisum1i, projectzones, projecttriangles, multinucstick, multimatmult_, chili_lm, sf_peaks
- **ZFS fitting bridge**: `scripts/zfs_fitter.py` — `fit_zfs_from_csv('data.csv', S=1, mw_freq=9.5)` runs esfit and returns D, E, g parameters
- **Key constraint**: Octave's private directory MUST be explicitly added to path for .mex files to work
- **Desktop package**: `C:\Users\xushaobo\Desktop\EasySpin-Octave-ZFS\` and `EasySpin-Octave-ZFS-完整包.md` contain the full handoff package for colleagues

When the user asks about ZFS fitting / EPR simulation / EasySpin, use these paths. For fitting, generate an Octave script → run with octave-cli --no-gui → parse results.

## ORCA (quantum chemistry)

- **Version**: 6.1.1 RELEASE
- **Path**: `C:\ORCA_6.1.1\orca.exe`
- **Interface**: `scripts/orca_interface.py` — `build_epr_input(smiles)`, `build_nmr_input(smiles)`, `build_opt_input(smiles)`, `parse_epr_output()`, `parse_nmr_output()`, `run_orca()`, `check_installation()`
- Supports: DFT geometry optimization, NMR chemical shifts (GIAO), EPR g-tensor + hyperfine coupling
- ORCA is free for academic use, no license needed

## Current development focus

- ✅ Step 1 — 反应预测模块：SMARTS 规则 + 数据库 + BDE 估算 + 合成可行性打分
- ✅ Step 2 — 数据库双端分离（literature / my_experiment）+ 相似性查询
- ✅ Step 3 — ORCA 接口：NMR/EPR/opt 输入生成 + 输出解析
- ✅ Step 4 — 谱图预测：NMR/MS/ESR 全覆盖 + 实验对比
- ✅ Step 5 — ZFS 拟合：EasySpin + Octave 桥接
- 🔜 待开发：UV-Vis 预测模块
- 🔜 待完善：SMARTS 规则扩充（随实验积累持续添加）

## 快速命令参考

```bash
python3 demo.py                                          # 端到端验证，跑通所有模块
python3 -c "from scripts.database import init_database; init_database()"  # 初始化数据库
python3 -c "from scripts.reaction_predictor import predict_conditions; print(predict_conditions('SMILES'))"  # 一键预测
python3 -c "from scripts.zfs_fitter import check_environment; print(check_environment())"  # 检查 ZFS 环境
```

## 修改反应规则的入口

1. **SMARTS 分子识别规则** → `scripts/database.py` → `seed_smarts_rules()` 函数（第152行）
2. **引发剂/催化剂知识库** → `scripts/reaction_predictor.py` → `RADICAL_INITIATORS` / `CATALYSTS_MEDIATORS` 字典
3. **反应类型库** → `scripts/reaction_predictor.py` → `RADICAL_REACTION_TYPES` 字典
4. **合成可行性评分逻辑** → `scripts/reaction_predictor.py` → `_assess_synthesizability()` 函数
5. **BDE 估算规则** → `scripts/reaction_predictor.py` → `_estimate_bde()` 函数

## The user

- First-year graduate student, second semester (研一下学期), field: radical synthesis (自由基合成)
- Has Aspen Plus/HYSYS v12 for process simulation
- Uses Zotero for reference management
- Prefers figures editable in Illustrator
- Works with: organic radicals, spin trapping (DMPO/PBN/DEPMPO), photoredox catalysis, ATRP
- Communicates via both VSCode IDE (this session) AND WeChat via cc-connect (separate sessions — the WeChat agent needs this CLAUDE.md to understand the local environment)

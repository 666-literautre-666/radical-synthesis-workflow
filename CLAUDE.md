# Radical Synthesis Semi-Automated Research Workflow

研一自由基合成半自动化研究流程。核心能力：反应预测 → DFT模拟 → 谱图表征 → 出图 → 数据库 + GNN持续进化。

## Python environment

- Python 3.14 → `python3`（默认 `python` 是 2.7，永远用 `python3`）
- 关键包：rdkit, nmrglue, matplotlib, seaborn, pyteomics, pandas, scipy, scikit-learn, openpyxl, torch, torch_geometric

## Critical paths

- **ORCA 6.1.1**: `C:\ORCA_6.1.1\orca.exe`，接口 `scripts/orca_interface.py`
- **Octave CLI**: `C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe`
- **EasySpin**: `C:\Users\xushaobo\easyspin\EasySpin-main\easyspin\`，启动脚本 `C:\Users\xushaobo\.octaverc`
- **EasySpin private**（MEX文件所在）: `C:\Users\xushaobo\easyspin\EasySpin-main\easyspin\private\`
- **ZFS拟合桥接**: `scripts/zfs_fitter.py` → 生成Octave脚本 → octave-cli --no-gui → 解析结果

## Key modules quick ref

| Module | 核心 API |
|--------|---------|
| `scripts/reaction_predictor.py` | `analyze_substrate()`, `predict_conditions()`, `_estimate_bde()` |
| `scripts/database.py` | `add_my_experiment()`, `add_literature()`, `query_similar_substrates()`, `seed_smarts_rules()` |
| `scripts/orca_interface.py` | `build_epr_input()`, `build_nmr_input()`, `build_opt_input()`, `parse_epr_output()` |
| `scripts/nmr_processor.py` | `predict_nmr_from_smiles()`, `read_bruker_nmr()` |
| `scripts/nmr_experimental.py` | `ExperimentalNMRPipeline` |
| `scripts/ms_processor.py` | `predict_ms()`, `plot_ms_comparison()` |
| `scripts/esr_processor.py` | `predict_g_value()`, `predict_hyperfine_pattern()`, `analyze_spin_trap()` |
| `scripts/zfs_fitter.py` | `fit_zfs_from_csv()`, `simulate_zfs()`, `check_environment()` |
| `scripts/plot_utils.py` | `ChemFigure` 上下文管理器, `save_figure()` |
| `bde_agent/gnn_train.py` | BDEGNNv3 — GINEConv双通道BDE预测 |
| `bde_agent/eval_model.py` | 模型评估 |

## Current development focus

- ✅ SMARTS规则 + 数据库 + BDE估算 + 合成可行性
- ✅ ORCA/Gaussian接口 + NMR/MS/ESR预测
- ✅ ZFS拟合 (EasySpin+Octave)
- ✅ GNN v2 (GINEConv×4, MAE 1.46, R² 0.9836)
- 🔄 GNN v2 接入 `_estimate_bde`（规则+ML双输出）
- 🔜 OrbitAll 自旋密度预训练 + 自由基BDE精调（暑假）
- 🔜 Δ-learning（模型只学"规则差多少"）
- 🔜 UV-Vis 预测模块

## 阶段六创新方向

1. 自由基物理特征注入GNN（自旋密度/电荷/多重度 → 节点特征）
2. 指纹+GNN双通道缝合（Morgan指纹MLP + 分子图GNN → 拼接预测BDE）
3. 差分ML（Δ-learning: `bde = rule_bde + delta`）

## 代码修改入口

1. SMARTS规则 → `scripts/database.py` → `seed_smarts_rules()`
2. 引发剂/催化剂 → `scripts/reaction_predictor.py` → `RADICAL_INITIATORS` / `CATALYSTS_MEDIATORS`
3. BDE估算 → `scripts/reaction_predictor.py` → `_estimate_bde()`
4. 合成可行性 → `scripts/reaction_predictor.py` → `_assess_synthesizability()`

## Commands

```bash
python3 demo.py                          # 端到端验证
python3 -c "from scripts.database import init_database; init_database()"
python3 -c "from scripts.reaction_predictor import predict_conditions; print(predict_conditions('SMILES'))"
```

## Self-evolution

绘图经验记录在 `data/plot_experience.jsonl`，每次出图后自动学习优化。所有图必须 SVG+PDF+CSV+pickle+PNG 五格式输出。

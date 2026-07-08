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
- ✅ GNN v5 (双通道 + 多任务, **MAE 1.03**, R² 0.9862)
- ✅ SpinPretrainNN v1 预训练 (QM9star 712k自由基, MAE 0.02, r=0.979)
- ✅ BDE Agent 三层引擎: 数据库查表(65k) → 规则 → GNN 全键预测
- 🔄 v5-Δ Δ-learning 训练中 (predict BDE - rule_BDE)
- 🔜 前插网络升级: 加键级 + S-T能隙 + 扩大预训练分子数
- 🔜 Δ-learning 验证通过后, 引入更多物理特征 (键级/S-T能隙/轨道能)
- 🔜 UV-Vis 预测模块

## v5 模型架构 (2026-07-07)

```
SMILES → SpinPretrainNN(冻结, 602k params) → 256维自旋嵌入
           ├── Channel A: 压缩 256→64 → 拼入节点特征 (10+64=74)
           ├── Channel B: 残差 256→hidden → 门控注入GNN末层
           └── 多任务: BDE(主) + spin(辅) + charge(辅)
```

| 版本 | 设计 | MAE | R² |
|------|------|-----|------|
| v2 | 纯GNN, 10维输入 | 1.46 | 0.9836 |
| v4 | 10+256拼接(闭壳层) | 4.18 | — |
| v5 | 双通道+多任务 | **1.03** | **0.9862** |
| v5-Δ | v5 + Δ-learning | 训练中 | — |

## 代码修改入口

1. SMARTS规则 → `scripts/database.py` → `seed_smarts_rules()`
2. 引发剂/催化剂 → `scripts/reaction_predictor.py` → `RADICAL_INITIATORS` / `CATALYSTS_MEDIATORS`
3. BDE估算 → `scripts/reaction_predictor.py` → `_estimate_bde()`
4. BDE Agent 推理 → `bde_agent/gnn_inference.py` → `analyze_bde()` / `predict_all_bonds()`
5. BDE GNN 训练 → `bde_agent/train_v5.py` / `train_v5_delta.py`
6. 前插网络预训练 → `bde_agent/train_spin.py`
7. 规则 BDE 基线 → `bde_agent/rule_bde.py` (24 键型)

## Commands

```bash
python3 demo.py                          # 端到端验证
python3 -c "from scripts.database import init_database; init_database()"
python3 -c "from scripts.reaction_predictor import predict_conditions; print(predict_conditions('SMILES'))"
```

## Self-evolution

绘图经验记录在 `data/plot_experience.jsonl`，每次出图后自动学习优化。所有图必须 SVG+PDF+CSV+pickle+PNG 五格式输出。

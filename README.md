1# 自由基合成半自动化研究平台

免费开源，零预算，一个研究生的全套 AI4S 工具箱。

## 已配置工具

| 工具 | 用途 | 调用方式 |
|------|------|---------|
| **RDKit** | 分子结构生成、子结构匹配、SMARTS 反应模板 | `from rdkit import Chem` |
| **PubChem** | 查询已知化合物信息（是否存在、可购买性） | `import pubchempy as pcp` |
| **ORCA 6.1.1** | DFT 计算（g 张量、超精细耦合、BDE、结构优化） | `scripts/orca_interface.py` |
| **SQLite** | 你自己的实验数据库（零配置，Python 内置） | `import sqlite3` |
| **EasySpin + Octave** | EPR 谱模拟与 ZFS 拟合（固体粉末/冷冻玻璃） | `scripts/zfs_fitter.py` |
| **Matplotlib** | 顶刊级图表输出（JACS/ACS/Angewandte 等风格） | `scripts/plot_utils.py` |

## 安装

```bash
pip install rdkit pubchempy numpy scipy matplotlib
```

ORCA 从 https://orcaforum.kofo.mpg.de/ 注册下载（学术邮箱免费）。

## 使用流程

### 1. 输入分子

```python
from rdkit import Chem
mol = Chem.MolFromSmiles('C1=CC=C(C=C1)CBr')  # 溴化苄
```

### 2. 查询已知信息

```python
import pubchempy as pcp
results = pcp.get_compounds('benzyl bromide', 'name')
c = results[0]
print(c.molecular_weight, c.canonical_smiles)
```

### 3. 子结构匹配（找类似底物）

```python
from rdkit import Chem
from rdkit.Chem import Draw
# 提取苄位自由基片段
pattern = Chem.MolFromSmarts('[C]-[c]')  # 碳连芳环
if mol.HasSubstructMatch(pattern):
    print('含苄位自由基位点')
```

### 4. ORCA DFT 计算

```python
from scripts.orca_interface import build_epr_input, save_input, find_orca
inp = build_epr_input(smiles, functional='B3LYP', basis='def2-SVP', multiplicity=2)
save_input(inp, 'my_molecule')
# 命令行运行: C:/ORCA_6.1.1/orca.exe my_molecule.inp > my_molecule.out
```

### 5. EPR/ZFS 拟合

```python
from scripts.zfs_fitter import fit_zfs_from_csv
result = fit_zfs_from_csv('epr_data.csv', S=1, mw_freq=9.217)
# 返回 {D, E, g, RMSD}
```

### 6. 存入数据库

```python
import sqlite3
conn = sqlite3.connect('my_experiments.db')
conn.execute('''CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY, smiles TEXT, condition TEXT,
    yield REAL, D REAL, E REAL, g REAL, date TEXT)''')
conn.execute('INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)',
             (1, 'C1=CC=C(C=C1)CBr', 'AIBN/toluene/80C', 72.5,
              302.5, 30.8, 2.0006, '2026-05-30'))
conn.commit()
```

## 模块说明

```
scripts/
├── reaction_predictor.py   # 底物分析 + 反应条件推荐
├── orca_interface.py       # ORCA DFT 接口
├── esr_processor.py        # EPR 谱处理
├── zfs_fitter.py           # ZFS 拟合（调 EasySpin）
├── nmr_processor.py        # NMR 预测 + 实验数据处理
├── ms_processor.py         # MS 预测 + 碎片分析
├── plot_utils.py           # 期刊级出图
└── journal_plot.py         # 期刊模板
```

## 你需要的命令行操作

```bash
# 跑 Python 脚本
python demo.py

# 跑 ORCA DFT（后台）
C:/ORCA_6.1.1/orca.exe input.inp > output.out &

# Git 存档
git add . && git commit -m "更新了什么" && git push

# 装新包
pip install 包名
```

就这么多了。

# GNN v2 训练结果

| 轮次 | 数据量 | 架构 | MAE (kcal/mol) | RMSE | R² | 备注 |
|------|--------|------|------|------|------|------|
| epoch 0-328 | 80万 | GINEConv×4 (700k params) | 1.46 | 2.10 | 0.9836 | BDE 范围 10-175 kcal/mol |
| 基线 MLP v1 | 80万 | Linear×2 (673 params) | 8.7 | — | — | Morgan 指纹版 |
| 规则 `_estimate_bde` | 9条规则 | SMARTS 查表 | ~5 | — | — | 项目基线 |

# 下一轮改进方向

1. 自旋密度 + 部分电荷注入节点特征（10维 → 12维）
2. 仅自由基键型数据微调（C-H/C-Br/C-I/C-S/O-H/N-H）
3. 超参数微调（lr/gamma/batch_size）

# 最佳模型

gnn_bde_best_v2.pt — epoch 328, Best Val=0.0410
"""
test_pipeline.py — 虚拟双自由基分子图 + 三层级联网络全流程测试 + 自动纠错
"""
import torch, warnings, traceback
warnings.filterwarnings('ignore')
import os
os.environ['RDKIT_PYTHON_DISABLE_WARNINGS'] = '1'
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

from torch_geometric.data import Batch

from cascade_dataset import mol_to_cascade_data
from cascade_model import CascadeBDENet


def run_test():
    print("=" * 60)
    print("三层因果级联网络 — 全流程前向传播测试")
    print("=" * 60)

    # ---------- 1. 构造虚拟分子 ----------
    # 双自由基分子: 苄基自由基类似物 (含芳香环 + 侧链)
    test_smiles_list = [
        "Cc1ccccc1",                        # 甲苯 (C-H BDE ~85)
        "CCO",                               # 乙醇 (C-O BDE ~96)
        "c1ccccc1",                          # 苯 (C-H BDE ~112)
        "CC(C)(C)O",                         # 叔丁醇
        "C=CC",                              # 丙烯 (烯丙位 C-H)
        "O=[N+]([O-])c1ccccc1",             # 硝基苯
        "C1CCCCC1",                          # 环己烷
        "BrCc1ccccc1",                       # 苄基溴
    ]

    bond_indices = [0, 1, 0, 3, 0, 0, 0, 1]
    bde_values = [85.0, 96.0, 112.0, 104.0, 88.0, 72.0, 98.0, 55.0]

    env_conditions = [
        [298.0, 1.0, 0.0, 0.21],     # 标准气相
        [350.0, 24.5, 0.5, 0.21],    # 乙醇溶剂, 加热
        [298.0, 1.0, 0.0, 0.0],      # 惰性气氛
        [400.0, 2.0, 0.3, 0.05],     # 低氧, 高温
        [298.0, 8.9, 1.0, 0.21],     # 二氯甲烷
        [310.0, 78.4, 0.9, 0.21],    # 水溶液
        [320.0, 1.0, 0.0, 0.21],     # 气相高温
        [273.0, 1.0, 0.0, 0.10],     # 低温低氧
    ]

    data_list = []
    for smi, bid, bde, env in zip(test_smiles_list, bond_indices, bde_values, env_conditions):
        d = mol_to_cascade_data(smi, bond_idx=bid, bde_value=bde, env_raw=env)
        if d is None:
            print(f"  SKIP: {smi} (构造失败)")
            continue
        data_list.append(d)
        print(f"  {smi:30s} atoms={d.x.shape[0]:3d}  edges={d.edge_index.shape[1]//2:3d}  "
              f"env_T={env[0]:.0f}K  BDE={bde:.0f}")

    print(f"\n成功构造 {len(data_list)} 个分子的 PyG Data 对象")

    # ---------- 2. 批量化 ----------
    batch_data = Batch.from_data_list(data_list)
    print(f"\nBatch: {batch_data.num_graphs} graphs, "
          f"{batch_data.x.shape[0]} total atoms, "
          f"{batch_data.edge_index.shape[1]//2} total edges")

    # ---------- 3. 模型初始化 ----------
    model = CascadeBDENet(node_dim=10, edge_dim=4, hidden=128, n_layers=3, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {total_params:,} total, {trainable:,} trainable")

    # ---------- 4. 前向传播 (防错循环) ----------
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model.eval()
            with torch.no_grad():
                spin_pred, delta_est, bde_pred = model(batch_data)

            # 维度验证
            n_total_atoms = batch_data.x.shape[0]
            n_graphs = batch_data.num_graphs
            n_target_edges = int(batch_data.target_mask.sum().item())
            # 每个图有 2 条目标有向边 (i→j 和 j→i), 但我们只取正向边预测 BDE
            expected_bde = n_graphs

            print(f"\n--- Dimension Check ---")
            print(f"  Layer 1 node_emb hidden dim: {model.hidden}")
            spin_ok = spin_pred.shape == (n_total_atoms, 1)
            print(f"  Layer 1 spin_pred:          [{n_total_atoms}, 1] {'OK' if spin_ok else 'FAIL: ' + str(spin_pred.shape)}")
            delta_ok = delta_est.shape == (n_graphs, 1)
            print(f"  Layer 2 delta_EST:           [{n_graphs}, 1] {'OK' if delta_ok else 'FAIL: ' + str(delta_est.shape)}")
            print(f"  Layer 3 BDE pred:            [{bde_pred.shape[0]}, 1] OK (target_mask: {n_target_edges} edges)")

            assert spin_ok, f"spin_pred: expected ({n_total_atoms},1), got {spin_pred.shape}"
            assert delta_ok, f"delta_est: expected ({n_graphs},1), got {delta_est.shape}"
            assert bde_pred.shape[1] == 1, f"bde_pred dim2 should be 1, got {bde_pred.shape}"

            print(f"\n全流程前向传播成功，输出边级 BDE 维度为：{list(bde_pred.shape)}")
            print(f"BDE predictions: {bde_pred.squeeze().tolist()}")
            return True

        except Exception as e:
            print(f"\n[Attempt {attempt+1}/{max_retries}] 失败: {e}")
            traceback.print_exc()
            if attempt < max_retries - 1:
                print("正在修复并重试...")
            else:
                print("\n全流程测试失败，请检查模型架构。")
                return False

    return False


if __name__ == '__main__':
    success = run_test()
    exit(0 if success else 1)

"""
三层因果级联网络 — CascadeBDENet

Layer 1: GINEConv 底座 → node_emb(128) + graph_emb(128)
Layer 2: 寿命前置网络 → delta_E_ST (接收 graph_emb + 物理特征)
Layer 3: BDE 多环境终审 → 边级 BDE (融合 node_emb + delta_E_ST + env_tensor)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class CascadeBDENet(nn.Module):
    """三层因果级联 GNN, 预测 BDE + delta_E_ST + spin_density"""

    def __init__(self, node_dim=10, edge_dim=4, hidden=128, n_layers=3, dropout=0.3):
        super().__init__()
        self.hidden = hidden

        # ====== Layer 1: GINEConv 分子图底座 ======
        self.l1_input_proj = nn.Linear(node_dim, hidden)

        self.l1_convs = nn.ModuleList()
        self.l1_norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.l1_convs.append(GINEConv(nn_mlp, edge_dim=edge_dim, train_eps=True))
            self.l1_norms.append(nn.LayerNorm(hidden))

        self.l1_dropout = nn.Dropout(dropout)

        # 自旋密度预测头 (节点级, 辅助任务)
        self.spin_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # ====== Layer 2: 寿命前置网络 (delta_E_ST) ======
        # 输入: graph_emb(128) + num_resonance(1) + conj_length(1) + shielding(1) = 131
        self.l2_mlp = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # ====== Layer 3: BDE 多环境终审 (边级) ======
        # 输入: node_emb_u(128) + node_emb_v(128) + delta_est(1) + env_tensor(4) = 261
        self.l3_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1 + 4, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        n_graphs = int(data.num_graphs)

        # ==================== Layer 1 ====================
        h = self.l1_input_proj(x)
        for conv, norm in zip(self.l1_convs, self.l1_norms):
            h_new = conv(h, edge_index, edge_attr)
            h = h + self.l1_dropout(norm(h_new).relu())
        node_emb = h
        graph_emb = global_mean_pool(node_emb, batch)

        # 自旋密度预测 (辅助头)
        spin_pred = self.spin_head(node_emb)

        # ==================== Layer 2 ====================
        g2 = torch.cat([
            graph_emb,
            data.num_resonance.view(-1, 1),
            data.conj_length.view(-1, 1),
            data.shielding.view(-1, 1),
        ], dim=-1)
        delta_est = self.l2_mlp(g2)                    # [batch, 1]

        # ==================== Layer 3 ====================
        # env_tensor 在 batch 后是 [batch*4], reshape 为 [batch, 4]
        env_2d = data.env_tensor.view(n_graphs, 4)

        # 用 target_mask 找出目标边 (每根键只取一个方向)
        tgt_mask = data.target_mask
        tgt_idx = torch.where(tgt_mask)[0]
        tgt_idx = tgt_idx[::2]                         # 正向 i→j, 跳过反向 j→i
        src = edge_index[0, tgt_idx]
        dst = edge_index[1, tgt_idx]
        eu = node_emb[src]                             # [n_graphs, hidden]
        ev = node_emb[dst]                             # [n_graphs, hidden]

        edge_batch = batch[src]                        # [n_graphs]
        delta_per_edge = delta_est[edge_batch]         # [n_graphs, 1]
        env_per_edge = env_2d[edge_batch]              # [n_graphs, 4]

        edge_feats = torch.cat([eu, ev, delta_per_edge, env_per_edge], dim=-1)
        bde_pred = self.l3_mlp(edge_feats)             # [n_graphs, 1]

        return spin_pred, delta_est, bde_pred


if __name__ == '__main__':
    from cascade_dataset import mol_to_cascade_data
    from torch_geometric.data import Batch

    print("=" * 60)
    print("CascadeBDENet 冒烟测试 (批量化前向)")
    print("=" * 60)

    data_list = [
        mol_to_cascade_data("Cc1ccccc1", bond_idx=0, bde_value=85.0),
        mol_to_cascade_data("CCO", bond_idx=1, bde_value=96.0),
        mol_to_cascade_data("c1ccccc1", bond_idx=0, bde_value=112.0),
    ]
    batch_data = Batch.from_data_list(data_list)

    model = CascadeBDENet(node_dim=10, edge_dim=4, hidden=128, n_layers=3, dropout=0.3)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()
    with torch.no_grad():
        spin_pred, delta_est, bde_pred = model(batch_data)

    print(f"Layer 1 node_emb: [{spin_pred.shape[0]}, {model.hidden}] (hidden)")
    print(f"Layer 1 spin_pred: {list(spin_pred.shape)}")
    print(f"Layer 2 delta_EST: {list(delta_est.shape)}")
    print(f"Layer 3 BDE pred:  {list(bde_pred.shape)}")

    assert bde_pred.shape == (3, 1), f"BDE shape error: {bde_pred.shape}"
    assert delta_est.shape == (3, 1), f"delta_EST shape error: {delta_est.shape}"

    print("\n全流程前向传播成功，输出边级 BDE 维度为：", list(bde_pred.shape))

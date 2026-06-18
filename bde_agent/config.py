CONFIG = {
    # 数据
    'data_path': "C:/Users/xushaobo/radical-synthesis-workflow/data/bde_rdf_with_multi_halo_model_2.csv.gz",
    'nrows': 800000,

    # 分子指纹 (MLP baseline, 保留兼容)
    'nBits': 512,
    'radius': 2,

    # GNN 模型结构 (GPU: 大模型 + 全量数据)
    'gnn_hidden': 256,
    'gnn_layers': 4,
    'dropout': 0.3,
    'hidden_layers': [512, 256, 128, 64],

    # 训练
    'lr': 0.0015,
    'weight_decay': 0.0001,
    'epochs': 1000,
    'batch_size': 2048,
    'patience': 200,
}
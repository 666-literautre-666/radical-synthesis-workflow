CONFIG={
    #数据
    'data_path':"data/bde_rdf_with_multi_halo_model_2.csv.gz",
    'nBits':512,#分子指纹范围
    'radius':2,#分子指纹半径
    #模型结构
    'hidden_layers':[512,256,128,64],#模型层超参
    'dropout':0.3,#防止过拟合
    #训练
    'lr':0.001,#学习率
    'weight_decay':0.0005,#正则化
    'epochs':3000,#训练轮数
    'batch_size':4096#分批数

    


}
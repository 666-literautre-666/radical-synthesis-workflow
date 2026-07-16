"""
链式训练: spin v2 结束 → 冻结导出 → v5.1 自动开训
"""
import subprocess, time, os

os.chdir(os.path.dirname(__file__))
PYTHON = r'C:\Users\xushaobo\radical-synthesis-workflow\gpu_env\Scripts\python.exe'

# Step 1: 等 spin 训练结束 (轮询 progress 文件)
progress_file = r'C:\Users\xushaobo\Desktop\spin_v2_progress.txt'
print("Waiting for spin v2 training to finish...")

last_epoch = 0
stall_count = 0
while True:
    try:
        with open(progress_file, 'r') as f:
            lines = f.readlines()
        if lines:
            last_line = lines[-1].strip()
            # 解析 epoch 号: "E 450  tr=..."
            parts = last_line.split()
            if parts[0].startswith('E'):
                current_epoch = int(parts[1])
                if current_epoch > last_epoch:
                    print(f"  Epoch {current_epoch}...")
                    last_epoch = current_epoch
                    stall_count = 0
                else:
                    stall_count += 1
    except:
        pass

    # 到了 499 以上说明训练结束
    if last_epoch >= 499:
        print("Spin v2 training completed!")
        break

    # 如果进度文件 10 分钟没更新，可能训练已结束
    if stall_count > 10:
        print("Progress file stalled, checking process...")
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True, shell=True
        )
        if 'train_spin' not in result.stdout:
            print("Spin process gone, assuming finished.")
            break
        stall_count = 0

    time.sleep(60)

# Step 2: 冻结导出 spin v2
print("\nFreezing spin v2 backbone...")
subprocess.run([PYTHON, '-c', '''
import torch
from spin_pretrain import SpinPretrainNN

model = SpinPretrainNN(node_dim=10, edge_dim=4, hidden=256, n_layers=4, dropout=0.0)
ckpt = torch.load("spin_pretrain_best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"])
print(f"Loaded: epoch={ckpt['epoch']}, val={ckpt['best_val']:.6f}")

freeze = {}
for k, v in model.state_dict().items():
    if not any(k.startswith(h) for h in ["spin_head", "charge_head", "gap_head", "bond_head", "graph_head"]):
        freeze[k] = v

out = "spin_pretrain_frozen_v2.pt"
torch.save({"backbone": freeze, "hidden": model.hidden, "epoch": ckpt["epoch"], "best_val": ckpt["best_val"]}, out)
print(f"Frozen -> {out}")
'''], check=True)

# Step 3: 开训 v5.1
print("\n" + "="*60)
print("Starting v5.1 training...")
print("="*60)
subprocess.run([PYTHON, '-u', 'train_v5.py'], check=True)

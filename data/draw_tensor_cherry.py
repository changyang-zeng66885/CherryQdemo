import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置可用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

# 加载 .pt 文件
file_path = '/home/zengchangyang/CherryQdemo/data/cherry_indices/llama2-7b-impact_all_parameters.pt'
data = torch.load(file_path)
# 将数据移动到第一个可用的 GPU（即设备 1）
device = torch.device("cuda:0")  # 由于设置了 CUDA_VISIBLE_DEVICES，设备 0 对应物理设备 1

# 检查加载的数据类型并移动张量到 GPU
if isinstance(data, dict):
    # 如果数据是字典，提取出值并移动到 GPU
    tensors = {key: value.to(device) for key, value in data.items() if isinstance(value, torch.Tensor)}
else:
    raise TypeError("Unsupported data type")

print(f"tensors:\n{tensors}")
# 将张量展平并转换为 NumPy 数组
flattened_values = tensors['model.embed_tokens'].cpu().to(torch.float32).numpy().flatten()

# 随机抽取 N 个变量
N = 1000
top_pct = 1  # 前1%
random_indices = np.random.choice(flattened_values.size, N, replace=False)
selected_values = flattened_values[random_indices]

# 找到最大的 top_pct% 的阈值
num_top_values = int(N * top_pct * 0.01)  # 前top_pct% 的个数
threshold = np.partition(selected_values, -num_top_values)[-num_top_values]

# 创建标记颜色的数组
colors = ['red' if value >= threshold else 'blue' for value in selected_values]

# 绘制图表
plt.figure(figsize=(4, 3))
plt.scatter(range(N), selected_values, c=colors, marker='o', s=3)
plt.title(f'Impact(num={N}, top {top_pct} %)')
plt.xlabel('Index')
plt.ylabel('Impact')
plt.tight_layout()
plt.savefig(f'/home/zengchangyang/CherryQdemo/data/cherries_top_{top_pct}_pct_a.png')
import torch
import subprocess

def get_free_gpus(threshold=10000):
    """获取使用显存少于阈值的所有可用 GPU ID."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], 
                                stdout=subprocess.PIPE, text=True)
        
        gpus = result.stdout.strip().split('\n')
        suitable_gpus = []
        
        for gpu in gpus:
            index, memory_used = gpu.split(',')
            memory_used = int(memory_used.strip().replace(' MiB', ''))
            if memory_used < threshold:  # 选择显存使用少于阈值的 GPU
                suitable_gpus.append(int(index))
        
        return suitable_gpus  # 返回所有合适的 GPU IDs

    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return []



free_gpu_ids = get_free_gpus()  # 获取所有空闲的 GPU IDs
print(f"free_gpu_ids: {free_gpu_ids}")

# input_text = "Commuters stuck in traffic on the Leesburg Pike"
# if free_gpu_ids:
#     # 使用 DataParallel 将模型并行化
#     model = nn.DataParallel(model, device_ids=free_gpu_ids)  
#     model = model.to(f'cuda:{free_gpu_ids[0]}')  # 将模型移动到第一个空闲 GPU

#     inputs = tokenizer.encode(input_text, return_tensors="pt")  # 不需要传递到特定 GPU
#     inputs = inputs.to(f'cuda:{free_gpu_ids[0]}')  # 将输入移动到第一个空闲 GPU
# else:
#     raise MemoryError("没有空闲的 GPU 可用。")
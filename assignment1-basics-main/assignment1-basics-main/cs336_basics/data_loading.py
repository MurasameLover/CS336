import numpy as np
import torch

def get_batch(
    x: np.ndarray,
    batch_size,
    context_length,
    device,
) -> tuple[torch.Tensor, torch.Tensor]: # 返回（inputs, targets）
    
    # 最大起始位置
    max_start = len(x) - context_length
    # 随机采样起始位置
    start_pos = np.random.randint(0, max_start, size=batch_size)
    # 提取inputs 
    inputs = np.stack([
        x[i: i+context_length]
        for i in start_pos
    ])
    # 提取targets（比 inputs 右移一个位置）
    targets = np.stack([
        x[i+1: i+context_length+1]
        for i in start_pos
    ])
    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
    # inputs 和 targets 的形状都是 (batch_size, context_length)
    

import torch

def softmax(
    x: torch.Tensor,
    dim: int = -1,  # 在dim维度上进行softmax，默认最后一个维度
) -> torch.Tensor:
    """
    数值稳定性问题:
    直接计算torch.exp(v)可能会溢出,例如v = [1000, 1001, 1002],
    exp(1002)已经超出 float32 范围,整个 Softmax 崩溃.

    利用 Softmax 的一个重要性质:
    对于任意常数 c, softmax(v)=softmax(v−c)
    """
    
    # torch.max(x, dim=dim) 返回的是命名元组
    # result = x.max(dim=-1, keepdim=True)
    # result 是 (values=Tensor, indices=Tensor)
    # 包含了最大值和最大值的位置
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
    # 两种求和方法完全等价
    # x_exp.sum(dim=dim, keepdim=True)      # Tensor 的方法
    # torch.sum(x_exp, dim=dim, keepdim=True)  # torch 模块函数
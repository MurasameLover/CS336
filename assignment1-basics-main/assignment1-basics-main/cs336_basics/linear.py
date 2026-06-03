import torch.nn as nn
import math
import torch

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        """ 
        创建权重矩阵，形状 (out_features, in_features)

        参数形状 — 存 W 还是 Wᵀ?

        存W,形状为 (d_out, d_in)。
        数学上是 y = Wx(列向量)，但在 PyTorch(row-major)中,
        对 batch 输入 x 形状 (..., d_in),forward 计算是：
        # x: (..., d_in), weight: (d_out, d_in)
        output = x @ weight.T   # → (..., d_out)

        """
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        """ 进行权重初始化 """
        # 线性层权重：𝒩(𝜇=0, 𝜎²=2/(d_in + d_out))，截断到 [−3𝜎, 3𝜎]
        std = math.sqrt(2.0/(in_features + out_features))
        # 用 trunc_normal_ 初始化
        # trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        #                        均值      标准差  截断下限  截断上限
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        在forward中,
        数学上是 y = Wx
        但 PyTorch 是 row-major 布局，输入 x 的形状是 (..., d_in)，你的 weight 形状是 (d_out, d_in)。
        所以要把 weight 转置一下，让 x 的最后一个维度 d_in 和 weight.T 的第一个维度 d_in 对齐

        @ 是 PyTorch 的矩阵乘法运算符，等价于 torch.matmul(x, self.weight.T)

        """
        return x @ self.weight.T
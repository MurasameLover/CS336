from .linear import Linear
import torch
import torch.nn as nn

"""
FFN(x) = W₂(SiLU(W₁x) ⊙ W₃x)

W₁: (d_ff, d_model)
W₂: (d_model, d_ff)
W₃: (d_ff, d_model)
SiLU(x) = x · σ(x)    # σ = sigmoid
"""
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 这里可以去看一下Linear的实现，forward中是return x @ self.weight.T，
        # 所以linear中weight是(out_features, in_features)
        self.linear1 = Linear(d_ff, d_model, device, dtype)
        self.linear2 = Linear(d_model, d_ff, device, dtype)
        self.linear3 = Linear(d_ff, d_model, device, dtype)

    """ silu激活函数（替代relu的作用） """
    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.silu(self.linear1(x)) * self.linear3(x)
        )
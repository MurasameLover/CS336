import torch.nn as nn
import torch

class RoPE(nn.Module):
    """
    RoPE 并不是对整个向量一次旋转,而是每两个维度组成一个二维向量.然后分别旋转.
    [d0, d1, d2, d3, d4, d5]
    会被拆成
        (d0,d1)
        (d2,d3)
        (d4,d5)
    
    对位置 i 的第 k 对维度：
    θ_k = 1 / Θ^(2k/d_k)     （Θ 就是 theta）
    cos(θ_k · i)    sin(θ_k · i)
    旋转这对维度:
    x'[2k]   = x[2k] × cos(θ_k·i) - x[2k+1] × sin(θ_k·i)
    x'[2k+1] = x[2k] × sin(θ_k·i) + x[2k+1] × cos(θ_k·i)
    """

    """
    可以考虑预先计算cos和sin，然后直接用，但本仓库尝试使用欧拉公式简化代码：

    把 x 的一对维度 (x_2k, x_{2k+1}) 看作一个复数 z = x_2k + i·x_{2k+1}，RoPE 就是乘上 e^{iθ}：
    z' = z · e^{iθ} = (x_2k + i·x_{2k+1})(cosθ + i·sinθ)
    展开实部 = x_2k·cosθ - x_{2k+1}·sinθ     ← 这就是原来的公式
    展开虚部 = x_2k·sinθ + x_{2k+1}·cosθ     ← 完全一致
    """

    """
    # 把最后一维 (real, imag) 看作复数
    z = torch.view_as_complex(x.reshape(...))  # → complex 类型张量

    # 从 (幅值, 角度) 创建复数 —— e^{iθ} = cosθ + i·sinθ
    rotation = torch.polar(torch.ones_like(angles), angles)  # |r|=1, 角度=θ

    # 复数相乘
    z_rotated = z * rotation

    # 转回实数
    result = torch.view_as_real(z_rotated).reshape(...)
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        # 先计算每组pair旋转的频率θ_k
        _2k = torch.arange(0, d_k, 2, device=device)
        freqs = 1.0 / theta ** (_2k / d_k)
        # 计算所有位置的旋转因子 e^{iθj·pos}
        pos = torch.arange(max_seq_len, device=device)     # (max_seq_len,)
        angles = pos[:, None] * freqs[None, :]             # (max_seq_len, d_k/2)
        # e^{iθ} = cosθ + i·sinθ
        rotations = torch.polar(torch.ones_like(angles), angles)  # complex: (max_seq_len, d_k/2)

        self.register_buffer("rotations", rotations, persistent=False)

    def forward(self, x, token_positions):
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)

        x_float = x.float()
        # 分组为复数: (..., seq_len, d_k/2) 每个元素是 complex
        x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2))

        # 取对应位置的旋转因子
        rot = self.rotations[token_positions]  # (..., seq_len, d_k/2)

        # 应用旋转后转回实数
        result = torch.view_as_real(x_complex * rot).reshape_as(x)
        return result.to(x.dtype)
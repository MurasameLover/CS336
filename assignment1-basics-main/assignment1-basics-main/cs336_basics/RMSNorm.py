import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    给定一个激活向量a,RMSNorm(𝑎ᵢ) = 𝑎ᵢ / RMS(𝒂) × 𝑔ᵢ
    其中𝑔ᵢ是可学习参数，共有d_model个这样的参数
    RMS(𝒂) = √( ¹/ₙ Σ 𝑎ᵢ² + ε )
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        x需要先变成float32,因为x ** 2可能会溢出 

        RMSNorm是在x的最后一个维度上进行的
        x.shape = [batch_size, seq_len, d_model]
        """
        in_dtype = x.dtype  #记录下输入数据的类型，方便后续恢复
        x = x.to(torch.float32)

        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 归一化
        x = x / rms * self.weight
        return x.to(in_dtype)
import torch
import torch.nn as nn
from .RMSNorm import RMSNorm
from .swiglu import SwiGLU
from .MultiHeadSelfAttention_with_RoPE import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.multiheadselfattention = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device)
        
        y = x + self.multiheadselfattention(self.norm1(x), token_positions)
        z = y + self.ffn(self.norm2(y))

        return z
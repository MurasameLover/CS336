import torch
import torch.nn as nn
from .attention import scaled_dot_product_attention
from .linear import Linear
from einops import rearrange
from .rope import RoPE

"""
    MultiHeadSelfAttention(x) = W_O · Concat(head₁, ..., head_h)

    整体流程:
        x: (batch, seq, d_model)
        ① Q/K/V 投影: x → Q, K, V (各用独立的 Linear, 形状不变)
        ② 拆多头: d_model → num_heads × d_k
        ③ 可选: RoPE 应用到 Q, K
        ④ 因果 mask + Scaled Dot-Product Attention
        ⑤ 合并多头: num_heads × d_k → d_model
        ⑥ 输出投影: W_O 让各头信息融合
        返回: (batch, seq, d_model)
"""

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = None,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        assert self.d_k * n_heads == d_model  #d_model must be divisible by n_heads
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # 保留一下基本信息，后续会使用
        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 拆头: (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        """
        这里不要用reshape进行拆头，而是用 einops 的 rearrange
        具体原因我解释不清，但是用reshape绝对报错，
        如果有人的解释清楚的话，欢迎提交PR
        """
        Q = rearrange(Q, "batch seq (head d_k) -> batch head seq d_k", head=self.n_heads)
        K = rearrange(K, "batch seq (head d_k) -> batch head seq d_k", head=self.n_heads)
        V = rearrange(V, "batch seq (head d_k) -> batch head seq d_k", head=self.n_heads)

        # 对Q，K应用 RoPE
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # 因果mask
        """
        语言模型训练时当前位置不能看到未来,第 i 个 token 只能关注j≤i的位置。
        mask:
            [
                [T,F,F,F],
                [T,T,F,F],
                [T,T,T,F],
                [T,T,T,T]
            ]
        即下三角矩阵
        """
        mask = torch.tril(
            torch.ones(seq_len, seq_len)
        ).to(x.device)

        # attention
        attention_output = scaled_dot_product_attention(Q, K, V, mask)

        # 合并头
        attention_output = rearrange(attention_output, "batch head seq d_k -> batch seq (head d_k)")

        # 映射
        """ 
        W_O 是 输出投影（output projection）矩阵。
        在 MHA 中，每个头独立计算 attention 后各自输出形状 (batch, head, seq, d_k)。
        合并头之后得到一个 (batch, seq, d_model)
        的张量。但这个张量是多个头输出的简单拼接，W_O
        的作用就是对这个拼接结果再做一次线性变换，让不同头的信息可以混合交互。
        """
        return self.W_O(attention_output)


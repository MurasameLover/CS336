import torch
import math
from .Softmax import softmax

""" scaled_dot_product_attention（缩放点积注意力） """
"""  Attention(Q, K, V) = softmax(QK^T / √d_k) V  """

def scaled_dot_product_attention(
    Q: torch.Tensor,    #(batch_size, ..., seq_len, d_k) 注意q,k,v的维度是不一样的
    K: torch.Tensor,    #(batch_size, ..., seq_len, d_k)
    V: torch.Tensor,    #(batch_size, ..., seq_len, d_v)
    mask: torch.Tensor | None = None,
):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    """
    Mask 掩码：
        有时我们希望某些位置不能互相看到，
        Mask 的形状：M ∈ {True,False} ^(n×m)
        True 表示允许关注，可以Attention。
    
    在得到scores之后，例如：
    scores =[1.2, 0.5, 2.0]
    mask = [True, True, False]
    则变成[1.2, 0.5, -inf]
    """
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention = softmax(scores, dim=-1)
    return attention @ V
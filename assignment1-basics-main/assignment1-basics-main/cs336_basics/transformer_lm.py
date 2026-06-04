from .transformer_block import TransformerBlock
from .embedding import Embedding
from .RMSNorm import RMSNorm
from .linear import Linear
import torch.nn as nn
import torch

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
            for _ in range(num_layers)
        ])
        self.embedding = Embedding(vocab_size, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor):
        x = self.embedding(token_ids)
        for layer in self.transformer_blocks:
            x = layer(x)
        x_norm = self.final_norm(x)
        logits = self.lm_head(x_norm)
        return logits
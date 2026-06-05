"""
GPT-2 style decoder-only transformer with RMSNorm, RoPE, SwiGLU, causal attention.

PyTorch implementation matching the JAX/Equinox model from the assignment.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ── RoPE ─────────────────────────────────────────────────────────────


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute cos/sin for RoPE, shape (seq_len, dim/2) each."""
    d = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    freqs = theta ** (-d)
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    x: torch.Tensor,  # (..., seq, head_dim)
    cos: torch.Tensor,  # (1, 1, seq, head_dim/2)
    sin: torch.Tensor,  # (1, 1, seq, head_dim/2)
) -> torch.Tensor:
    """Apply RoPE to the last dimension of x (assumed even)."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ── RMSNorm ──────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ── Attention ────────────────────────────────────────────────────────


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # 1 when no GQA

        assert self.hidden_size == self.n_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE cache will be set externally
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads."""
        b, s, n_kv, d = x.shape
        if self.n_rep == 1:
            return x
        return x[:, :, :, None, :].expand(b, s, n_kv, self.n_rep, d).reshape(b, s, self.n_heads, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim)

        # Apply per-head RMSNorm and RoPE on (b, n_heads, s, head_dim)
        q = q.transpose(1, 2)  # (b, n_heads, s, head_dim)
        k = k.transpose(1, 2)  # (b, n_kv_heads, s, head_dim)
        v = v.transpose(1, 2)

        # Per-head RMSNorm
        q = torch.stack([self.q_norm(q[:, i]) for i in range(self.n_heads)], dim=1)
        k = torch.stack([self.k_norm(k[:, i]) for i in range(self.n_kv_heads)], dim=1)

        # RoPE: apply along head_dim
        assert self.cos is not None and self.sin is not None
        cos = self.cos[:s].view(1, 1, s, -1)  # (1, 1, s, head_dim/2)
        sin = self.sin[:s].view(1, 1, s, -1)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Repeat KV heads for GQA (n_rep = 1 since n_heads == n_kv_heads)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (b, n_heads, s, s)
        causal_mask = torch.triu(torch.full((1, 1, s, s), float("-inf"), device=x.device), diagonal=1)
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = attn @ v  # (b, n_heads, s, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(b, s, self.hidden_size)
        out = self.o_proj(out)
        return out


# ── SwiGLU MLP ───────────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Decoder Layer ────────────────────────────────────────────────────


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = r + x

        r = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = r + x
        return x


# ── Causal LM ────────────────────────────────────────────────────────


class MiniCausalLM(nn.Module):
    """Full decoder-only transformer with LM head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Create layers, all sharing the RoPE cache via the first layer's attention
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE - will be moved to correct device on first forward
        cos, sin = precompute_freqs_cis(config.head_dim, config.seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq)
        labels: torch.Tensor | None = None,  # (batch, seq)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Returns:
            logits: (batch, seq, vocab_size)
            loss: scalar or None (if labels provided, the mean CE loss)
        """
        b, s = input_ids.shape
        assert s <= self.config.seq_len, f"seq_len {s} exceeds max {self.config.seq_len}"

        # Embed
        h = self.embed_tokens(input_ids)  # (b, s, d_model)

        # Share RoPE cache across all layers
        cos = self.rope_cos[:s].to(h.dtype)
        sin = self.rope_sin[:s].to(h.dtype)
        for layer in self.layers:
            layer.self_attn.cos = cos
            layer.self_attn.sin = sin

        # Apply layers
        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = h @ self.embed_tokens.weight.T  # tied embeddings

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    def num_non_embedding_params(self) -> int:
        """Return non-embedding parameter count."""
        return sum(p.numel() for name, p in self.named_parameters() if "embed" not in name and "lm_head" not in name)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __str__(self) -> str:
        return f"MiniCausalLM(params={self.num_params():,}, non_emb={self.num_non_embedding_params():,}, config={self.config})"

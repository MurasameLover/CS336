"""
Model configuration for miniature GPT-2 style decoder-only transformer.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""

    # Architecture
    vocab_size: int = 5000
    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 4  # = num_attention_heads (no GQA)
    head_dim: int = 32
    intermediate_size: int = 512  # = hidden_size * 4 typically
    attention_bias: bool = False

    # Normalization
    rms_norm_eps: float = 1e-6

    # RoPE
    rope_theta: float = 10000.0

    # Embedding
    tie_word_embeddings: bool = False

    # Training context
    seq_len: int = 256

    @property
    def jax_dtype(self):
        return None  # PyTorch — dtype handled separately

    @property
    def d_model(self):
        return self.hidden_size

    @property
    def n_layers(self):
        return self.num_hidden_layers

    @property
    def n_heads(self):
        return self.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.num_key_value_heads

    def num_non_embedding_params(self) -> int:
        """
        Compute non-embedding parameter count.
        Standard formula: 12 * n_layers * d_model^2
        (each transformer layer has: Q,K,V,O projections + gate/up/down MLP = 12 matrices)
        """
        return 12 * self.num_hidden_layers * self.hidden_size * self.hidden_size

    def num_embedding_params(self) -> int:
        """Embedding + LM head params."""
        head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        return self.vocab_size * self.hidden_size + head

    def num_params(self) -> int:
        """Total parameter estimate."""
        return self.num_non_embedding_params() + self.num_embedding_params()

    def total_flops_per_token(self, batch_size: int = 1) -> float:
        """Estimated FLOPs per token (forward pass)."""
        # Rough estimate: 6 * num_params per token (from scaling laws literature)
        # For forward + backward: 2 * 6 * num_params * num_tokens
        return 12.0 * self.num_params()

    def __str__(self) -> str:
        n_emb = self.num_non_embedding_params()
        total = self.num_params()
        return (
            f"ModelConfig(d_model={self.hidden_size}, n_layers={self.num_hidden_layers}, "
            f"n_heads={self.num_attention_heads}, d_head={self.head_dim}, "
            f"d_ff={self.intermediate_size}, "
            f"non_emb_params={n_emb:,}, total_params={total:,}, "
            f"vocab={self.vocab_size})"
        )


@dataclass
class IsoflopsRun:
    """A single training run configuration within an IsoFLOPs profile."""

    # Unique identifier
    run_id: str

    # Compute budget and model size
    compute_budget: float  # C in FLOPs
    model_config: ModelConfig  # N via config
    num_train_tokens: int  # D = C / (6*N)

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_frac: float = 0.05
    weight_decay: float = 0.01

    # Results (filled after training)
    final_val_loss: float | None = None
    training_time_seconds: float | None = None

    @property
    def n_params(self) -> int:
        return self.model_config.num_params()

    @property
    def num_optimizer_steps(self) -> int:
        tokens_per_step = self.model_config.seq_len * self.batch_size
        return self.num_train_tokens // tokens_per_step

    def __str__(self) -> str:
        return (
            f"IsoflopsRun(id={self.run_id}, C={self.compute_budget:.2e}, "
            f"N={self.n_params:,}, D={self.num_train_tokens:,}, "
            f"steps={self.num_optimizer_steps:,}, loss={'N/A' if self.final_val_loss is None else f'{self.final_val_loss:.4f}'})"
        )

    def __repr__(self) -> str:
        return self.__str__()

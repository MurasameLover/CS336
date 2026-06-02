# CS336 — Transformers

My work for Stanford's CS336 course on large language models and transformer architectures. Each folder contains a complete assignment with its own code, tests, and handout.

## Assignments

| # | Topic | Key Coverage |
|---|-------|-------------|
| 1 | **Basics** — BPE tokenizer, transformer blocks, training loop | BPE tokenization, attention, residual connections, AdamW, learning rate scheduling |
| 2 | **Systems** — Efficient kernels, distributed training, mixed precision | Triton kernels, flash attention, FSDP/DDP, `torch.compile`, bf16/fp8 |
| 3 | **Scaling** — Scaling laws, loss prediction, infrastructure | Compute-optimal training, data curation, large-scale launch scripts |
| 4 | **Data** — Data pipelines, deduplication, filtering | Quality filtering, deduplication (MinHash, exact), mixing strategies |
| 5 | **Alignment** — RLHF, DPO, preference optimization | Reward modeling, PPO, DPO, rejection sampling |

Each assignment was completed as **independent implementation work** — writing tokenizers, model blocks, optimizers, and training infrastructure from scratch following the course handouts.

Built with Python, PyTorch, and uv for environment management.

"""
Training loop for MiniCausalLM with warmup-cosine LR schedule.
"""

import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, IsoflopsRun

logger = logging.getLogger(__name__)


def get_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float = 3e-4,
    final_lr_frac: float = 0.1,
):
    """
    Warmup-cosine LR schedule.

    Linearly warms up from 0 to peak_lr over warmup_steps,
    then cosine decays to peak_lr * final_lr_frac.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (1 - final_lr_frac) * cosine + final_lr_frac

    return LambdaLR(optimizer, lr_lambda)


def train_on_data(
    model: nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    run: IsoflopsRun,
    device: torch.device,
    run_dir: Path,
    *,
    log_every: int = 50,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    """
    Train a model on exactly num_train_tokens, then evaluate on val.

    Returns final validation loss.
    """
    model = model.to(device)
    model.train()

    seq_len = run.model_config.seq_len
    batch_size = run.batch_size
    tokens_per_step = seq_len * batch_size
    total_steps = run.num_optimizer_steps

    assert total_steps > 0, f"num_optimizer_steps must be positive, got {total_steps}"

    # Prepare training data: flatten and slice required number of tokens
    train_flat = train_tokens.reshape(-1)
    n_train_needed = total_steps * tokens_per_step
    if len(train_flat) < n_train_needed:
        # Repeat if needed
        n_repeats = (n_train_needed // len(train_flat)) + 1
        train_flat = train_flat.repeat(n_repeats)
    train_flat = train_flat[:n_train_needed]
    train_data = train_flat.view(-1, seq_len)

    # Prepare validation data
    val_flat = val_tokens.reshape(-1)
    n_val_tokens = run.model_config.seq_len * batch_size * 4  # 4 val batches
    if len(val_flat) < n_val_tokens:
        n_val_tokens = (len(val_flat) // seq_len) * seq_len
    val_data = val_flat[:n_val_tokens].view(-1, seq_len)

    # DataLoaders
    train_ds = TensorDataset(train_data)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    val_ds = TensorDataset(val_data)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    warmup_steps = int(total_steps * run.warmup_frac)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps, run.learning_rate)

    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == torch.float16))

    step = 0
    epoch = 0
    losses = []

    print(f"  Training: {total_steps} steps, {run.num_train_tokens:,} tokens, "
          f"bs={batch_size}, lr={run.learning_rate:.2e}, warmup={warmup_steps}")
    train_start = time.perf_counter()

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            x = batch[0][:, :-1].to(device, non_blocking=True).long()
            y = batch[0][:, 1:].to(device, non_blocking=True).long()

            with torch.amp.autocast(device.type, dtype=dtype):
                _, loss = model(x, labels=y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())

            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"    step {step:>6d}/{total_steps}  loss={loss.item():.4f}  lr={lr_now:.2e}")

            step += 1

    train_time = time.perf_counter() - train_start
    print(f"  Training done in {train_time:.1f}s ({train_time/60:.1f}min)")

    # Evaluate on validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0][:, :-1].to(device, non_blocking=True).long()
            y = batch[0][:, 1:].to(device, non_blocking=True).long()
            with torch.amp.autocast(device.type, dtype=dtype):
                _, loss = model(x, labels=y)
            val_losses.append(loss.item())

    val_loss = sum(val_losses) / len(val_losses)
    print(f"  Validation loss: {val_loss:.4f}")

    # Save results
    results = {
        "run_id": run.run_id,
        "compute_budget": run.compute_budget,
        "n_params": run.n_params,
        "num_train_tokens": run.num_train_tokens,
        "num_optimizer_steps": total_steps,
        "final_val_loss": val_loss,
        "train_time_seconds": train_time,
        "config": {
            "hidden_size": run.model_config.hidden_size,
            "num_hidden_layers": run.model_config.num_hidden_layers,
            "num_attention_heads": run.model_config.num_attention_heads,
            "head_dim": run.model_config.head_dim,
            "intermediate_size": run.model_config.intermediate_size,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), run_dir / "model.pt")

    return val_loss

"""
Validation experiment: train at the predicted optimal configuration and compare
actual loss vs predicted loss.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig, IsoflopsRun
from data import prepare_data
from model import MiniCausalLM
from train import train_on_data

BASE_DIR = Path(__file__).resolve().parent
VALIDATE_DIR = BASE_DIR / "validation_run"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

VALIDATE_DIR.mkdir(parents=True, exist_ok=True)


def find_nearest_config(
    n_pred: float, d_pred: float, max_n_params: int = 60_000_000
) -> tuple[ModelConfig, int]:
    """
    Given predicted N_opt and D_opt, find the closest model configuration
    that fits in memory (max_n_params constraint) and compute D accordingly.
    """
    from run_isoflops import MODEL_PROTOTYPES

    # Find closest prototype by non-embedding params
    best_name = None
    best_diff = float("inf")

    for name, mc in MODEL_PROTOTYPES.items():
        n_emb = mc.num_non_embedding_params()
        diff = abs(n_emb - n_pred)
        if diff < best_diff and mc.num_params() <= max_n_params:
            best_diff = diff
            best_name = name

    if best_name is None:
        # Fallback: use the largest config
        best_name = "XL"

    mc = MODEL_PROTOTYPES[best_name]
    # Compute D = C / (6*N) for this model
    C_target = d_pred * 6 * n_pred  # Recover target FLOPs
    D_actual = max(200_000, int(C_target / (6 * mc.num_params())))

    print(f"  Selected config: {best_name} ({mc.num_params():,} params, "
          f"{mc.num_non_embedding_params():,} non-emb)")
    print(f"  D for validation: {D_actual:,} tokens")

    return mc, D_actual


def run_validation(
    device: torch.device,
    predictions: dict,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Run the validation experiment.

    Predictions dict should contain:
        predicted_N, predicted_D, predicted_loss, target_budget
    """
    print("=" * 60)
    print("Validation: Testing Extrapolation Accuracy")
    print("=" * 60)

    n_pred = predictions["predicted_N"]
    d_pred = predictions["predicted_D"]
    loss_pred = predictions["predicted_loss"]
    C_target = predictions.get("target_budget", None)

    print(f"\n  Prediction from scaling law:")
    print(f"    N_opt = {n_pred:,.0f}  ({n_pred/1e6:.1f}M)")
    print(f"    D_opt = {d_pred:,.0f}  ({d_pred/1e6:.1f}M tokens)")
    print(f"    L_min (predicted) = {loss_pred:.4f}")

    # Prepare data
    print("\n[1/3] Preparing data...")
    tokenizer, train_tokens, val_tokens, meta = prepare_data(seq_len=256)
    train_t = torch.from_numpy(train_tokens)
    val_t = torch.from_numpy(val_tokens)
    print(f"  Train tokens available: {train_t.numel():,}")
    print(f"  Val tokens available: {val_t.numel():,}")

    # Find closest feasible model configuration
    print("\n[2/3] Selecting model config...")
    mc, D_train = find_nearest_config(n_pred, d_pred)
    D_train = min(D_train, train_t.numel())  # Cap to available data

    # Create run
    run_config = IsoflopsRun(
        run_id="validation",
        compute_budget=C_target or (6 * mc.num_params() * D_train),
        model_config=mc,
        num_train_tokens=D_train,
        batch_size=32,
        learning_rate=3e-4,
    )

    print(f"\n  Validation run config:")
    print(f"    Model: {mc}")
    print(f"    Training tokens: {D_train:,}")
    print(f"    Optimizer steps: {run_config.num_optimizer_steps:,}")
    print(f"    C (effective): {run_config.compute_budget:.2e}")
    print(f"    Predicted loss: {loss_pred:.4f}")

    # Train
    print("\n[3/3] Training validation model...")
    model = MiniCausalLM(mc)
    val_loss = train_on_data(
        model=model,
        train_tokens=train_t,
        val_tokens=val_t,
        run=run_config,
        device=device,
        run_dir=VALIDATE_DIR,
        dtype=dtype,
    )

    # Compare
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"  Predicted loss:  {loss_pred:.4f}")
    print(f"  Actual loss:     {val_loss:.4f}")
    print(f"  Difference:      {val_loss - loss_pred:+.4f} ({abs(val_loss - loss_pred) / loss_pred * 100:.1f}%)")

    results = {
        "predicted_loss": loss_pred,
        "actual_loss": val_loss,
        "abs_error": abs(val_loss - loss_pred),
        "relative_error_pct": abs(val_loss - loss_pred) / loss_pred * 100,
        "model_config": {
            "hidden_size": mc.hidden_size,
            "num_hidden_layers": mc.num_hidden_layers,
            "num_attention_heads": mc.num_attention_heads,
            "head_dim": mc.head_dim,
            "intermediate_size": mc.intermediate_size,
            "n_params": mc.num_params(),
        },
        "train_tokens": D_train,
        "compute_budget": run_config.compute_budget,
    }

    # Save results
    import json
    (VALIDATE_DIR / "validation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {VALIDATE_DIR / 'validation_results.json'}")

    # Generate comparison plot
    _plot_comparison(predictions, results)

    return results


def _plot_comparison(predictions: dict, results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Predicted", "Actual"]
    values = [predictions["predicted_loss"], results["actual_loss"]]
    colors = ["steelblue", "coral"]

    ax.bar(labels, values, color=colors, width=0.4, edgecolor="black")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Prediction vs Actual Loss")

    diff = results["actual_loss"] - predictions["predicted_loss"]
    ax.text(1, values[1], f"  {values[1]:.4f}", ha="left", va="center", fontsize=10)
    ax.text(0, values[0], f"  {values[0]:.4f}", ha="left", va="center", fontsize=10)

    error_pct = results["relative_error_pct"]
    ax.text(0.5, max(values) * 0.95, f"Δ = {diff:+.4f} ({error_pct:.1f}%)",
            ha="center", fontsize=11, style="italic")

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_comparison.pdf", dpi=150)
    print(f"Comparison plot saved to {FIG_DIR / 'validation_comparison.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load predictions from analyze.py if available, else use defaults
    import json
    analysis_file = BASE_DIR / "analysis_predictions.json"
    if analysis_file.exists():
        predictions = json.loads(analysis_file.read_text())
    else:
        print("No predictions file found. Run analyze.py first, or using defaults...")
        predictions = {
            "predicted_N": 20_000_000,
            "predicted_D": 10_000_000,
            "predicted_loss": 3.0,
            "target_budget": 1e17,
        }

    run_validation(device, predictions, dtype=torch.bfloat16)

"""
IsoFLOPs experiment runner.

For each compute budget C_i, trains several models of varying size N_i
and finds which N gives the lowest validation loss.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import torch

from config import ModelConfig, IsoflopsRun
from data import prepare_data
from train import train_on_data

logger = logging.getLogger(__name__)

# ── Workspace ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "isoflops_runs"
RESULTS_FILE = BASE_DIR / "isoflops_results.json"

# ── Model prototype templates ────────────────────────────────────────

MODEL_PROTOTYPES = {
    "XXS": ModelConfig(vocab_size=5000, hidden_size=48, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, head_dim=12, intermediate_size=192),
    "XS":  ModelConfig(vocab_size=5000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, head_dim=16, intermediate_size=256),
    "S":   ModelConfig(vocab_size=5000, hidden_size=96, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, head_dim=24, intermediate_size=384),
    "M":   ModelConfig(vocab_size=5000, hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, head_dim=32, intermediate_size=512),
    "L":   ModelConfig(vocab_size=5000, hidden_size=192, num_hidden_layers=5, num_attention_heads=6, num_key_value_heads=6, head_dim=32, intermediate_size=768),
}


def generate_run_matrix(compute_budgets: list[float]) -> list[list[IsoflopsRun]]:
    """
    For each compute budget C, generate IsoflopsRun entries for each model prototype.
    D = C / (6 * N) determines how many tokens to train on.

    Skips configurations that would require more tokens than available
    or produce too few steps.
    """
    matrix = []
    for i, C in enumerate(compute_budgets):
        runs_for_budget = []
        for name, mc in MODEL_PROTOTYPES.items():
            N = mc.num_params()
            D = int(C / (6 * N))  # tokens to train on
            steps = D // (mc.seq_len * 32)  # default batch_size 32

            # Sanity checks
            if D < mc.seq_len * 32:
                print(f"  SKIP {name}: D={D:,} too few tokens for C={C:.1e}, N={N:,}")
                continue
            if steps < 20:
                print(f"  SKIP {name}: only {steps} steps for C={C:.1e}, N={N:,}")
                continue

            run = IsoflopsRun(
                run_id=f"C{i}_{name}",
                compute_budget=C,
                model_config=mc,
                num_train_tokens=D,
                batch_size=32,
            )
            runs_for_budget.append(run)

        matrix.append(runs_for_budget)
    return matrix


def print_run_matrix(matrix: list[list[IsoflopsRun]]) -> None:
    """Pretty-print the run matrix."""
    for i, runs in enumerate(matrix):
        C = runs[0].compute_budget
        print(f"\n  C = {C:.1e} FLOPs ({len(runs)} model sizes)")
        print(f"  {'Name':>5}  {'N':>10}  {'D':>12}  {'Steps':>8}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*8}")
        for r in runs:
            name = r.run_id.split("_")[1]
            print(f"  {name:>5}  {r.n_params:>10,}  {r.num_train_tokens:>12,}  {r.num_optimizer_steps:>8,}")


def run_isoflops(
    device: torch.device,
    compute_budgets: list[float],
    *,
    rebuild_data: bool = False,
    lr: float = 3e-4,
    dtype: torch.dtype = torch.bfloat16,
) -> list[list[IsoflopsRun]]:
    """
    Run the full IsoFLOPs experiment suite.

    Returns matrix of IsoflopsRun with final_val_loss filled in.
    """
    print("=" * 60)
    print("IsoFLOPs Experiment Runner")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1/3] Preparing data...")
    if rebuild_data:
        if RUNS_DIR.exists():
            shutil.rmtree(RUNS_DIR)
        tokenizer, train_tokens, val_tokens, meta = prepare_data(
            seq_len=256, tokenizer=None, special_tokens=None
        )
        # Save tokenized data
    else:
        # Try loading cached data, or prepare fresh
        tokenizer, train_tokens, val_tokens, meta = prepare_data(
            seq_len=256, tokenizer=None, special_tokens=None
        )

    train_t = torch.from_numpy(train_tokens)
    val_t = torch.from_numpy(val_tokens)

    print(f"  Train: {train_t.numel():,} tokens across {train_t.shape[0]} sequences")
    print(f"  Val:   {val_t.numel():,} tokens")

    # 2. Generate run matrix
    print("\n[2/3] Generating run matrix...")
    matrix = generate_run_matrix(compute_budgets)
    print_run_matrix(matrix)

    total_runs = sum(len(runs) for runs in matrix)
    print(f"\n  Total training runs: {total_runs}")

    # 3. Run experiments
    print(f"\n[3/3] Running experiments on {device}...")

    all_results = []

    for budget_idx, runs in enumerate(matrix):
        C = runs[0].compute_budget
        print(f"\n  --- Budget {budget_idx + 1}/{len(matrix)}: C = {C:.1e} ---")

        for run_idx, run in enumerate(runs):
            run_dir = RUNS_DIR / run.run_id
            if run_dir.exists():
                # Check if results already exist
                result_file = run_dir / "results.json"
                if result_file.exists():
                    data = json.loads(result_file.read_text())
                    run.final_val_loss = data["final_val_loss"]
                    run.training_time_seconds = data["train_time_seconds"]
                    print(f"\n  [{run.run_id}] loaded cached result: loss={run.final_val_loss:.4f}")
                    all_results.append(data)
                    continue

            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{run.run_id}] N={run.n_params:,}, D={run.num_train_tokens:,}, "
                  f"steps={run.num_optimizer_steps:,}")

            try:
                model = build_model(run.model_config)

                val_loss = train_on_data(
                    model=model,
                    train_tokens=train_t,
                    val_tokens=val_t,
                    run=run,
                    device=device,
                    run_dir=run_dir,
                    dtype=dtype,
                )

                run.final_val_loss = val_loss
                run.training_time_seconds = (run_dir / "results.json").exists() and (
                    json.loads((run_dir / "results.json").read_text())["train_time_seconds"]
                )

                # Load saved results
                result_data = json.loads((run_dir / "results.json").read_text())
                all_results.append(result_data)

            except Exception as e:
                print(f"  ERROR in run {run.run_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

            # Small delay between runs to let GPU cool
            if run_idx < len(runs) - 1:
                time.sleep(5)

    # Save all results
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to {RESULTS_FILE}")

    return matrix


def build_model(config: ModelConfig) -> torch.nn.Module:
    """Build a model from config."""
    from model import MiniCausalLM
    return MiniCausalLM(config)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    budgets = [1e15, 3e15, 1e16, 3e16]
    run_isoflops(device, compute_budgets=budgets, lr=3e-4, dtype=torch.bfloat16)

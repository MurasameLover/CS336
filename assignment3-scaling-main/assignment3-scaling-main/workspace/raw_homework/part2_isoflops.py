"""
CS336 Assignment 3 - Part 2: IsoFLOPs Scaling Laws

Task (a): Compute-optimal model size scaling law  N_opt = A * C^a
Task (b): Compute-optimal dataset size scaling law  D_opt = B * C^b

Approach: For each IsoFLOPs profile (fixed C), find the run with the lowest
final validation loss. Its parameter count N gives N_opt(C), and D_opt(C) is
computed as C / (6 * N_opt). Then fit power laws in log space.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

matplotlib.rcParams.update({"font.size": 12})
sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "isoflops_curves.json"
OODIR = Path(__file__).resolve().parent / "figures"
OODIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────


def load_data(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by_budget(runs: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = {}
    for run in runs:
        groups.setdefault(run["compute_budget"], []).append(run)
    for c in groups:
        groups[c].sort(key=lambda r: r["parameters"])
    return groups


def power_law(x, A, a):
    return A * x**a


def find_optimal_points(groups: dict[float, list[dict]]):
    """Return (C_arr, N_opt_arr, D_opt_arr) for the run with lowest loss per budget."""
    C_list, N_list, D_list = [], [], []
    for c in sorted(groups):
        best = min(groups[c], key=lambda r: r["final_loss"])
        n_opt = best["parameters"]
        d_opt = c / (6 * n_opt)  # C = 6ND → D = C / (6N)
        C_list.append(c)
        N_list.append(n_opt)
        D_list.append(d_opt)
    return np.array(C_list), np.array(N_list, dtype=float), np.array(D_list, dtype=float)


def fit_power_law_logspace(x, y):
    """Fit y = A * x^a in log space (standard for scaling laws). Returns (A, a)."""
    coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    a = coeffs[0]
    A = np.exp(coeffs[1])
    return A, a


def print_fit_results(name: str, y_label: str, A: float, a: float) -> None:
    print(f"\n--- {name} ---")
    print(f"  Log-space fit (primary):")
    print(f"    {y_label} = {A:.4f} * C^{a:.4f}")
    print(f"    log({y_label}) = {np.log(A):.4f} + {a:.4f} * log(C)")
    print(f"    Exponent = {a:.4f}")


def plot_scaling_law(
    C_data,
    y_data,
    A_fit: float,
    a_fit: float,
    y_label_tex: str,       # e.g. "N_{opt}" or "D_{opt}"
    y_label_axis: str,      # e.g. "Optimal Model Size $N_{opt}$ (parameters)"
    title: str,
    save_name: str,
    target_budgets=(1e23, 1e24),
):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data points
    ax.scatter(C_data, y_data, color="steelblue", s=60, zorder=3, label="IsoFLOPs optimal points")

    # Fitted curve
    C_smooth = np.logspace(np.log10(5e18), np.log10(3e24), 200)
    y_smooth = power_law(C_smooth, A_fit, a_fit)
    ax.plot(C_smooth, y_smooth, "r-", linewidth=2,
            label=f"${y_label_tex} = {A_fit:.3f}\,C^{{{a_fit:.3f}}}$")

    # Shaded regions
    ax.axvspan(5e18, max(C_data) * 1.1, alpha=0.08, color="green", label="Fitting range")
    ax.axvspan(max(C_data) * 1.1, 3e24, alpha=0.08, color="orange", label="Extrapolation")

    # Predictions
    for c_target in target_budgets:
        y_pred = power_law(c_target, A_fit, a_fit)
        ax.scatter([c_target], [y_pred], color="darkred", s=120, marker="*", zorder=5)
        label = f"$10^{{{int(np.log10(c_target))}}}$"
        ax.annotate(label,
                    (c_target, y_pred), textcoords="offset points",
                    xytext=(15, 10), fontsize=9, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C$ (FLOPs)")
    ax.set_ylabel(y_label_axis)
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    save_path = OODIR / save_name
    fig.savefig(save_path, dpi=150)
    print(f"  Figure saved to {save_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────


def main():
    runs = load_data(DATA_PATH)
    groups = group_by_budget(runs)
    budgets = sorted(groups.keys())
    print(f"Found {len(budgets)} IsoFLOPs profiles:")
    for c in budgets:
        print(f"  C = {c:.0e}  ({len(groups[c])} runs)")

    # Extract optimal points per budget
    C_arr, N_arr, D_arr = find_optimal_points(groups)

    print("\n" + "=" * 75)
    print(f"{'C (FLOPs)':>20}  {'N_opt':>15}  {'D_opt (tokens)':>20}  {'Min loss':>12}")
    print("=" * 75)
    for c, n, d, budget in zip(C_arr, N_arr, D_arr, budgets):
        best_loss = min(groups[budget], key=lambda r: r["final_loss"])["final_loss"]
        print(f"{c:>20.0e}  {n:>15,.0f}  {d:>20,.0f}  {best_loss:>12.4f}")

    # ── Task (a): N_opt scaling ──────────────────────────────────────
    A_N, a_N = fit_power_law_logspace(C_arr, N_arr)
    print_fit_results("Task (a): Model Size Scaling", "N_{opt}", A_N, a_N)

    plot_scaling_law(
        C_data=C_arr,
        y_data=N_arr,
        A_fit=A_N,
        a_fit=a_N,
        y_label_tex="N_{opt}",
        y_label_axis="Optimal Model Size $N_{opt}$ (parameters)",
        title="Task (a): Scaling Law for Optimal Model Size",
        save_name="part2a_model_size_scaling.pdf",
    )

    # ── Task (b): D_opt scaling ──────────────────────────────────────
    A_D, a_D = fit_power_law_logspace(C_arr, D_arr)
    print_fit_results("Task (b): Dataset Size Scaling", "D_{opt}", A_D, a_D)

    plot_scaling_law(
        C_data=C_arr,
        y_data=D_arr,
        A_fit=A_D,
        a_fit=a_D,
        y_label_tex="D_{opt}",
        y_label_axis="Optimal Dataset Size $D_{opt}$ (tokens)",
        title="Task (b): Scaling Law for Optimal Dataset Size",
        save_name="part2b_dataset_size_scaling.pdf",
    )

    # ── Predictions ──────────────────────────────────────────────────
    target_budgets = [1e23, 1e24]
    print(f"\n" + "=" * 75)
    print(f"  Predicted optimal configurations")
    print("=" * 75)
    for c_target in target_budgets:
        n_pred = power_law(c_target, A_N, a_N)
        d_pred = power_law(c_target, A_D, a_D)
        # consistency check: C ≈ 6 * N * D
        c_recovered = 6 * n_pred * d_pred
        print(f"\n  C = {c_target:.0e} FLOPs")
        print(f"    N_opt = {n_pred:,.0f} parameters  ({n_pred/1e9:.2f}B)")
        print(f"    D_opt = {d_pred:,.0f} tokens  ({d_pred/1e9:.1f}B)")
        print(f"    C ≈ 6·N·D  →  {c_recovered:.2e}  (recovered)")

    print(f"\nAll figures saved to {OODIR}/")


if __name__ == "__main__":
    main()

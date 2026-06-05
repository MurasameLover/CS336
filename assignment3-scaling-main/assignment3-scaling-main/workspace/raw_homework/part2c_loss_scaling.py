"""
CS336 Assignment 3 — Part 2c: Loss vs Compute Scaling Law

Fits a power law  L_min = k * C^(-γ)  to the IsoFLOPs Pareto frontier.

The exponent γ tells us how quickly loss drops as we scale compute.
A larger γ means more efficient returns to scale.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 12})
sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "isoflops_curves.json"
OODIR = Path(__file__).resolve().parent / "figures"
OODIR.mkdir(parents=True, exist_ok=True)


def load_data(path):
    with open(path) as f:
        return json.load(f)


def group_by_budget(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r["compute_budget"], []).append(r)
    for c in groups:
        groups[c].sort(key=lambda x: x["parameters"])
    return groups


def main():
    runs = load_data(DATA_PATH)
    groups = group_by_budget(runs)
    budgets = sorted(groups.keys())

    # Per budget: find the minimum loss (Pareto frontier)
    C_arr, L_min_arr = [], []
    print(f"IsoFLOPs Pareto frontier ({len(budgets)} points):")
    print(f"{'C (FLOPs)':>20}  {'L_min':>12}  {'N at min':>15}")
    print("-" * 55)
    for c in budgets:
        best = min(groups[c], key=lambda r: r["final_loss"])
        C_arr.append(c)
        L_min_arr.append(best["final_loss"])
        print(f"{c:>20.0e}  {best['final_loss']:>12.4f}  {best['parameters']:>15,d}")

    C = np.array(C_arr)
    L = np.array(L_min_arr)

    # Fit  L = k * C^(-γ)   →   log L = log k - γ * log C
    log_C = np.log(C)
    log_L = np.log(L)
    coeffs = np.polyfit(log_C, log_L, deg=1)
    gamma = -coeffs[0]          # slope → -γ
    log_k = coeffs[1]
    k = np.exp(log_k)

    print(f"\nFitted power law (log-space):")
    print(f"  L_min = {k:.4f} * C^({-gamma:.4f})")
    print(f"  log(L_min) = {log_k:.4f} - {gamma:.4f} * log(C)")
    print(f"  Exponent γ = {gamma:.4f}")

    # Predictions
    targets = [1e23, 1e24]
    print(f"\nPredictions:")
    print("-" * 55)
    for ct in targets:
        L_pred = k * ct ** (-gamma)
        print(f"  C = {ct:.0e} FLOPs  →  L_min = {L_pred:.4f}")

    # --------------------- Plot ---------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(C, L, color="steelblue", s=60, zorder=3, label="IsoFLOPs Pareto points")

    C_smooth = np.logspace(np.log10(5e18), np.log10(3e24), 200)
    L_smooth = k * C_smooth ** (-gamma)
    ax.plot(C_smooth, L_smooth, "r-", linewidth=2,
            label=f"$L_{{min}} = {k:.3f}\,C^{{{-gamma:.3f}}}$")

    ax.axvspan(5e18, max(C) * 1.1, alpha=0.08, color="green", label="Fitting range")
    ax.axvspan(max(C) * 1.1, 3e24, alpha=0.08, color="orange", label="Extrapolation")

    for ct in targets:
        Lp = k * ct ** (-gamma)
        ax.scatter([ct], [Lp], color="darkred", s=120, marker="*", zorder=5)
        label = f"$10^{{{int(np.log10(ct))}}}$: {Lp:.3f}"
        ax.annotate(label, (ct, Lp), textcoords="offset points",
                    xytext=(15, 10), fontsize=9, color="darkred",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C$ (FLOPs)")
    ax.set_ylabel("Minimum Validation Loss $L_{min}$")
    ax.set_title("Scaling Law: Loss vs Compute Budget (Pareto Frontier)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    save_path = OODIR / "part2c_loss_scaling.pdf"
    fig.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

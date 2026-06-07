"""
Analyze IsoFLOPs results: fit power laws, generate plots, make predictions.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 12})

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "isoflops_results.json"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> list[dict]:
    import json
    with open(RESULTS_FILE) as f:
        return json.load(f)


def group_by_budget(results: list[dict]) -> dict[float, list[dict]]:
    groups = {}
    for r in results:
        c = r["compute_budget"]
        groups.setdefault(c, []).append(r)
    for c in groups:
        groups[c].sort(key=lambda x: x["n_params"])
    return groups


def power_law(x, A, a):
    return A * x**a


def fit_logspace(x, y):
    """Fit y = A * x^a in log space."""
    coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    a = coeffs[0]
    A = np.exp(coeffs[1])
    return A, a


def analyze():
    results = load_results()
    groups = group_by_budget(results)
    budgets = sorted(groups.keys())

    print(f"Loaded {len(results)} runs across {len(budgets)} budgets\n")

    # ── 1. Per-budget U-curves ─────────────────────────────────────
    print("=" * 60)
    print("IsoFLOPs Profiles (U-curves)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    C_vals, N_opt_vals, D_opt_vals, L_min_vals = [], [], [], []

    for idx, c in enumerate(budgets):
        runs = groups[c]
        Ns = np.array([r["n_params"] for r in runs])
        losses = np.array([r["final_val_loss"] for r in runs])
        Ds = c / (6 * Ns)

        best_idx = np.argmin(losses)
        N_opt = Ns[best_idx]
        L_min = losses[best_idx]
        D_opt = Ds[best_idx]

        C_vals.append(c)
        N_opt_vals.append(N_opt)
        D_opt_vals.append(D_opt)
        L_min_vals.append(L_min)

        print(f"\nC = {c:.1e} FLOPs:")
        print(f"  {'Name':>6}  {'N':>10}  {'D (tokens)':>15}  {'Loss':>8}")
        for i, r in enumerate(runs):
            marker = " ← best" if i == best_idx else ""
            print(f"  {r['run_id']:>6}  {Ns[i]:>10,}  {Ds[i]:>15,.0f}  {losses[i]:>8.4f}{marker}")

        # U-curve plot
        ax = axes[idx]
        ax.scatter(Ns, losses, color="steelblue", s=50, zorder=3)
        # Sort for line plot
        sort_idx = np.argsort(Ns)
        ax.plot(Ns[sort_idx], losses[sort_idx], "b-", alpha=0.4)
        ax.scatter([N_opt], [L_min], color="red", s=100, marker="*", zorder=4,
                   label=f"Optimum: {N_opt:,}")
        ax.set_xscale("log")
        ax.set_xlabel("Model Size N (parameters)")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"C = {c:.1e} FLOPs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots (there are 6 panels but only len(budgets) used)
    for idx in range(len(budgets), 6):
        axes[idx].set_visible(False)

    fig.suptitle("IsoFLOPs Profiles: Loss vs Model Size per Compute Budget", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "isoflops_u_curves.pdf", dpi=150)
    print(f"\nU-curves saved to {FIG_DIR / 'isoflops_u_curves.pdf'}")

    # ── 2. Power law: N_opt vs C ──────────────────────────────────
    C_arr = np.array(C_vals)
    N_arr = np.array(N_opt_vals, dtype=float)
    D_arr = np.array(D_opt_vals, dtype=float)
    L_arr = np.array(L_min_vals)

    A_N, a_N = fit_logspace(C_arr, N_arr)
    A_D, a_D = fit_logspace(C_arr, D_arr)
    k_L, gamma_L = fit_logspace(C_arr, L_arr)
    gamma_L = -gamma_L  # L = k * C^(-γ), so slope is -γ

    print("\n" + "=" * 60)
    print("Fitted Power Laws (log-space)")
    print("=" * 60)
    print(f"  N_opt = {A_N:.4f} * C^{a_N:.4f}")
    print(f"  D_opt = {A_D:.4f} * C^{a_D:.4f}")
    print(f"  L_min = {k_L:.4f} * C^(-{gamma_L:.4f})")
    print(f"\n  Self-consistency: a + b = {a_N + a_D:.4f} (should be ≈ 1.0)")

    # ── 3. Scaling law plots ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    C_smooth = np.logspace(np.log10(min(C_arr) * 0.8), np.log10(max(C_arr) * 5), 200)

    # N_opt plot
    ax = axes[0]
    ax.scatter(C_arr, N_arr, color="steelblue", s=60, zorder=3)
    ax.plot(C_smooth, power_law(C_smooth, A_N, a_N), "r-", linewidth=2,
            label=f"$N_{{opt}} = {A_N:.3f}\,C^{{{a_N:.3f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C$ (FLOPs)")
    ax.set_ylabel("$N_{opt}$ (parameters)")
    ax.set_title("Optimal Model Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # D_opt plot
    ax = axes[1]
    ax.scatter(C_arr, D_arr, color="steelblue", s=60, zorder=3)
    ax.plot(C_smooth, power_law(C_smooth, A_D, a_D), "r-", linewidth=2,
            label=f"$D_{{opt}} = {A_D:.3f}\,C^{{{a_D:.3f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C$ (FLOPs)")
    ax.set_ylabel("$D_{opt}$ (tokens)")
    ax.set_title("Optimal Dataset Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # L_min plot
    ax = axes[2]
    ax.scatter(C_arr, L_arr, color="steelblue", s=60, zorder=3)
    ax.plot(C_smooth, k_L * C_smooth ** (-gamma_L), "r-", linewidth=2,
            label=f"$L_{{min}} = {k_L:.3f}\,C^{{{-gamma_L:.3f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget $C$ (FLOPs)")
    ax.set_ylabel("$L_{min}$")
    ax.set_title("Minimum Loss (Pareto Frontier)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Scaling Laws from IsoFLOPs Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_laws.pdf", dpi=150)
    print(f"\nScaling laws figure saved to {FIG_DIR / 'scaling_laws.pdf'}")

    # ── 4. Predictions ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)

    # Find the best target budget: 10x the max fitted budget
    C_max_fitted = max(C_arr)
    C_target = C_max_fitted * 10  # extrapolate 10x

    N_pred = power_law(C_target, A_N, a_N)
    D_pred = power_law(C_target, A_D, a_D)
    L_pred = k_L * C_target ** (-gamma_L)
    C_check = 6 * N_pred * D_pred

    print(f"\n  Max fitted budget:   C = {C_max_fitted:.1e} FLOPs")
    print(f"  Target budget:       C = {C_target:.1e} FLOPs (10× extrapolation)")
    print(f"\n  Predicted N_opt:    {N_pred:,.0f}  ({N_pred/1e6:.2f}M)")
    print(f"  Predicted D_opt:    {D_pred:,.0f}  ({D_pred/1e6:.2f}M tokens)")
    print(f"  Predicted L_min:    {L_pred:.4f}")
    print(f"  Consistency check:  6·N·D = {C_check:.2e} (target: {C_target:.2e})")

    return {
        "A_N": A_N, "a_N": a_N,
        "A_D": A_D, "a_D": a_D,
        "k_L": k_L, "gamma_L": gamma_L,
        "target_budget": C_target,
        "predicted_N": N_pred,
        "predicted_D": D_pred,
        "predicted_loss": L_pred,
    }


if __name__ == "__main__":
    analyze()

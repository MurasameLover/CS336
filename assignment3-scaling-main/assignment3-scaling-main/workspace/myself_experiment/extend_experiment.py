"""
分析 D_opt 为何呈现近线性关系，并添加新预算扩展实验。
"""
import sys; sys.path.insert(0, '.')
import json, torch, time
from pathlib import Path
from config import ModelConfig, IsoflopsRun
from data import prepare_data
from model import MiniCausalLM
from train import train_on_data

# ── 分析 D_opt 问题 ─────────────────────────────────────────
print("=" * 60)
print("分析 D_opt = C/(6*N_opt) 的构成")
print("=" * 60)

from run_isoflops import MODEL_PROTOTYPES

# 对每个模型分析 embedding vs transformer 参数占比
print(f"\n{'Name':>6} {'d_model':>8} {'Layers':>7} {'Total':>10} {'Non-emb':>10} {'Embed%':>8} {'Non-emb%':>8}")
print("-" * 60)
for name, mc in MODEL_PROTOTYPES.items():
    total = mc.num_params()
    non_emb = mc.num_non_embedding_params()
    emb = total - non_emb
    emb_pct = emb / total * 100
    non_emb_pct = non_emb / total * 100
    print(f"{name:>6} {mc.hidden_size:>8} {mc.num_hidden_layers:>7} {total:>10,} {non_emb:>10,} {emb_pct:>7.1f}% {non_emb_pct:>7.1f}%")

# 分析 N_opt 随 C 增加时的构成变化
print(f"\n{'Budget C':>12} {'N_opt':>10} {'Non-emb':>10} {'D_opt(C/6N)':>14}")
print("-" * 50)
results = json.loads(Path('isoflops_results.json').read_text())
groups = {}
for r in results:
    groups.setdefault(r['compute_budget'], []).append(r)

# Load existing optimal N values from analysis
for C in sorted(groups.keys()):
    runs = groups[C]
    best = min(runs, key=lambda r: r['final_val_loss'])
    N = best['n_params']
    D = C / (6 * N)
    # Find model config for this N
    for name, mc in MODEL_PROTOTYPES.items():
        if mc.num_params() == N:
            non_emb = mc.num_non_embedding_params()
            break
    else:
        non_emb = 0
    old_budgets = [5e14, 1e15, 3e15]
    status = "fit" if C in old_budgets else "NEW"
    print(f"{C:>12.1e} {N:>10,} {non_emb:>10,} {D:>14,.0f}  [{status}]")

# ── 添加新预算 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("添加新预算: 1e16 (M, L, XL) 和 3e16 (L, XL)")
print(f"{'='*60}")

tok, train_t, val_t, meta = prepare_data()
train_data = torch.from_numpy(train_t); val_data = torch.from_numpy(val_t)
device = torch.device('cuda')

# New budgets (avoid 5e15 - too close to validation)
new_budgets = [
    (3, 1e16, ["M", "L", "XL"]),   # budget_idx=3
    (4, 3e16, ["L", "XL"]),         # budget_idx=4
]

for idx, C, model_names in new_budgets:
    print(f"\n--- Budget {idx}: C={C:.1e} ---")
    for name in model_names:
        mc = MODEL_PROTOTYPES[name]
        run_id = f"C{idx}_{name}"
        run_dir = Path('isoflops_runs') / run_id
        rf = run_dir / 'results.json'
        if rf.exists():
            d = json.loads(rf.read_text())
            print(f"  [{run_id}] cached: loss={d['final_val_loss']:.4f}")
            continue

        N = mc.num_params()
        D = int(C / (6 * N))
        run = IsoflopsRun(run_id=run_id, compute_budget=C, model_config=mc, num_train_tokens=D, batch_size=32)
        run_dir.mkdir(parents=True, exist_ok=True)
        steps = D // (256 * 32)
        print(f"\n  [{run_id}] N={N:,} D={D:,} steps={steps:,}")
        model = MiniCausalLM(mc)
        loss = train_on_data(model, train_data, val_data, run, device, run_dir, log_every=max(1, steps//5))
        print(f"  loss={loss:.4f}")
        time.sleep(3)

# Recombine
all_results = []
for f in Path('isoflops_runs').glob('*/results.json'):
    all_results.append(json.loads(f.read_text()))
Path('isoflops_results.json').write_text(json.dumps(all_results, indent=2))
print(f"\nAll {len(all_results)} runs saved")

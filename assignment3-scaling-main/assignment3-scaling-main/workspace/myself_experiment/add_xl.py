import sys; sys.path.insert(0, '.')
import torch, json, time
from pathlib import Path
from config import ModelConfig, IsoflopsRun
from data import prepare_data
from model import MiniCausalLM
from train import train_on_data

XL = ModelConfig(vocab_size=5000, hidden_size=256, num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=8, head_dim=32, intermediate_size=1024)
print(f'XL: total={XL.num_params():,} non-emb={XL.num_non_embedding_params():,}')

budgets = [(0, 5e14), (1, 1e15), (2, 3e15)]
tok, train_t, val_t, meta = prepare_data()
train_data = torch.from_numpy(train_t); val_data = torch.from_numpy(val_t)
device = torch.device('cuda')

for idx, C in budgets:
    N = XL.num_params()
    D = int(C / (6 * N))
    run = IsoflopsRun(run_id=f'C{idx}_XL', compute_budget=C, model_config=XL, num_train_tokens=D, batch_size=32)
    run_dir = Path('isoflops_runs') / run.run_id
    rf = run_dir / 'results.json'
    if rf.exists():
        d = json.loads(rf.read_text())
        loss = d["final_val_loss"]
        print(f'[{run.run_id}] cached: loss={loss:.4f}')
        continue
    run_dir.mkdir(parents=True, exist_ok=True)
    model = MiniCausalLM(XL)
    steps = D // (256 * 32)
    print(f'\n[{run.run_id}] C={C:.1e} N={N:,} D={D:,} steps={steps:,}')
    loss = train_on_data(model, train_data, val_data, run, device, run_dir, log_every=max(1, steps//5))
    print(f'  loss={loss:.4f}')
    time.sleep(3)

all_results = []
for f in Path('isoflops_runs').glob('*/results.json'):
    all_results.append(json.loads(f.read_text()))
Path('isoflops_results.json').write_text(json.dumps(all_results, indent=2))
print(f'\nAll {len(all_results)} runs saved')

"""M4c curves: M4b, the two low-lr runs and the two mapping variants on one panel."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CFG = Path(r'E:\FNO_research\repos\fno_light\repro_lu2026\configs')
FIG = Path(r'E:\FNO_research\repos\fno_light\repro_lu2026\reports')
M4B_LOG = Path(r'E:\FNO_research\results\m4b_train_20261020_132524\m4_train_log.jsonl')


def series(recs, key):
    ep, val = [], []
    for r in recs:
        if key in r and r[key] is not None:
            ep.append(r['epoch'])
            val.append(r[key])
    return ep, val


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
colors = {'M4b (lr 1e-3, 500 ep)': 'tab:gray', 'lr 3e-4 (100 ep)': 'tab:green',
          'lr 1e-4 (100 ep)': 'tab:purple', 'mapping local (150 ep)': 'tab:orange',
          'mapping interp (150 ep)': 'tab:red'}

m4b = [json.loads(x) for x in M4B_LOG.read_text(encoding='utf-8').splitlines() if x.strip()]
m4b150 = [r for r in m4b if r['epoch'] <= 150]
low = json.loads((CFG / 'm4c_lowlr.json').read_text(encoding='utf-8'))
abl = json.loads((CFG / 'm4c_ablation.json').read_text(encoding='utf-8')) if (CFG / 'm4c_ablation.json').exists() else {'variants': {}}

runs = {'M4b (lr 1e-3, 500 ep)': m4b150}
for lr, v in low['runs'].items():
    runs[f'lr {float(lr):g}'.replace('0.0003', '3e-4').replace('0.0001', '1e-4') + ' (100 ep)'] = v['history']
for name, v in abl['variants'].items():
    runs[f'mapping {name} (150 ep)'] = v['history']

for name, recs in runs.items():
    c = colors.get(name, None)
    ep, tr = series(recs, 'train_l1')
    axes[0].plot(ep, tr, color=c, lw=1.1, label=name)
    ep2, va = series(recs, 'val_l1')
    axes[1].plot(ep2, va, color=c, lw=1.1, ls='--', label=name)

for ax, title in zip(axes, ['train L1 (normalized speed)', 'validation L1 (normalized speed)']):
    ax.set(xlabel='epoch', ylabel='L1', yscale='log', title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle('M4c: learning-rate and grid-mapping comparisons (scattering input, batch 32, seed 42)')
fig.tight_layout()
p = FIG / 'm4c_curves.png'
if p.exists():
    raise SystemExit(f'refusing to overwrite {p}')
fig.savefig(p, dpi=120)
print('wrote', p)

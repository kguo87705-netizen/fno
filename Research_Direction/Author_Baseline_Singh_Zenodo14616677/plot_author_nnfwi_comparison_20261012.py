"""Compare saved author conventional and NN reconstructions; never optimize."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.build' / 'singh-nnfwi-author-20261012'
cases = [c for c in range(1, 5) if (OUT / f'case{c}' / 'final_fields.npz').exists()]
fig, axes = plt.subplots(len(cases), 3, figsize=(12, 2.8 * len(cases)), squeeze=False)
all_metrics = []
costs = json.loads((ROOT / '.build/author-singh-summary-20261008/time_axis_costs.json').read_text())
times = [84.93598270000075, 86.83277369999996, 86.74213929999678, 81.7757844000007]
for row, case in enumerate(cases):
    saved = torch.load(ROOT / f'.build/author-singh-case{case}-35epochs-20261008/fields.pt', weights_only=True)
    conv = saved['reconstruction'].numpy().squeeze()
    truth = saved['truth'].numpy().squeeze()
    nn = np.load(OUT / f'case{case}' / 'final_fields.npz')
    assert np.array_equal(truth, nn['truth'])
    error = (conv.astype(np.float64) - truth.astype(np.float64)) ** 2
    cm = dict(case=case, material_mse=float(error[1:-1, 1:-1].mean()),
              inner_mse=float(error[2:-2, 2:-2].mean()),
              final_time_integral_cost=costs[str(case)]['final']['last_axis_time'],
              historical_wall_seconds=times[case - 1])
    nm = json.loads((OUT / f'case{case}/metrics.json').read_text())
    all_metrics.append(dict(conventional=cm, nn=nm))
    fields = [truth, conv, nn['reconstruction']]
    titles = [f'Case {case}: truth', f'Conventional: MSE {cm["material_mse"]:.5f}',
              f'NN (35 updates): MSE {nm["material_mse"]:.5f}']
    for col in range(3):
        im = axes[row, col].imshow(fields[col][1:-1, 1:-1].T, vmin=0, vmax=1, cmap='coolwarm', origin='upper')
        axes[row, col].set_title(titles[col], fontsize=11)
        axes[row, col].axis('off')
fig.subplots_adjust(right=.90, hspace=.22, wspace=.12)
fig.colorbar(im, cax=fig.add_axes([.92, .15, .015, .7]), label='Material gamma')
path = OUT / ('four_case_comparison.png' if len(cases) == 4 else 'partial_comparison.png')
fig.savefig(path, dpi=170, bbox_inches='tight')
plt.close(fig)
(OUT / 'comparison_metrics.json').write_text(json.dumps(all_metrics, indent=2), encoding='utf-8')
print(json.dumps(all_metrics, indent=2))
print(path)

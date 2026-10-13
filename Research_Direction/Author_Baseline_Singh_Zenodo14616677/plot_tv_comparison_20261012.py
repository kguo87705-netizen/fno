"""Summarize the authorized TV grid without further inversion."""
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.build/singh-tv-author-20261012'
rows = json.loads((OUT / 'results.json').read_text())
reference = json.loads((ROOT / '.build/singh-nnfwi-author-20261012/comparison_metrics.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--expected-k', nargs='+', type=float, default=[.1, 1.])
args = parser.parse_args()
groups = {k: sorted([r for r in rows if r['k'] == k], key=lambda r: r['case']) for k in (.1, 1., 10.)}
complete = {k: rs for k, rs in groups.items() if len(rs) == 4}
if not complete:
    raise RuntimeError('No complete four-case TV weight available')
best = min(complete, key=lambda k: np.mean([r['material_mse'] for r in complete[k]]))
grid_complete = all(k in complete for k in args.expected_k)
summary = dict(best_k_by_mean_material_mse=best if grid_complete else None,
    plotted_k=best, grid_complete=grid_complete,
    selection=('oracle single common k; minimum four-case mean material MSE' if grid_complete
               else 'Grid incomplete: only completed weights shown; no oracle optimum established'), weights={})
for k, rs in complete.items():
    mse = np.array([r['material_mse'] for r in rs])
    inner = np.array([r['inner_mse'] for r in rs])
    nn_mse = np.array([r['nn']['material_mse'] for r in reference])
    nn_inner = np.array([r['nn']['inner_mse'] for r in reference])
    summary['weights'][str(k)] = dict(material_mse=mse.tolist(), inner_mse=inner.tolist(),
        ratio_to_nn=(mse / nn_mse).tolist(), ratio_inner_to_nn=(inner / nn_inner).tolist(),
        pass_cases=(mse <= 1.25 * nn_mse).tolist(), pass_count=int(np.sum(mse <= 1.25 * nn_mse)),
        inner_pass_count=int(np.sum(inner <= 1.25 * nn_inner)),
        mean_material_mse=float(mse.mean()), lambdas=[r['tv_lambda'] for r in rs],
        final_cost=[r['final_time_integral_cost'] for r in rs],
        fit_seconds=sum(r['fit_seconds'] for r in rs), wall_seconds=sum(r['case_wall_seconds'] for r in rs))
if grid_complete:
    summary['per_case_oracle_lower_bound'] = np.min(np.array([v['material_mse'] for v in summary['weights'].values()]), axis=0).tolist()
ordered_k = sorted(complete)
means = [float(np.mean([r['material_mse'] for r in complete[k]])) for k in ordered_k]
summary['tested_k'] = ordered_k
summary['mean_mse_by_k'] = means
summary['case_mse_by_increasing_k'] = [[complete[k][c]['material_mse'] for k in ordered_k] for c in range(4)]
summary['case_mse_monotone_decreasing_on_tested_grid'] = [bool(np.all(np.diff(values) <= 0)) for values in summary['case_mse_by_increasing_k']]
summary['total_case_wall_seconds'] = sum(r['case_wall_seconds'] for r in rows)
summary['conventional_timing'] = next(r for r in rows if r['k'] == 0)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
fig, axes = plt.subplots(4, 4, figsize=(16, 11))
for row, case in enumerate(range(1, 5)):
    conv = torch.load(ROOT / f'.build/author-singh-case{case}-35epochs-20261008/fields.pt', weights_only=True)
    truth = conv['truth'].numpy().squeeze()
    nn = np.load(ROOT / f'.build/singh-nnfwi-author-20261012/case{case}/final_fields.npz')['reconstruction']
    tv = np.load(OUT / f'k{best:g}/case{case}/final_fields.npz')['reconstruction']
    arrays = [truth, conv['reconstruction'].numpy().squeeze(), nn, tv]
    errors = [None, reference[row]['conventional']['material_mse'], reference[row]['nn']['material_mse'], complete[best][row]['material_mse']]
    tv_title = f'TV oracle common k={best:g}' if grid_complete else f'TV k={best:g} (grid incomplete)'
    titles = ['Truth', 'Conventional FWI', 'Author NN FWI', tv_title]
    for col in range(4):
        im = axes[row, col].imshow(arrays[col][1:-1, 1:-1].T, vmin=0, vmax=1, cmap='coolwarm')
        title = f'Case {case}: {titles[col]}'
        if errors[col] is not None:
            title += f'\nMSE {errors[col]:.6f}'
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
fig.subplots_adjust(right=.94, hspace=.3, wspace=.09)
fig.colorbar(im, cax=fig.add_axes([.95, .15, .012, .7]), label='Material gamma')
fig.savefig(OUT / f'four_case_three_methods_best_k{best:g}.png', dpi=180, bbox_inches='tight')
plt.close(fig)
# A common-scale enlargement supports the requested visual check of the thin crack.
case = 3
conv = torch.load(ROOT / '.build/author-singh-case3-35epochs-20261008/fields.pt', weights_only=True)
arrays = [conv['truth'].numpy().squeeze(), conv['reconstruction'].numpy().squeeze(),
    np.load(ROOT / '.build/singh-nnfwi-author-20261012/case3/final_fields.npz')['reconstruction']]
arrays += [np.load(OUT / f'k{k:g}/case3/final_fields.npz')['reconstruction'] for k in ordered_k]
fig, axes = plt.subplots(1, len(arrays), figsize=(3.5 * len(arrays), 3))
for ax, array, title in zip(axes, arrays, ['Truth', 'Conventional', 'NN'] + [f'TV k={k:g}' for k in ordered_k]):
    ax.imshow(array[1:-1, 1:-1].T, vmin=0, vmax=1, cmap='coolwarm', interpolation='nearest')
    ax.set_xlim(0, 85)
    ax.set_ylim(55, 0)
    ax.set_title(title)
fig.suptitle('Case 3: thin-crack region; same color scale 0 to 1')
fig.tight_layout()
fig.savefig(OUT / 'case3_thin_crack_zoom_tv_weights.png', dpi=180)
plt.close(fig)
print(json.dumps(summary, indent=2))

"""Plot all nine fixed-geometry contrasts without interpolation of minima."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('.build/author-reduced-material-20261008')
report = json.loads((out/'summary.json').read_text())
records = report['records']
x = [r['contrast'] for r in records]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for name, label in [('released', 'Released observations'), ('self_consistent', 'Same-stencil observations')]:
    axes[0, 0].plot(x, [r[name]['cost']*1e12 for r in records], 'o-', label=label)
    axes[0, 1].plot(x, [r[name]['waveform_relative_l2']*100 for r in records], 'o-', label=label)
axes[0, 0].set(title='Correct time cost, all nine points', ylabel='Cost x 1e12')
axes[0, 1].set(title='Waveform error against each target', ylabel='Relative L2 (%)')
axes[1, 0].plot(x, [r['physical_material_half_mse'] for r in records], 'o-', color='tab:green')
axes[1, 0].set(title='Representation distance to truth', ylabel='Physical-grid half MSE')
for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.set(xscale='log', xlabel='Defect contrast a (known support)')
    ax.grid(alpha=.25)
axes[0, 0].legend(fontsize=9)
axes[0, 1].legend(fontsize=9)
axes[1, 1].axis('off')
lines = ['Case1: known geometry, no optimization', '4 shots, 24 receivers, 2000 time steps',
         'Nine contrasts fixed before evaluation', 'Segments connect sampled values only',
         'No claim of continuous/global optimum', '']
for name in ['released', 'self_consistent']:
    a = report['best_sampled'][name]
    r = next(r for r in records if r['contrast'] == a)
    lines += [name, f"sampled min a={a:g}; waveform={r[name]['waveform_relative_l2']*100:.5f}%"]
lines += ['', f"True defect value: {report['truth_defect_values'][0]:.8g}",
          f"Support: {report['support_physical_cells']} physical cells",
          'Same-grid consistency is an internal check.', 'Independent physics remains unverified.']
axes[1, 1].text(0, 1, '\n'.join(lines), va='top', fontsize=10)
fig.savefig(out/'reduced_material_scan.png', dpi=180)
plt.close(fig)

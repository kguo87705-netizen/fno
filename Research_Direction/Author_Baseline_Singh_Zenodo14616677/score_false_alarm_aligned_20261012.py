"""Threshold scan of existing fields only; no inversion or forward solver."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.build/singh-tv-author-20261012'
METHODS = ['conventional', 'nn', 'tv_k1']
DISPLAY = {'conventional': 'Conventional FWI', 'nn': 'NN', 'tv_k1': 'TV k=1'}
THRESHOLDS = np.arange(2, 19) / 20
NAMES = {
    1: ['single_ellipse'], 2: ['rectangle', 'circle_d8'],
    3: ['left_ellipse', 'thin_slit', 'long_ellipse'],
    4: ['left_circle_d6', 'cross', 'central_circle_d4', 'central_circle_d3', 'central_circle_d2', 'right_circle_d4']}
DX, DY = 100 / 255, 50 / 127
cases = {}
for c in range(1, 5):
    saved = torch.load(ROOT / f'.build/author-singh-case{c}-35epochs-20261008/fields.pt', weights_only=True)
    truth = saved['truth'].numpy().squeeze()[1:-1, 1:-1].astype(float)
    fields = {'conventional': saved['reconstruction'].numpy().squeeze()[1:-1, 1:-1].astype(float)}
    for m, path in [('nn', ROOT / f'.build/singh-nnfwi-author-20261012/case{c}/final_fields.npz'),
                    ('tv_k1', OUT / f'k1/case{c}/final_fields.npz')]:
        a = np.load(path)
        assert np.array_equal(truth, a['truth'][1:-1, 1:-1])
        fields[m] = a['reconstruction'][1:-1, 1:-1].astype(float)
    defect = truth < .5
    labels, n = ndimage.label(defect, structure=np.ones((3, 3)))
    assert n == len(NAMES[c])
    background = ~ndimage.binary_dilation(defect, structure=np.ones((3, 3)), iterations=2)
    contrast = abs(float(np.unique(truth[defect]).item()) - 1)
    cases[c] = dict(truth=truth, fields=fields, defect=defect, labels=labels,
                    background=background, contrast=contrast)

result = dict(convention='Owner-defined, not an industry standard',
    region='physical 256x128, ghost cells excluded', connectivity=8,
    background='Chebyshev distance greater than 2 pixels from truth defects',
    selection='Maximum detected count under pooled FPR cap; ties: lowest FPR, then highest threshold',
    defect_recall_cutoff=.5, scan={}, working_points={})
for m in METHODS:
    rows = []
    for t in THRESHOLDS:
        false_pixels, bg_pixels = 0, 0
        counts, per_case, misses = [], {}, []
        for c, a in cases.items():
            abnormal = abs(a['fields'][m] - 1) >= t * a['contrast']
            recalls = [float(abnormal[a['labels'] == i].mean()) for i in range(1, len(NAMES[c]) + 1)]
            detected = [x >= .5 for x in recalls]
            fp = int((abnormal & a['background']).sum())
            nb = int(a['background'].sum())
            false_pixels += fp
            bg_pixels += nb
            counts.append(sum(detected))
            misses.extend(f'case{c}:{name}' for name, hit in zip(NAMES[c], detected) if not hit)
            per_case[str(c)] = dict(recalls=recalls, detected=detected, false_alarm_fraction=fp / nb)
        rows.append(dict(threshold=float(t), false_alarm_fraction=false_pixels / bg_pixels,
            detected_total=sum(counts), detected_per_case=counts, total_defects=12,
            thin_slit_recall=per_case['3']['recalls'][1], missed_defects=misses, cases=per_case))
    assert len(rows) == 17
    assert all(a['detected_total'] >= b['detected_total'] and
               a['false_alarm_fraction'] >= b['false_alarm_fraction'] and
               a['thin_slit_recall'] >= b['thin_slit_recall'] for a, b in zip(rows, rows[1:]))
    result['scan'][m] = rows
    result['working_points'][m] = {}
    for cap in [.01, .005]:
        eligible = [r for r in rows if r['false_alarm_fraction'] <= cap]
        best = min(eligible, key=lambda r: (-r['detected_total'], r['false_alarm_fraction'], -r['threshold'])) if eligible else None
        result['working_points'][m][str(cap)] = best
        assert best is None or best['false_alarm_fraction'] <= cap

# Cross-check the reused definition against the preceding saved evaluation.
previous = json.loads((OUT / 'defect_detection_scores.json').read_text(encoding='utf-8'))
for m in METHODS:
    for t in [.3, .5, .7]:
        row = next(x for x in result['scan'][m] if np.isclose(x['threshold'], t))
        old = previous['aggregate'][m][str(t)]
        assert row['detected_total'] == old['detected_total']
        assert row['false_alarm_fraction'] == old['pooled_false_alarm_fraction']
        assert row['thin_slit_recall'] == old['thin_slit_recall']

fig, ax = plt.subplots(figsize=(8, 5))
for m in METHODS:
    rows = result['scan'][m]
    ax.plot([100*r['false_alarm_fraction'] for r in rows], [r['detected_total'] for r in rows], 'o-', label=DISPLAY[m], markersize=4)
    for cap, marker in [(.01, 's'), (.005, 'D')]:
        row = result['working_points'][m][str(cap)]
        if row:
            ax.scatter(100*row['false_alarm_fraction'], row['detected_total'], s=100, marker=marker, facecolors='none', edgecolors='black', zorder=4)
ax.axvline(1, color='gray', linestyle='--', linewidth=1)
ax.axvline(.5, color='gray', linestyle=':', linewidth=1)
ax.set_xscale('symlog', linthresh=.1)
ax.set_xticks([0, .1, .5, 1, 5, 10, 25, 50])
ax.set_xticklabels(['0', '0.1', '0.5', '1', '5', '10', '25', '50'])
ax.set_xlabel('Pooled background false-alarm area (%) [symlog scale]')
ax.set_ylabel('Detected defects / 12')
ax.set_yticks(range(13))
ax.set_ylim(-.3, 12.4)
ax.grid(alpha=.25)
ax.legend()
ax.set_title('Saved results: threshold 0.10 to 0.90, step 0.05\nSquares: 1% cap; diamonds: 0.5% cap')
fig.tight_layout()
fig.savefig(OUT / 'false_alarm_aligned_detection_curve.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(3, 4, figsize=(15, 7.7), sharex=True, sharey=True)
extent = (-DX/2, 100+DX/2, 50+DY/2, -DY/2)
for row, m in enumerate(METHODS):
    best = result['working_points'][m]['0.01']
    for col, (c, a) in enumerate(cases.items()):
        ax = axes[row, col]
        abnormal = abs(a['fields'][m]-1) >= best['threshold']*a['contrast']
        ax.imshow(abnormal.T, cmap='gray_r', vmin=0, vmax=1, extent=extent, interpolation='nearest')
        ax.contour(np.arange(256)*DX, np.arange(128)*DY, a['defect'].T, levels=[.5], colors='red', linewidths=.65)
        score = best['cases'][str(c)]
        ax.set_title(f'Case {c}: {sum(score["detected"])}/{len(NAMES[c])}, local FPR {score["false_alarm_fraction"]:.2%}', fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(50, 0)
        if col == 0:
            ax.set_ylabel(f'{DISPLAY[m]}\nt={best["threshold"]:.2f}; pooled FPR={best["false_alarm_fraction"]:.2%}\ny (mm)', fontsize=9)
        if row == 2:
            ax.set_xlabel('x (mm)')
fig.suptitle('1% pooled false-alarm cap: black = abnormal pixels; red = ground-truth boundaries', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(OUT / 'false_alarm_1pct_binary_four_cases.png', dpi=180)
plt.close(fig)
(OUT / 'false_alarm_aligned_scores.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({m: {cap: {k:v for k,v in row.items() if k != 'cases'} if row else None
    for cap,row in points.items()} for m,points in result['working_points'].items()}, indent=2))

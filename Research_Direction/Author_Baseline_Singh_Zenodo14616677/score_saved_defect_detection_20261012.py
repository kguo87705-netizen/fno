"""Score saved reconstructions using the owner's detection convention; no solver."""
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
METHODS = ['conventional', 'nn', 'tv_k1', 'tv_k10']
THRESHOLDS = [.3, .5, .7]
EXPECTED = [1, 2, 3, 6]
DX, DY = 100 / 255, 50 / 127
result = dict(convention='Project-specific, not an industry standard',
    connectivity=8, background_distance='Chebyshev grid distance > 2',
    recall_detection_cutoff=.5, pixel_thresholds=THRESHOLDS, cases={}, aggregate={})
case_names = {1: ['single_ellipse'], 2: ['rectangle', 'circle_d8'],
    3: ['left_ellipse', 'thin_slit', 'long_ellipse'],
    4: ['left_circle_d6', 'cross', 'central_circle_d4', 'central_circle_d3', 'central_circle_d2', 'right_circle_d4']}
for case in range(1, 5):
    old = torch.load(ROOT / f'.build/author-singh-case{case}-35epochs-20261008/fields.pt', weights_only=True)
    truth = old['truth'].numpy().squeeze()[1:-1, 1:-1].astype(np.float64)
    fields = {'conventional': old['reconstruction'].numpy().squeeze()[1:-1, 1:-1].astype(np.float64)}
    locations = dict(nn=ROOT / f'.build/singh-nnfwi-author-20261012/case{case}/final_fields.npz',
        tv_k1=OUT / f'k1/case{case}/final_fields.npz', tv_k10=OUT / f'k10/case{case}/final_fields.npz')
    for method, path in locations.items():
        saved = np.load(path)
        assert np.array_equal(truth, saved['truth'][1:-1, 1:-1])
        fields[method] = saved['reconstruction'][1:-1, 1:-1].astype(np.float64)
    defect = truth < .5
    labels, count = ndimage.label(defect, structure=np.ones((3, 3)))
    assert count == EXPECTED[case - 1]
    background = ~ndimage.binary_dilation(defect, structure=np.ones((3, 3)), iterations=2)
    gamma_background = float(np.unique(truth[~defect]).item())
    gamma_defect = float(np.unique(truth[defect]).item())
    contrast = abs(gamma_defect - gamma_background)
    defects = []
    for label, name in enumerate(case_names[case], 1):
        coordinates = np.argwhere(labels == label)
        defects.append(dict(label=label, name=name, pixels=int(len(coordinates)),
            centroid_mm=(coordinates.mean(axis=0) * [DX, DY]).tolist(),
            bbox_grid=(np.ptp(coordinates, axis=0) + 1).tolist()))
    case_result = dict(total_defects=count, background_pixels=int(background.sum()),
        gamma_background=gamma_background, gamma_defect=gamma_defect, defects=defects, scores={})
    for method, field in fields.items():
        case_result['scores'][method] = {}
        for threshold in THRESHOLDS:
            abnormal = np.abs(field - gamma_background) >= threshold * contrast
            recalls = [float(abnormal[labels == label].mean()) for label in range(1, count + 1)]
            detected = [r >= .5 for r in recalls]
            false_pixels = int(np.count_nonzero(abnormal & background))
            case_result['scores'][method][str(threshold)] = dict(recalls=recalls, detected=detected,
                detected_count=int(sum(detected)), false_alarm_pixels=false_pixels,
                false_alarm_fraction=false_pixels / int(background.sum()))
        # Increasing the contrast threshold must not increase recall or false alarms.
        for lower, upper in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
            ls, us = case_result['scores'][method][str(lower)], case_result['scores'][method][str(upper)]
            assert np.all(np.array(ls['recalls']) >= np.array(us['recalls']))
            assert ls['false_alarm_pixels'] >= us['false_alarm_pixels']
    result['cases'][str(case)] = case_result
    if case in (3, 4):
        fig, axes = plt.subplots(1, 5, figsize=(19, 4))
        extent = (-DX / 2, 100 + DX / 2, 50 + DY / 2, -DY / 2)
        axes[0].imshow(np.ma.masked_where(labels.T == 0, labels.T), cmap='tab10', vmin=0, vmax=10, extent=extent, interpolation='nearest')
        axes[0].set_title(f'Case {case}: truth defect labels')
        for d in defects:
            axes[0].text(*d['centroid_mm'], str(d['label']), color='black', fontsize=10, ha='center', va='center')
        for ax, method in zip(axes[1:], METHODS):
            abnormal = np.abs(fields[method] - gamma_background) >= .5 * contrast
            ax.imshow(abnormal.T, cmap='gray_r', vmin=0, vmax=1, extent=extent, interpolation='nearest')
            ax.contour(np.arange(256) * DX, np.arange(128) * DY, defect.T, levels=[.5], colors='red', linewidths=.5)
            score = case_result['scores'][method]['0.5']
            ax.set_title(f'{method}: {score["detected_count"]}/{count}\nFPR {score["false_alarm_fraction"]:.2%}')
        for ax in axes:
            ax.set_xlim(0, 100)
            ax.set_ylim(50, 0)
            ax.set_xlabel('x (mm)')
        axes[0].set_ylabel('y (mm)')
        fig.tight_layout()
        fig.savefig(OUT / f'case{case}_defect_detection_t05.png', dpi=170)
        plt.close(fig)

for method in METHODS:
    result['aggregate'][method] = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        scores = [result['cases'][str(c)]['scores'][method][key] for c in range(1, 5)]
        false_pixels = sum(s['false_alarm_pixels'] for s in scores)
        total_background = sum(result['cases'][str(c)]['background_pixels'] for c in range(1, 5))
        result['aggregate'][method][key] = dict(
            detected_per_case=[s['detected_count'] for s in scores],
            detected_total=sum(s['detected_count'] for s in scores), total_defects=sum(EXPECTED),
            false_alarm_fraction_per_case=[s['false_alarm_fraction'] for s in scores],
            pooled_false_alarm_fraction=false_pixels / total_background,
            thin_slit_recall=result['cases']['3']['scores'][method][key]['recalls'][1],
            case4_detected_defects=[d['name'] for d, hit in zip(result['cases']['4']['defects'], scores[3]['detected']) if hit])
(OUT / 'defect_detection_scores.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(aggregate=result['aggregate'], case4_per_defect={
    method: {str(t): result['cases']['4']['scores'][method][str(t)]['recalls'] for t in THRESHOLDS}
    for method in METHODS}), indent=2))

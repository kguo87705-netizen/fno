"""Read saved fields only: regional errors, geometry and crack sampling scale."""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.build/singh-tv-author-20261012'
DX, DY = 100 / 255, 50 / 127  # millimetres
SPACING = np.array([DX, DY])
results = {'spacing_mm': SPACING.tolist(), 'wavelength_mm': 6000 / 500000 * 1000,
           'region_definition': 'truth<0.5; exterior band=two 8-neighbour dilations excluding defect; remainder=background',
           'cases': {}}

def geometry(mask):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    items = []
    for label in range(1, count + 1):
        region = labels == label
        indices = np.argwhere(region)
        points = indices * SPACING
        mean = points.mean(axis=0)
        cov = np.cov(points.T) if len(points) > 1 else np.zeros((2, 2))
        values, vectors = np.linalg.eigh(cov)
        major = vectors[:, -1]
        minor = vectors[:, 0]
        lengths = np.ptp(points @ vectors, axis=0) + np.abs(vectors).T @ SPACING
        area = len(points) * DX * DY
        thicknesses, frequencies = np.unique(region.sum(axis=1)[region.any(axis=1)], return_counts=True)
        item = dict(label=label, pixels=len(points), area_mm2=float(area),
            centroid_mm=mean.tolist(), bbox_grid=(np.ptp(indices, axis=0) + 1).tolist(),
            major_extent_mm=float(lengths[1]), minor_extent_mm=float(lengths[0]),
            effective_area_over_length_width_mm=float(area / lengths[1]),
            occupied_y_counts_per_x=thicknesses.tolist(),
            occupied_y_counts_frequencies={str(int(t)): int(n) for t, n in zip(thicknesses, frequencies)})
        items.append(item)
    return items

for case in range(1, 5):
    saved = torch.load(ROOT / f'.build/author-singh-case{case}-35epochs-20261008/fields.pt', weights_only=True)
    truth = saved['truth'].numpy().squeeze()[1:-1, 1:-1].astype(float)
    tv_saved = np.load(OUT / f'k1/case{case}/final_fields.npz')
    nn_saved = np.load(ROOT / f'.build/singh-nnfwi-author-20261012/case{case}/final_fields.npz')
    assert np.array_equal(truth, tv_saved['truth'][1:-1, 1:-1])
    assert np.array_equal(truth, nn_saved['truth'][1:-1, 1:-1])
    tv = tv_saved['reconstruction'][1:-1, 1:-1].astype(float)
    nn = nn_saved['reconstruction'][1:-1, 1:-1].astype(float)
    assert truth.shape == (256, 128)
    assert len(np.unique(truth)) == 2
    defect = truth < .5
    band = ndimage.binary_dilation(defect, structure=np.ones((3, 3)), iterations=2) & ~defect
    background = ~(defect | band)
    assert np.all(defect.astype(int) + band + background == 1)
    errors = {'tv': (tv - truth) ** 2, 'nn': (nn - truth) ** 2}
    mse = {name: float(array.mean()) for name, array in errors.items()}
    excess = mse['tv'] - mse['nn']
    data = dict(geometry=geometry(defect), truth_values=np.unique(truth).tolist(),
                total_mse=mse, excess_mse=excess, regions={})
    for name, region in zip(['defect', 'outer_two_grid_band', 'background'], [defect, band, background]):
        contribution = {method: float(error[region].sum() / truth.size) for method, error in errors.items()}
        data['regions'][name] = dict(pixels=int(region.sum()),
            fraction_of_grid=float(region.mean()), mse_contribution=contribution,
            share_of_method_mse_percent={method: value / mse[method] * 100 for method, value in contribution.items()},
            excess_mse_contribution=contribution['tv'] - contribution['nn'],
            share_of_excess_percent=(contribution['tv'] - contribution['nn']) / excess * 100)
    for method in errors:
        assert np.isclose(sum(r['mse_contribution'][method] for r in data['regions'].values()), mse[method])
    assert np.isclose(sum(r['share_of_excess_percent'] for r in data['regions'].values()), 100)
    results['cases'][str(case)] = data
    if case in (2, 4):
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
        fields = [truth, nn, tv, np.abs(tv - truth), np.abs(nn - truth)]
        titles = [f'Case {case}: truth', 'NN', 'TV k=1', '|TV - truth|', '|NN - truth|']
        for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
            im = ax.imshow(field.T, vmin=0, vmax=1, cmap='coolwarm' if i < 3 else 'magma',
                           extent=(-DX / 2, 100 + DX / 2, 50 + DY / 2, -DY / 2), interpolation='nearest')
            if i >= 3:
                ax.contour(np.arange(256) * DX, np.arange(128) * DY, defect.T,
                           levels=[.5], colors='cyan', linewidths=.45)
            ax.set_xlim(0, 100)
            ax.set_ylim(50, 0)
            ax.set_title(title)
            ax.set_xlabel('x (mm)')
        axes[0].set_ylabel('y (mm)')
        fig.subplots_adjust(right=.94, wspace=.27)
        fig.colorbar(im, cax=fig.add_axes([.95, .22, .008, .57]), label='Absolute error (common scale 0 to 1)')
        fig.savefig(OUT / f'case{case}_truth_nn_tv_errors.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        region_field = np.where(defect, 0, np.where(band, 1, 2))
        ax.imshow(region_field.T, cmap=ListedColormap(['#2468b4', '#f2ac45', '#eee']),
            vmin=0, vmax=2, interpolation='nearest', extent=(0, 100, 50, 0))
        ax.set_title(f'Case {case}: truth regions: blue=defect, orange=exterior 2-grid band')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        fig.tight_layout()
        fig.savefig(OUT / f'case{case}_truth_regions.png', dpi=180)
        plt.close(fig)

(OUT / 'regional_error_geometry.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))

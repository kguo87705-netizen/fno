"""M4d-E stage 1, items 1-3: input dependence, metric checks, prediction statistics.

Uses the M4c local checkpoint (no training). Dice threshold is fixed at 5000 m/s;
the threshold sweep is diagnostic only. The mean-image reference uses training labels only.

Conditions on the validation split:
    normal   - the real scattering FMC
    permuted - the same inputs randomly reassigned to other targets
    zero     - all-zero scattering field (= defect-free, ideal output is uniform 5800 m/s)
The training split gets normal and permuted.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

CODE = r'E:\FNO_research\repos\fno_light\repro_lu2026\code'
sys.path.insert(0, CODE)
import m3_data as m3
import m4_fno as m4
import m4b_diagnostics as diag

BASE = r'E:\FNO_research\repos\fno_light\repro_lu2026'
CFG = BASE + r'\configs\m4b_fno.json'
LOCAL_RUN = Path(r'E:\FNO_research\results\m4c_local_20261020_170451')
OUT = BASE + r'\configs\m4d_E_diagnostics.json'
FIG = Path(BASE) / 'reports'


def build_cfg():
    cfg = m3.cfg_load(CFG)
    cfg['model'] = dict(cfg['model'])
    cfg['model']['mapping'] = 'local'
    return cfg


def predict(model, x, dev, batch=32):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            outs.append(model(x[i:i + batch].to(dev)).float().cpu())
    return torch.cat(outs)


def dice_at(pred_ms, true_ms, thr):
    a = pred_ms < thr
    b = true_ms < thr
    inter = (a & b).flatten(1).sum(1).double()
    na = a.flatten(1).sum(1).double()
    nb = b.flatten(1).sum(1).double()
    return (2 * inter / (na + nb).clamp(min=1))


def main():
    cfg = build_cfg()
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    win = m4.gaussian_window(11, 1.5, torch.device('cpu'))
    ck = torch.load(LOCAL_RUN / 'best.pt', map_location=dev, weights_only=False)
    model = m4.FNO2d(cfg['model']).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()

    # Training labels only, for the mean-image reference.
    maps = []
    for r in splits['train']:
        with h5py.File(m3.sample_path(cfg, r['sample_id']), 'r') as f:
            maps.append(f['c_200'][:])
    mean_map = np.mean(np.stack(maps), axis=0).astype(np.float32)

    res = {'checkpoint': str(LOCAL_RUN / 'best.pt'), 'dice_threshold_ms': 5000.0, 'splits': {}}
    for split in ['train', 'validation']:
        ds = m3.M2Dataset(cfg, splits[split], norm)
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        t = torch.stack([ds[i][1] for i in range(len(ds))])
        g = torch.Generator().manual_seed(cfg['seed'])
        perm = torch.randperm(x.shape[0], generator=g)
        p_n = predict(model, x, dev)
        p_perm = predict(model, x[perm], dev)
        tm = m4.to_ms(t, norm)
        entry = {}
        base = diag.suite(p_n, t, norm, win)
        entry['normal'] = {
            'dice': base['dice'], 'iou': base['iou'],
            'defect_mae_ms': base['defect_mae_ms'], 'overall_mae_ms': base['mae_ms'],
            'relative_l2_rms_of_samples': base['relative_l2_rms_of_samples'],
            'pred_cross_sample_std_ms': float(m4.to_ms(p_n, norm).std(dim=0).mean()),
            'true_cross_sample_std_ms': float(tm.std(dim=0).mean()),
        }
        pm_n = m4.to_ms(p_n, norm)
        pm_perm = m4.to_ms(p_perm, norm)
        entry['permuted'] = {
            'dice': float(dice_at(pm_perm, tm, 5000.0).mean()),
            'defect_mae_ms': float((pm_perm - tm).abs()[tm < 5000.0].mean()),
            'mean_abs_diff_vs_normal_ms': float((pm_perm - pm_n).abs().mean()),
            'pred_cross_sample_std_ms': float(pm_perm.std(dim=0).mean()),
        }
        if split == 'validation':
            zero_x = torch.full_like(x, float(norm['fmc']['offset']))
            p_zero = predict(model, zero_x, dev)
            pm_zero = m4.to_ms(p_zero, norm)
            entry['zero'] = {
                'dice': float(dice_at(pm_zero, tm, 5000.0).mean()),
                'defect_mae_ms': float((pm_zero - tm).abs()[tm < 5000.0].mean()),
                'mean_abs_diff_vs_normal_ms': float((pm_zero - pm_n).abs().mean()),
                'pred_cross_sample_std_ms': float(pm_zero.std(dim=0).mean()),
                'mean_predicted_speed_ms': float(pm_zero.mean()),
                'min_predicted_speed_ms': float(pm_zero.min()),
            }
            # Threshold sweep (diagnostic only) and reference statistics.
            sweep = {}
            for thr in [3000.0, 4000.0, 5000.0, 5500.0]:
                sweep[str(int(thr))] = {
                    'model_dice': float(dice_at(pm_n, tm, thr).mean()),
                    'mean_image_dice': float(dice_at(torch.tensor(np.repeat(mean_map[None, None], tm.shape[0], 0)),
                                                     tm, thr).mean()),
                }
            entry['threshold_sweep'] = sweep
            d = dice_at(pm_n, tm, 5000.0)
            d_ref = dice_at(torch.tensor(np.repeat(mean_map[None, None], tm.shape[0], 0)), tm, 5000.0)
            gain = d - d_ref
            entry['dice_stats'] = {
                'mean': float(d.mean()), 'median': float(d.median()),
                'std': float(d.std(unbiased=True)), 'min': float(d.min()), 'max': float(d.max()),
                'mean_image_mean': float(d_ref.mean()),
                'gain_over_mean_image_mean': float(gain.mean()),
                'gain_over_mean_image_median': float(gain.median()),
                'fraction_positive_gain': float((gain > 0).float().mean()),
            }
            entry['prediction_histogram'] = {
                'bins': np.linspace(1500.0, 5800.0, 44).round(1).tolist(),
                'counts': np.histogram(pm_n.numpy(), bins=np.linspace(1500.0, 5800.0, 44))[0].tolist(),
                'mean': float(pm_n.mean()), 'std': float(pm_n.std()),
            }
        res['splits'][split] = entry
        print(split, json.dumps(entry, indent=1)[:1500], flush=True)

    with Path(OUT).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('wrote', OUT)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h = res['splits']['validation']['prediction_histogram']
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6))
    ax.hist(np.repeat(h['bins'][:-1], h['counts']), bins=h['bins'], color='tab:blue', alpha=0.8)
    ax.set(xlabel='predicted speed (m/s)', ylabel='pixels', title='E-local predictions on the 40 validation samples')
    fig.tight_layout()
    p = FIG / 'm4d_E_pred_hist.png'
    if p.exists():
        raise SystemExit(f'refusing to overwrite {p}')
    fig.savefig(p, dpi=120)
    print('wrote', p)


if __name__ == '__main__':
    main()

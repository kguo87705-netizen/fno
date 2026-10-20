"""M4b diagnostics: separate "not enough iterations" from "not enough data".

Evaluates the saved M4 and M4b best checkpoints on the training and validation splits
(never the test split), adds defect-region metrics, compares against two learned-free
references, and overlays the two loss histories.

Usage:
    python m4b_diagnostics.py <m4_run_dir> <m4b_run_dir>
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

CODE = r'E:\FNO_research\repos\fno_light\repro_lu2026\code'
sys.path.insert(0, CODE)
import m3_data as m3
import m4_fno as m4

CFG = r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4b_fno.json'
DIAG_JSON = r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4b_diagnostics.json'
FIG = Path(r'E:\FNO_research\repos\fno_light\repro_lu2026\reports')
VAL_FIG_IDS = ['m2_000378', 'm2_000401', 'm2_000405']
TRAIN_FIG_IDS = ['m2_000000', 'm2_000001', 'm2_000002']
DEFECT_MS = 5000.0


def collect(model, loader, dev):
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(dev, non_blocking=True))
            preds.append(p.float().cpu())
            trues.append(y.float())
    return torch.cat(preds), torch.cat(trues)


def suite(pred_n, true_n, norm, ssim_win):
    """Overall and defect-region metrics, all in m/s."""
    pred_n = pred_n.float().cpu()
    true_n = true_n.float().cpu()
    pm = m4.to_ms(pred_n, norm)
    tm = m4.to_ms(true_n, norm)
    diff = pm - tm
    rel = (diff.flatten(1).norm(dim=1) / tm.flatten(1).norm(dim=1))
    ssim_all = m4.ssim(pm, tm, ssim_win, 4300.0)
    mask_t = tm < DEFECT_MS
    pred_bin = pm < DEFECT_MS
    inter = (pred_bin & mask_t).flatten(1).sum(dim=1).double()
    nt = mask_t.flatten(1).sum(dim=1).double()
    npb = pred_bin.flatten(1).sum(dim=1).double()
    dice = 2 * inter / (nt + npb).clamp(min=1)
    iou = inter / (nt + npb - inter).clamp(min=1)
    mae_mask = []
    for i in range(pm.shape[0]):
        m = mask_t[i]
        mae_mask.append(float(diff[i][m].abs().mean()) if int(m.sum()) else float('nan'))
    return {
        'relative_l2_rms_of_samples': float(np.sqrt(np.mean(rel.numpy() ** 2))),
        'relative_l2_mean_of_samples': float(rel.mean()),
        'mae_ms': float(diff.abs().mean()),
        'rmse_ms': float((diff.flatten(1) ** 2).mean(dim=1).sqrt().mean()),
        'ssim': float(ssim_all),
        'defect_pixels_true': float(nt.mean()),
        'defect_pixels_pred': float(npb.mean()),
        'defect_mae_ms': float(np.nanmean(mae_mask)),
        'dice': float(dice.mean()),
        'iou': float(iou.mean()),
        'dice_per_sample': dice.tolist(),
        'IoU_note': 'IoU = mean over samples of |A and B| / |A or B| with threshold 5000 m/s',
    }


def ids_for(rows):
    return [r['sample_id'] for r in rows]


def main():
    m4_run = Path(sys.argv[1])
    m4b_run = Path(sys.argv[2])
    cfg = m3.cfg_load(CFG)
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg['train']['tf32'])
    win = m4.gaussian_window(cfg['eval']['ssim_window'], cfg['eval']['ssim_sigma'], torch.device('cpu'))

    train_loader = m3.make_loader(cfg, splits['train'], norm, False, 32, num_workers=4)
    val_loader = m3.make_loader(cfg, splits['validation'], norm, False, 32, num_workers=4)

    # Learned-free references.
    maps = []
    for r in splits['train']:
        with h5py.File(m3.sample_path(cfg, r['sample_id']), 'r') as f:
            maps.append(f['c_200'][:])
    mean_map = np.mean(np.stack(maps), axis=0).astype(np.float32)
    mean_n = torch.tensor(mean_map * norm['speed']['scale']
                          + norm['speed']['offset'])[None, None]

    out = {'defect_threshold_ms': DEFECT_MS, 'splits': {}, 'reference': {}}
    model_preds = {}
    for tag, run_dir in [('m4', m4_run), ('m4b', m4b_run)]:
        ck = torch.load(run_dir / 'best.pt', map_location=dev, weights_only=False)
        model = m4.FNO2d(cfg['model']).to(dev)
        model.load_state_dict(ck['model'])
        model.eval()
        entry = {'run_dir': str(run_dir), 'best_epoch': int(ck['epoch']),
                 'best_val_l1': float(ck['val_l1'])}
        for split, loader in [('train', train_loader), ('validation', val_loader)]:
            p, t = collect(model, loader, dev)
            entry[split] = suite(p, t, norm, win)
            if split == 'validation':
                model_preds[(tag, 'validation')] = p
            else:
                model_preds[(tag, 'train')] = p
            if split == 'validation':
                val_true = t
            else:
                train_true = t
        out['splits'][tag] = entry
        print(tag, 'best epoch', entry['best_epoch'], 'val dice', round(entry['validation']['dice'], 4),
              'train dice', round(entry['train']['dice'], 4), flush=True)

    # References: background constant, training-mean image.
    refs = {}
    for split, true_n in [('train', train_true), ('validation', val_true)]:
        const_n = torch.full_like(true_n, norm['speed']['offset'] + cfg['speed']['max_ms'] * norm['speed']['scale'])
        refs[f'constant_background_{split}'] = suite(const_n, true_n, norm, win)
        refs[f'training_mean_image_{split}'] = suite(mean_n.expand_as(true_n), true_n, norm, win)
    out['reference'] = refs

    # Loss histories, so the report can state whether validation was still improving.
    hist = {}
    for tag, run_dir in [('m4', m4_run), ('m4b', m4b_run)]:
        log = run_dir / 'm4_train_log.jsonl'
        recs = [json.loads(x) for x in log.read_text(encoding='utf-8').splitlines() if x.strip()]
        hist[tag] = recs
    for tag in ['m4', 'm4b']:
        v = [r['val_l1'] for r in hist[tag]]
        last = v[-50:] if len(v) > 50 else v
        out.setdefault('trend', {})[tag] = {
            'epochs': len(v),
            'val_l1_first': v[0], 'val_l1_last': v[-1], 'val_l1_best': min(v),
            'best_epoch': int(np.argmin(v)) + 1,
            'val_l1_mean_last_25': float(np.mean(v[-25:])),
            'val_l1_mean_prev_25': float(np.mean(v[-50:-25])) if len(v) > 50 else None,
            'still_improving': bool(len(v) > 50 and np.mean(v[-25:]) < np.mean(v[-50:-25])),
        }

    # Curves: M4 against M4b.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for tag, color in [('m4', 'tab:blue'), ('m4b', 'tab:orange')]:
        ep = [r['epoch'] for r in hist[tag]]
        ax.plot(ep, [r['train_l1'] for r in hist[tag]], color=color, lw=1,
                label=f'{tag} train (best val {min(r["val_l1"] for r in hist[tag]):.4f})')
        ax.plot(ep, [r['val_l1'] for r in hist[tag]], color=color, lw=1, ls='--', label=f'{tag} validation')
    ax.set(xlabel='epoch', ylabel='L1 on normalized speed', yscale='log',
           title='M4 (early stop) vs M4b (fixed 500 epochs, no early stop)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / 'm4b_curves_compare.png'
    if p.exists():
        raise SystemExit(f'refusing to overwrite {p}')
    fig.savefig(p, dpi=cfg['preview']['dpi'])
    plt.close(fig)
    print('wrote', p)

    # Sample figures: the same 3 validation ids as the M4 report, plus 3 training ids.
    val_ids = ids_for(splits['validation'])
    train_ids = ids_for(splits['train'])
    for split, wanted, pred_key in [('validation', VAL_FIG_IDS, 'validation'),
                                    ('train', TRAIN_FIG_IDS, 'train')]:
        pm_all = model_preds[('m4b', pred_key)]
        tm_all = val_true if split == 'validation' else train_true
        index = val_ids if split == 'validation' else train_ids
        for sid in wanted:
            if sid not in index:
                print('skip missing id', sid)
                continue
            i = index.index(sid)
            gt = m4.to_ms(tm_all[i, 0], norm).numpy()
            pr = m4.to_ms(pm_all[i, 0], norm).numpy()
            err = pr - gt
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
            im = axes[0].imshow(gt, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
            fig.colorbar(im, ax=axes[0], label='m/s')
            axes[0].set(title=f'{sid} truth', xlabel='Lateral (mm)', ylabel='Depth (mm)')
            im = axes[1].imshow(pr, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
            fig.colorbar(im, ax=axes[1], label='m/s')
            axes[1].set(title='M4b prediction', xlabel='Lateral (mm)')
            lim = float(np.abs(err).max())
            im = axes[2].imshow(err, extent=[0, 40, 40, 0], vmin=-lim, vmax=lim, cmap='coolwarm')
            fig.colorbar(im, ax=axes[2], label='m/s')
            axes[2].set(title='error', xlabel='Lateral (mm)')
            fig.suptitle(f'M4b on {split} sample {sid}')
            fig.tight_layout()
            pp = FIG / f'm4b_diag_{split[:5]}_{sid}.png'
            if pp.exists():
                raise SystemExit(f'refusing to overwrite {pp}')
            fig.savefig(pp, dpi=cfg['preview']['dpi'])
            plt.close(fig)
            print('wrote', pp)

    out['config'] = {'config_path': CFG, 'model': cfg['model'], 'train': cfg['train'],
                     'seed': cfg['seed'], 'input': cfg['scatter']['input'],
                     'normalization_file': cfg['scatter']['normalization_path']}
    with Path(DIAG_JSON).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('wrote', DIAG_JSON)

    for tag in ['m4', 'm4b']:
        for split in ['train', 'validation']:
            s = out['splits'][tag][split]
            print(f'{tag:4s} {split:10s} relL2 {s["relative_l2_rms_of_samples"]:.5f} '
                  f'MAE {s["mae_ms"]:7.2f} RMSE {s["rmse_ms"]:7.2f} SSIM {s["ssim"]:.4f} '
                  f'defectMAE {s["defect_mae_ms"]:7.2f} dice {s["dice"]:.4f} iou {s["iou"]:.4f}')


if __name__ == '__main__':
    main()

"""M4d checks on the M4c local model: defect-region metrics and input sensitivity."""
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

CFG = r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4b_fno.json'
LOCAL_RUN = Path(r'E:\FNO_research\results\m4c_local_20261020_170451')
OUT = r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4d_local_checks.json'


def mean_image(cfg, splits):
    maps = []
    for r in splits['train']:
        with h5py.File(m3.sample_path(cfg, r['sample_id']), 'r') as f:
            maps.append(f['c_200'][:])
    return np.mean(np.stack(maps), axis=0).astype(np.float32)


def main():
    cfg = m3.cfg_load(CFG)
    cfg['model'] = dict(cfg['model'])
    cfg['model']['mapping'] = 'local'
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    win = m4.gaussian_window(11, 1.5, torch.device('cpu'))
    ck = torch.load(LOCAL_RUN / 'best.pt', map_location=dev, weights_only=False)
    model = m4.FNO2d(cfg['model']).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    mean_map = mean_image(cfg, splits)
    mm = torch.tensor(mean_map).flatten()
    mm = (mm - mm.mean()) / mm.std()

    out = {'checkpoint': str(LOCAL_RUN / 'best.pt'), 'splits': {}}
    for split in ['train', 'validation']:
        loader = m3.make_loader(cfg, splits[split], norm, False, 32, 4)
        p, t = diag.collect(model, loader, dev)
        s = diag.suite(p, t, norm, win)
        pm = m4.to_ms(p, norm)
        tm = m4.to_ms(t, norm)
        cors = []
        for i in range(pm.shape[0]):
            v = pm[i, 0].flatten()
            v = (v - v.mean()) / (v.std() + 1e-12)
            cors.append(float((v * mm).mean()))
        g = torch.Generator().manual_seed(cfg['seed'])
        perm = torch.randperm(p.shape[0], generator=g)
        out['splits'][split] = {
            'samples': int(p.shape[0]),
            'overall_mae_ms': s['mae_ms'],
            'defect_mae_ms': s['defect_mae_ms'],
            'dice': s['dice'], 'iou': s['iou'],
            'relative_l2_rms_of_samples': s['relative_l2_rms_of_samples'],
            'rmse_ms': s['rmse_ms'], 'ssim': s['ssim'],
            'pred_cross_sample_std_ms': float(pm.std(dim=0).mean()),
            'true_cross_sample_std_ms': float(tm.std(dim=0).mean()),
            'ratio_pred_over_true': float(pm.std(dim=0).mean() / tm.std(dim=0).mean()),
            'mean_spatial_corr_with_training_mean_image': float(np.mean(cors)),
            'l1_true_pairing': float(F.l1_loss(p, t)),
            'l1_shuffled_pairing': float(F.l1_loss(p, t[perm])),
            'l1_change_when_shuffled': float(F.l1_loss(p, t[perm]) - F.l1_loss(p, t)),
        }
        print(split, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in out['splits'][split].items()}, indent=1), flush=True)
    with Path(OUT).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()

"""M4d-E stage 1 item 1 (revised): permutation-based input-dependence test.

Permutation is the main test: a fixed-seed derangement (no fixed points) reassigns the
inputs, and for every sample i we compare the Dice of the permuted prediction against
label i and against label j. Zero input is out-of-distribution (there is no defect-free
sample in training) and is reported descriptively only.
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

BASE = r'E:\FNO_research\repos\fno_light\repro_lu2026'
CFG = BASE + r'\configs\m4b_fno.json'
LOCAL_RUN = Path(r'E:\FNO_research\results\m4c_local_20261020_170451')
OUT = BASE + r'\configs\m4d_E_inputdep.json'
THR = 5000.0


def derangement(n, seed):
    g = np.random.default_rng(seed)
    while True:
        p = g.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def predict(model, x, dev, batch=32):
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            outs.append(model(x[i:i + batch].to(dev)).float().cpu())
    return torch.cat(outs)


def dice(pred_ms, true_ms):
    a = pred_ms < THR
    b = true_ms < THR
    inter = (a & b).flatten(1).sum(1).double()
    return (2 * inter / (a.flatten(1).sum(1).double() + b.flatten(1).sum(1).double()).clamp(min=1))


def main():
    cfg = m3.cfg_load(CFG)
    cfg['model'] = dict(cfg['model'])
    cfg['model']['mapping'] = 'local'
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = torch.load(LOCAL_RUN / 'best.pt', map_location=dev, weights_only=False)
    model = m4.FNO2d(cfg['model']).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()

    maps = []
    for r in splits['train']:
        with h5py.File(m3.sample_path(cfg, r['sample_id']), 'r') as f:
            maps.append(f['c_200'][:])
    mean_map = torch.tensor(np.mean(np.stack(maps), 0).astype(np.float32))

    res = {'checkpoint': str(LOCAL_RUN / 'best.pt'), 'dice_threshold_ms': THR,
           'permutation': {'type': 'single derangement, no fixed points', 'seed': 7},
           'splits': {}}
    for split in ['train', 'validation']:
        ds = m3.M2Dataset(cfg, splits[split], norm)
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        t = torch.stack([ds[i][1] for i in range(len(ds))])
        n = x.shape[0]
        perm = derangement(n, 7)
        assert len(set(perm.tolist())) == n and not np.any(perm == np.arange(n))
        tm = m4.to_ms(t, norm)
        pm = m4.to_ms(predict(model, x, dev), norm)          # normal
        pm_p = m4.to_ms(predict(model, x[perm], dev), norm)  # permuted inputs
        d_norm = dice(pm, tm)              # prediction i vs label i
        d_perm_i = dice(pm_p, tm)          # permuted prediction vs label i
        d_perm_j = dice(pm_p, tm[perm])    # permuted prediction vs its own label j
        drop = d_norm - d_perm_i
        frac = (d_perm_j > d_perm_i)
        room = (pm_p - mean_map[None, None]).abs()
        entry = {
            'samples': int(n),
            'dice_normal_mean': float(d_norm.mean()), 'dice_normal_median': float(d_norm.median()),
            'dice_permuted_vs_label_i_mean': float(d_perm_i.mean()),
            'dice_permuted_vs_label_i_median': float(d_perm_i.median()),
            'dice_permuted_vs_own_label_j_mean': float(d_perm_j.mean()),
            'dice_permuted_vs_own_label_j_median': float(d_perm_j.median()),
            'drop_vs_normal_mean': float(drop.mean()), 'drop_vs_normal_median': float(drop.median()),
            'fraction_dice_j_gt_dice_i': float(frac.float().mean()),
            'mean_abs_diff_vs_mean_image_ms': float(room.mean()),
            'per_sample': {'dice_normal': d_norm.tolist(), 'dice_perm_i': d_perm_i.tolist(),
                           'dice_perm_j': d_perm_j.tolist()},
        }
        if split == 'validation':
            zero_x = torch.full_like(x, float(norm['fmc']['offset']))
            pz = m4.to_ms(predict(model, zero_x, dev), norm)
            entry['zero_input_descriptive'] = {
                'note': 'out of distribution: training set has no defect-free sample',
                'mean_abs_diff_vs_mean_image_ms': float((pz - mean_map[None, None]).abs().mean()),
                'fraction_pixels_below_5000': float((pz < THR).float().mean()),
                'prediction_mean_ms': float(pz.mean()),
                'prediction_std_across_samples_ms': float(pz.std(dim=0).mean()),
            }
        res['splits'][split] = entry
        print(split, json.dumps({k: v for k, v in entry.items() if k != 'per_sample'}, indent=1),
              flush=True)
    with Path(OUT).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()

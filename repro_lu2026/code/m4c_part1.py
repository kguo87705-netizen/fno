"""M4c part 1: cheap diagnostics before any structural change.

Stages
    sens   - input sensitivity of the saved M4b checkpoint (no training)
    subset - memorisation curve for N = 8/32/128/375 training samples, 1500 steps each
    lowlr  - full training set at lr 3e-4 and 1e-4 for 100 epochs

Every run uses the M4b input (scattering field, train-set min/max), L1 loss, batch 32,
seed 42, and reports training-set and validation-set metrics including defect-region
MAE, Dice and IoU. The test split is never touched.
"""
import json
import sys
import time
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
M4B_RUN = Path(r'E:\FNO_research\results\m4b_train_20261020_132524')
OUT_JSON = {
    'sens': r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4c_sensitivity.json',
    'subset': r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4c_subsets.json',
    'lowlr': r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4c_lowlr.json',
}


def write_json(path, obj):
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def prep(cfg):
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg['train']['tf32'])
    win = m4.gaussian_window(cfg['eval']['ssim_window'], cfg['eval']['ssim_sigma'], torch.device('cpu'))
    return splits, norm, win


def load_m4b(cfg, dev):
    ck = torch.load(M4B_RUN / 'best.pt', map_location=dev, weights_only=False)
    model = m4.FNO2d(cfg['model']).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck


def mean_image(cfg, splits):
    maps = []
    for r in splits['train']:
        with h5py.File(m3.sample_path(cfg, r['sample_id']), 'r') as f:
            maps.append(f['c_200'][:])
    return np.mean(np.stack(maps), axis=0).astype(np.float32)


def stage_sens(cfg, splits, norm, win, dev):
    model, ck = load_m4b(cfg, dev)
    mean_map = mean_image(cfg, splits)
    res = {'checkpoint': str(M4B_RUN / 'best.pt'), 'best_epoch': int(ck['epoch']),
           'best_val_l1': float(ck['val_l1']), 'splits': {}}
    for split in ['train', 'validation']:
        loader = m3.make_loader(cfg, splits[split], norm, False, 32, num_workers=4)
        p, t = diag.collect(model, loader, dev)
        pm = m4.to_ms(p, norm)
        tm = m4.to_ms(t, norm)
        pred_std = float(pm.std(dim=0).mean())
        true_std = float(tm.std(dim=0).mean())
        # Spatial correlation of each prediction with the training-mean image.
        mm = torch.tensor(mean_map).flatten()
        mm = (mm - mm.mean()) / mm.std()
        cors = []
        for i in range(pm.shape[0]):
            v = pm[i, 0].flatten()
            v = (v - v.mean()) / (v.std() + 1e-12)
            cors.append(float((v * mm).mean()))
        # Shuffled pairing: predictions stay, targets are permuted.
        gen = torch.Generator().manual_seed(cfg['seed'])
        perm = torch.randperm(pm.shape[0], generator=gen)
        l1_true = float(F.l1_loss(p, t))  # normalised L1 on the true pairing
        l1_perm = float(F.l1_loss(p, t[perm]))
        res['splits'][split] = {
            'samples': pm.shape[0],
            'pred_cross_sample_std_ms': pred_std,
            'true_cross_sample_std_ms': true_std,
            'ratio_pred_over_true': pred_std / true_std,
            'mean_spatial_corr_with_training_mean_image': float(np.mean(cors)),
            'val_l1_normalized_true_pairing': l1_true,
            'val_l1_normalized_shuffled_pairing': l1_perm,
            'l1_change_when_shuffled': l1_perm - l1_true,
            'mae_ms_true_pairing': float((pm - tm).abs().mean()),
            'mae_ms_shuffled_pairing': float((pm - tm[perm]).abs().mean()),
        }
        print(split, json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in res['splits'][split].items()}, indent=1), flush=True)
    write_json(OUT_JSON['sens'], res)
    print('wrote', OUT_JSON['sens'])


def train_steps(cfg, norm, rows, steps, lr, dev, seed, batch):
    m4.seed_all(seed)
    ds = m3.M2Dataset(cfg, rows, norm)
    model = m4.FNO2d(cfg['model']).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(rows)
    xbuf = ybuf = None
    if n <= batch:  # keep the whole subset on the device
        xb = torch.stack([ds[i][0] for i in range(n)]).to(dev)
        yb = torch.stack([ds[i][1] for i in range(n)]).to(dev)
        xbuf, ybuf = xb, yb
    t0 = time.perf_counter()
    hist = []
    for step in range(1, steps + 1):
        model.train()
        if xbuf is not None:
            x, y = xbuf, ybuf
        else:
            idx = torch.randint(0, n, (batch,), generator=g).tolist()
            xs, ys = zip(*[ds[i] for i in idx])
            x = torch.stack(xs).to(dev)
            y = torch.stack(ys).to(dev)
        opt.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(x), y)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            hist.append({'step': step, 'l1': float(loss.detach()), 'seconds': time.perf_counter() - t0})
            if step % 300 == 0:
                print(f'  step {step} l1 {float(loss.detach()):.5f} ({time.perf_counter()-t0:.0f}s)', flush=True)
    model.eval()
    with torch.no_grad():
        if xbuf is not None:
            p = model(xbuf).float().cpu()
            t = ybuf.float().cpu()
        else:
            loader = m3.make_loader(cfg, rows, norm, False, 32, num_workers=4)
            p, t = diag.collect(model, loader, dev)
    return model, p, t, hist


def stage_subset(cfg, splits, norm, win, dev):
    res = {'samples': [8, 32, 128, 375], 'steps': 1500, 'lr': 1e-3, 'scheduler': 'constant',
           'batch': 'min(32, N)', 'runs': {}}
    for n in res['samples']:
        rows = splits['train'][:n]
        print(f'--- N={n}', flush=True)
        model, p, t, hist = train_steps(cfg, norm, rows, 1500, 1e-3, dev, cfg['seed'], min(32, n))
        s = diag.suite(p, t, norm, win)
        res['runs'][str(n)] = {'train_l1_final': hist[-1]['l1'],
                              'seconds': hist[-1]['seconds'],
                              'history': hist, 'metrics': s}
        print(f'N={n}: l1 {hist[-1]["l1"]:.5f} dice {s["dice"]:.4f} iou {s["iou"]:.4f} '
              f'defectMAE {s["defect_mae_ms"]:.1f} ({hist[-1]["seconds"]:.0f}s)', flush=True)
    write_json(OUT_JSON['subset'], res)
    print('wrote', OUT_JSON['subset'])


def stage_lowlr(cfg, splits, norm, win, dev):
    res = {'epochs': 100, 'batch': 32, 'runs': {}}
    train_loader = m3.make_loader(cfg, splits['train'], norm, True, 32, 4,
                                  torch.Generator().manual_seed(cfg['seed']))
    val_loader = m3.make_loader(cfg, splits['validation'], norm, False, 32, 2)
    for lr in [3e-4, 1e-4]:
        m4.seed_all(cfg['seed'])
        model = m4.FNO2d(cfg['model']).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
        t0 = time.perf_counter()
        hist = []
        for ep in range(1, 101):
            model.train()
            tot, nb = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = F.l1_loss(model(x), y)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                nb += 1
            sch.step()
            rec = {'epoch': ep, 'train_l1': tot / nb, 'seconds': time.perf_counter() - t0}
            if ep % 5 == 0 or ep == 1:
                model.eval()
                with torch.no_grad():
                    v = m4.evaluate(model, val_loader, dev, norm)
                rec['val_l1'] = v['l1']
            hist.append(rec)
            if ep % 20 == 0:
                print(f'  lr {lr:g} epoch {ep} train {rec["train_l1"]:.5f} '
                      f'val {rec.get("val_l1", float("nan")):.5f} ({rec["seconds"]:.0f}s)', flush=True)
        model.eval()
        tl = m3.make_loader(cfg, splits['train'], norm, False, 32, 4)
        p, t = diag.collect(model, tl, dev)
        s_train = diag.suite(p, t, norm, win)
        pv, tv = diag.collect(model, val_loader, dev)
        s_val = diag.suite(pv, tv, norm, win)
        res['runs'][f'{lr:g}'] = {'history': hist, 'train': s_train, 'validation': s_val,
                                  'seconds': time.perf_counter() - t0}
        print(f'lr {lr:g}: train dice {s_train["dice"]:.4f} val dice {s_val["dice"]:.4f} '
              f'train MAE {s_train["mae_ms"]:.2f} ({time.perf_counter()-t0:.0f}s)', flush=True)
    write_json(OUT_JSON['lowlr'], res)
    print('wrote', OUT_JSON['lowlr'])


def main():
    stage = sys.argv[1]
    cfg = m3.cfg_load(CFG)
    splits, norm, win = prep(cfg)
    dev = torch.device('cuda')
    if stage == 'sens':
        stage_sens(cfg, splits, norm, win, dev)
    elif stage == 'subset':
        stage_subset(cfg, splits, norm, win, dev)
    elif stage == 'lowlr':
        stage_lowlr(cfg, splits, norm, win, dev)
    else:
        raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
    main()

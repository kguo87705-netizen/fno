"""M4c part 3: T/R grid-mapping ablation, 150 epochs per variant.

Only the mapping layer changes. Everything else matches M4b: scattering input with
train-set min/max, width 32, 40x40 modes, 4 Fourier layers, L1, batch 32, seed 42.
Variants
    local  - T: Conv2d(kernel 8x1, stride 5, padding 2) giving 1000 -> 200; R: linear
             interpolation 64 -> 200 followed by a learnable 1x3 convolution
    interp - both axes by bilinear interpolation, no learnable mapping at all

Stages
    run   <variant> <lr>   train one variant, save history and metrics
    figs  <variant>       sample figures for the chosen variant
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE = r'E:\FNO_research\repos\fno_light\repro_lu2026\code'
sys.path.insert(0, CODE)
import m3_data as m3
import m4_fno as m4
import m4b_diagnostics as diag

CFG = r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4b_fno.json'
OUT = Path(r'E:\FNO_research\repos\fno_light\repro_lu2026\configs\m4c_ablation.json')
FIG = Path(r'E:\FNO_research\repos\fno_light\repro_lu2026\reports')
EPOCHS = 150
VAL_EVERY = 5
VAL_IDS = ['m2_000378', 'm2_000401', 'm2_000405']
TRAIN_IDS = ['m2_000000', 'm2_000001', 'm2_000002']


def cfg_for(variant):
    cfg = m3.cfg_load(CFG)
    cfg['model'] = dict(cfg['model'])
    cfg['model']['mapping'] = variant
    return cfg


def run_variant(variant, lr, dev):
    cfg = cfg_for(variant)
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    m4.seed_all(cfg['seed'])
    train_loader = m3.make_loader(cfg, splits['train'], norm, True, 32, 4,
                                  torch.Generator().manual_seed(cfg['seed']))
    val_loader = m3.make_loader(cfg, splits['validation'], norm, False, 32, 2)
    model = m4.FNO2d(cfg['model']).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    run = m3.new_dir(cfg['results_root'], f'm4c_{variant}')
    hist = []
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
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
        if ep % VAL_EVERY == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                v = m4.evaluate(model, val_loader, dev, norm)
            rec['val_l1'] = v['l1']
        hist.append(rec)
        if ep % 25 == 0:
            print(f'  {variant} ep {ep} train {rec["train_l1"]:.5f} '
                  f'val {rec.get("val_l1", float("nan")):.5f} ({rec["seconds"]:.0f}s)', flush=True)
    model.eval()
    tl = m3.make_loader(cfg, splits['train'], norm, False, 32, 4)
    p, t = diag.collect(model, tl, dev)
    s_train = diag.suite(p, t, norm, m4.gaussian_window(11, 1.5, torch.device('cpu')))
    pv, tv = diag.collect(model, val_loader, dev)
    s_val = diag.suite(pv, tv, norm, m4.gaussian_window(11, 1.5, torch.device('cpu')))
    torch.save({'model': model.state_dict(), 'config': cfg['model'], 'epoch': EPOCHS,
                'variant': variant, 'lr': lr}, run / 'best.pt')
    out = {'variant': variant, 'lr': lr, 'epochs': EPOCHS, 'parameters': n_par,
           'history': hist, 'train': s_train, 'validation': s_val,
           'seconds': time.perf_counter() - t0, 'run_dir': str(run)}
    print(f'{variant}: lr {lr:g} train dice {s_train["dice"]:.4f} val dice {s_val["dice"]:.4f} '
          f'train MAE {s_train["mae_ms"]:.2f} ({out["seconds"]:.0f}s)', flush=True)
    return out


def merge(out):
    cur = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {'variants': {}}
    cur['variants'][out['variant']] = out
    cur['epochs'] = EPOCHS
    OUT.write_text(json.dumps(cur, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('wrote', OUT)


def stage_figs(variant, dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg = cfg_for(variant)
    _, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    cur = json.loads(OUT.read_text(encoding='utf-8'))
    run_dir = Path(cur['variants'][variant]['run_dir'])
    ck = torch.load(run_dir / 'best.pt', map_location=dev, weights_only=False)
    model = m4.FNO2d(cfg['model']).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    for split, wanted in [('validation', VAL_IDS), ('train', TRAIN_IDS)]:
        loader = m3.make_loader(cfg, splits[split], norm, False, 32, 4)
        p, t = diag.collect(model, loader, dev)
        index = [r['sample_id'] for r in splits[split]]
        for sid in wanted:
            if sid not in index:
                print('skip', sid)
                continue
            i = index.index(sid)
            gt = m4.to_ms(t[i, 0], norm).numpy()
            pr = m4.to_ms(p[i, 0], norm).numpy()
            err = pr - gt
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
            im = axes[0].imshow(gt, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
            fig.colorbar(im, ax=axes[0], label='m/s')
            axes[0].set(title=f'{sid} truth', xlabel='Lateral (mm)', ylabel='Depth (mm)')
            im = axes[1].imshow(pr, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
            fig.colorbar(im, ax=axes[1], label='m/s')
            axes[1].set(title=f'{variant} prediction', xlabel='Lateral (mm)')
            lim = float(np.abs(err).max())
            im = axes[2].imshow(err, extent=[0, 40, 40, 0], vmin=-lim, vmax=lim, cmap='coolwarm')
            fig.colorbar(im, ax=axes[2], label='m/s')
            axes[2].set(title='error', xlabel='Lateral (mm)')
            fig.suptitle(f'M4c variant "{variant}" on {split} sample {sid}')
            fig.tight_layout()
            pp = FIG / f'm4c_{variant}_{split[:5]}_{sid}.png'
            if pp.exists():
                raise SystemExit(f'refusing to overwrite {pp}')
            fig.savefig(pp, dpi=cfg['preview']['dpi'])
            plt.close(fig)
            print('wrote', pp)


def main():
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    if sys.argv[1] == 'run':
        merge(run_variant(sys.argv[2], float(sys.argv[3]), dev))
    elif sys.argv[1] == 'figs':
        stage_figs(sys.argv[2], dev)
    else:
        raise SystemExit('unknown stage')


if __name__ == '__main__':
    main()

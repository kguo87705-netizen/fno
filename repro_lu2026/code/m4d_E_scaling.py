"""M4d-E stage 1 items 4-5: data-scaling and capacity controls with two seeds.

One run = one fresh local-mapping model trained for a fixed 1800 steps (cosine over the
steps, no early stopping). Results are written per run so interrupted sessions keep what
finished; `merge` aggregates means and seed differences.

Usage
    python m4d_E_scaling.py run <N> <seed> <modes> <cache 0|1>
    python m4d_E_scaling.py merge
    python m4d_E_scaling.py fig
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

CODE = r'E:\FNO_research\repos\fno_light\repro_lu2026\code'
sys.path.insert(0, CODE)
import m3_data as m3
import m4_fno as m4
import m4b_diagnostics as diag

BASE = r'E:\FNO_research\repos\fno_light\repro_lu2026'
CFG = BASE + r'\configs\m4b_fno.json'
OUTDIR = Path(r'E:\FNO_research\results\m4d_E_runs')
MERGED = BASE + r'\configs\m4d_E_scaling.json'
FIG = Path(BASE) / 'reports'
STEPS = 1800
LR = 1e-3
BATCH = 32


def build_cfg(modes=40, seed=0):
    cfg = m3.cfg_load(CFG)
    cfg['model'] = dict(cfg['model'])
    cfg['model']['mapping'] = 'local'
    cfg['model']['modes'] = modes
    cfg['seed'] = seed
    return cfg


class RamSet(torch.utils.data.Dataset):
    def __init__(self, cfg, rows, norm):
        base = m3.M2Dataset(cfg, rows, norm)
        n = len(base)
        x0, y0 = base[0]
        # Preallocate: torch.stack would hold both copies at once (12 GB peak at N=375).
        self.x = torch.empty((n,) + tuple(x0.shape), dtype=x0.dtype)
        self.y = torch.empty((n,) + tuple(y0.shape), dtype=y0.dtype)
        self.x[0], self.y[0] = x0, y0
        for i in range(1, n):
            x, y = base[i]
            self.x[i] = x
            self.y[i] = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def param_groups(model):
    spec = sum(p.numel() for n, p in model.named_parameters() if '.spec.' in n)
    mapping = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith('t_conv') or n.startswith('r_post'))
    total = sum(p.numel() for p in model.parameters())
    return {'total': total, 'spectral': spec, 'mapping': mapping, 'other': total - spec - mapping}


def main():
    stage = sys.argv[1]
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    if stage == 'run':
        n, seed, modes = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        cache = bool(int(sys.argv[5]))
        cfg = build_cfg(modes, seed)
        _, splits = m3.load_splits(cfg)
        norm = m3.norm_from_cfg(cfg)
        rows = splits['train'][:n]
        m4.seed_all(seed)
        ds = RamSet(cfg, rows, norm) if cache else m3.M2Dataset(cfg, rows, norm)
        model = m4.FNO2d(cfg['model']).to(dev)
        groups = param_groups(model)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-6)
        g = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        hist = []
        for step in range(1, STEPS + 1):
            model.train()
            idx = torch.randint(0, n, (min(BATCH, n),), generator=g).tolist()
            x = torch.stack([ds[i][0] for i in idx]).to(dev, non_blocking=True)
            y = torch.stack([ds[i][1] for i in idx]).to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(x), y)
            loss.backward()
            opt.step()
            sch.step()
            if step % 200 == 0:
                hist.append({'step': step, 'train_l1': float(loss.detach()),
                             'seconds': time.perf_counter() - t0})
                if step % 600 == 0:
                    print(f'  N={n} s={seed} m={modes} step {step} '
                          f'l1 {float(loss.detach()):.5f} ({time.perf_counter()-t0:.0f}s)', flush=True)
        model.eval()
        win = m4.gaussian_window(11, 1.5, torch.device('cpu'))
        p_tr, t_tr = [], []
        with torch.no_grad():
            for i in range(0, n, BATCH):
                sel = list(range(i, min(i + BATCH, n)))
                xs = torch.stack([ds[j][0] for j in sel]).to(dev)
                ys = torch.stack([ds[j][1] for j in sel]).to(dev)
                p_tr.append(model(xs).float().cpu())
                t_tr.append(ys.float().cpu())
        s_tr = diag.suite(torch.cat(p_tr), torch.cat(t_tr), norm, win)
        val_loader = m3.make_loader(cfg, splits['validation'], norm, False, 32, 2)
        pv, tv = diag.collect(model, val_loader, dev)
        s_val = diag.suite(pv, tv, norm, win)
        out = {'label': f'E_local_N{n}_m{modes}_s{seed}', 'n_train': n, 'seed': seed,
               'modes': modes, 'steps': STEPS, 'cache': cache, 'parameters': groups,
               'seconds': time.perf_counter() - t0, 'history': hist,
               'train': s_tr, 'validation': s_val}
        OUTDIR.mkdir(parents=True, exist_ok=True)
        p = OUTDIR / f'N{n}_m{modes}_s{seed}.json'
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(f'N={n} m={modes} s={seed}: train dice {s_tr["dice"]:.4f} val dice {s_val["dice"]:.4f} '
              f'val defectMAE {s_val["defect_mae_ms"]:.1f} val MAE {s_val["mae_ms"]:.2f} '
              f'({out["seconds"]:.0f}s) -> {p.name}', flush=True)
    elif stage == 'merge':
        import numpy as np
        runs = {}
        for p in sorted(OUTDIR.glob('*.json')):
            d = json.loads(p.read_text(encoding='utf-8'))
            runs[d['label']] = d
        agg = {}
        for key in sorted({(d['n_train'], d['modes']) for d in runs.values()}):
            sel = [d for d in runs.values() if (d['n_train'], d['modes']) == key]
            agg[f'N{key[0]}_m{key[1]}'] = {
                'n_train': key[0], 'modes': key[1], 'seeds': [d['seed'] for d in sel],
                'val_dice_mean': float(np.mean([d['validation']['dice'] for d in sel])),
                'val_dice_values': [d['validation']['dice'] for d in sel],
                'val_dice_seed_diff': float(np.max([d['validation']['dice'] for d in sel])
                                            - np.min([d['validation']['dice'] for d in sel])),
                'train_dice_mean': float(np.mean([d['train']['dice'] for d in sel])),
                'val_defect_mae_mean': float(np.mean([d['validation']['defect_mae_ms'] for d in sel])),
                'val_mae_mean': float(np.mean([d['validation']['mae_ms'] for d in sel])),
                'val_rel_l2_mean': float(np.mean([d['validation']['relative_l2_rms_of_samples'] for d in sel])),
                'parameters': sel[0]['parameters'],
                'seconds': [d['seconds'] for d in sel],
            }
        merged = {'steps': STEPS, 'lr': LR, 'batch': BATCH, 'runs': agg,
                  'raw_labels': list(runs.keys())}
        Path(MERGED).write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n',
                                encoding='utf-8', newline='\n')
        for k, v in agg.items():
            print(k, 'val dice mean %.4f' % v['val_dice_mean'], 'seed diff %.4f' % v['val_dice_seed_diff'],
                  'train dice %.4f' % v['train_dice_mean'])
        print('wrote', MERGED)
    elif stage == 'fig':
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        d = json.loads(Path(MERGED).read_text(encoding='utf-8'))
        xs, ys, ys_tr = [], [], []
        for k, v in d['runs'].items():
            if v['modes'] != 40:
                continue
            xs.append(v['n_train'])
            ys.append(v['val_dice_mean'])
            ys_tr.append(v['train_dice_mean'])
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]; ys_tr = np.array(ys_tr)[order]
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
        ax.plot(xs, ys, 'o-', label='validation Dice (mean of 2 seeds)')
        ax.plot(xs, ys_tr, 's--', label='training-subset Dice')
        ax.axhline(0.5838, color='k', ls=':', label='training-mean image 0.5838')
        m20 = d['runs'].get('N375_m20')
        if m20:
            ax.scatter([375], [m20['val_dice_mean']], marker='*', s=150, color='tab:red',
                       label=f"modes 20, N=375 ({m20['val_dice_mean']:.3f})")
        ax.set(xlabel='training samples used', ylabel='Dice (threshold 5000 m/s)',
               title='E-local data scaling, 1800 steps per run, 2 seeds')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIG / 'm4d_E_scaling.png'
        if p.exists():
            raise SystemExit(f'refusing to overwrite {p}')
        fig.savefig(p, dpi=120)
        print('wrote', p)
    else:
        raise SystemExit('unknown stage')


if __name__ == '__main__':
    main()

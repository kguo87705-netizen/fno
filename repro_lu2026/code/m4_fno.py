"""M4: plain PyTorch FNO2d baseline that maps a 64-transmitter FMC to a 200x200 speed map.

Deliberately minimal: pointwise lifting, learnable linear resampling of T and R, four
standard Fourier layers with 40x40 modes, and a two-layer pointwise output MLP. No U-Net
branch, no attention, no physics loss.

Stages:
    bench    - step time, peak GPU memory and a full-training estimate
    overfit  - 8 training samples only, loss must approach zero before the full run
    train    - full training with early stopping, then validation metrics and figures
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import m3_data as m3


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_tf32(flag):
    torch.backends.cuda.matmul.allow_tf32 = bool(flag)
    torch.backends.cudnn.allow_tf32 = bool(flag)


class SpectralConv2d(nn.Module):
    """Spectral convolution keeping modes1 x modes2 low-frequency modes."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.cin, self.cout, self.m1, self.m2 = cin, cout, m1, m2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w3 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w4 = nn.Parameter(scale * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros(b, self.cout, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, -self.m1:, :self.m2], self.w2)
        out[:, :, :self.m1, -self.m2:] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :self.m1, -self.m2:], self.w3)
        out[:, :, -self.m1:, -self.m2:] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, -self.m1:, -self.m2:], self.w4)
        return torch.fft.irfft2(out, s=(h, w), norm='ortho')


class FourierLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes, modes)
        self.point = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spec(x) + self.point(x))


class FNO2d(nn.Module):
    """FMC (B,64,1000,64) -> speed (B,1,200,200), both normalized."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.lift = nn.Linear(c['in_channels'] + 2, c['width'])
        self.t_resample = nn.Linear(c['t_in'], c['t_out'])
        self.r_resample = nn.Linear(c['r_in'], c['r_out'])
        self.layers = nn.ModuleList([FourierLayer(c['width'], c['modes']) for _ in range(c['n_layers'])])
        self.project = nn.Sequential(
            nn.Linear(c['width'], c['q_hidden']), nn.GELU(),
            nn.Linear(c['q_hidden'], c['out_channels']))
        self.register_buffer('t_grid', torch.linspace(0.0, 1.0, c['t_in']).view(1, 1, c['t_in'], 1), persistent=False)
        self.register_buffer('r_grid', torch.linspace(0.0, 1.0, c['r_in']).view(1, 1, 1, c['r_in']), persistent=False)

    def forward(self, x):
        b, _, nt, nr = x.shape
        t = self.t_grid.expand(b, 1, nt, nr)
        r = self.r_grid.expand(b, 1, nt, nr)
        h = torch.cat([x, t, r], dim=1)                     # (B, C+2, T, R)
        h = self.lift(h.permute(0, 2, 3, 1))                # (B, T, R, width)
        h = h.permute(0, 3, 1, 2)                           # (B, width, T, R)
        h = self.t_resample(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # T: 1000 -> 200
        h = self.r_resample(h)                              # R: 64 -> 200
        for layer in self.layers:
            h = layer(h)
        return self.project(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    parts = {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()}
    return total, parts


def to_ms(t, norm):
    return (t - norm['speed']['offset']) / norm['speed']['scale']


def gaussian_window(size, sigma, device):
    c = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(c ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(a, b, win, data_range):
    """Mean SSIM over valid windows; a, b are (N,1,H,W) in m/s."""
    mu1 = F.conv2d(a, win)
    mu2 = F.conv2d(b, win)
    mu1s, mu2s, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, win) - mu1s
    s2 = F.conv2d(b * b, win) - mu2s
    s12 = F.conv2d(a * b, win) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1s + mu2s + c1) * (s1 + s2 + c2))
    return float(m.mean())


def evaluate(model, loader, device, norm):
    """Validation loss (normalized L1) plus per-sample metrics in m/s."""
    model.eval()
    l1_sum, n = 0.0, 0
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            p = model(x)
            l1_sum += float(F.l1_loss(p, y, reduction='sum'))
            n += y.numel()
            preds.append(p.float().cpu())
            trues.append(y.float().cpu())
    p = torch.cat(preds)
    t = torch.cat(trues)
    pm, tm = to_ms(p, norm), to_ms(t, norm)
    diff = pm - tm
    per = {
        'per_sample_rel_l2': (diff.flatten(1).norm(dim=1) / tm.flatten(1).norm(dim=1)).tolist(),
        'per_sample_mae_ms': diff.abs().flatten(1).mean(dim=1).tolist(),
        'per_sample_rmse_ms': (diff.flatten(1) ** 2).mean(dim=1).sqrt().tolist(),
    }
    win = gaussian_window(11, 1.5, pm.device)
    per['ssim'] = [ssim(pm[i:i + 1], tm[i:i + 1], win, 4300.0) for i in range(pm.shape[0])]
    return dict(l1=l1_sum / n, pred=pm, true=tm, per=per)


def aggregate(per):
    a = np.asarray(per['per_sample_rel_l2'])
    mae = np.asarray(per['per_sample_mae_ms'])
    rmse = np.asarray(per['per_sample_rmse_ms'])
    ss = np.asarray(per['ssim'])
    rel = float(np.sqrt(np.mean(a ** 2)))
    return {'relative_l2_mean_of_samples': float(a.mean()), 'relative_l2_rms_of_samples': rel,
            'relative_l2_min': float(a.min()), 'relative_l2_max': float(a.max()),
            'mae_ms': float(mae.mean()), 'rmse_ms': float(rmse.mean()),
            'ssim': float(ss.mean()), 'ssim_min': float(ss.min()), 'ssim_max': float(ss.max())}


def stage_bench(cfg, splits, norm, device):
    print('building model')
    model = FNO2d(cfg['model']).to(device)
    total, parts = count_params(model)
    print(f'parameters {total:,} ' + json.dumps({k: v for k, v in parts.items()}))
    tr = cfg['train']
    rows = splits['train'][:cfg['loader']['batch_size']]
    ds = m3.M2Dataset(cfg, rows, norm)
    xb = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    yb = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr['lr'])
    res = {}
    for bs in [tr['batch_size']]:
        if bs > xb.shape[0]:
            break
        torch.cuda.reset_peak_memory_stats()
        model.train()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(xb[:bs]), yb[:bs])
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(xb[:bs]), yb[:bs])
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        step = (time.perf_counter() - t0) / 5
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        steps = math.ceil(cfg['split_counts']['train'] / bs)
        epoch = step * steps + 2.0
        res[bs] = {'step_seconds': step, 'steps_per_epoch': steps, 'epoch_seconds_est': epoch,
                   'peak_allocated_gib': peak, 'loss': float(loss.detach())}
        print(f'batch {bs}: {step*1000:.0f} ms/step, {steps} steps/epoch, epoch ~{epoch:.1f}s, '
              f'peak {peak:.2f} GiB, loss {float(loss.detach()):.4f}')
    total_s = res[tr['batch_size']]['epoch_seconds_est'] * tr['max_epochs']
    out = {'parameters': total, 'parameters_by_module': parts, 'device': torch.cuda.get_device_name(0),
           'batch_benchmarks': res, 'max_epochs': tr['max_epochs'],
           'full_training_seconds_est': total_s, 'full_training_hours_est': total_s / 3600,
           'note': 'estimate assumes every one of max_epochs runs and adds 2 s per epoch for validation'}
    run = m3.new_dir(cfg['results_root'], 'm4_bench')
    m3.write_json(run / 'm4_bench.json', out)
    print(f'estimated full training {total_s/60:.1f} min ({total_s/3600:.3f} h) -> {run}')
    return out


def stage_overfit(cfg, splits, norm, device):
    ov = cfg['overfit']
    seed_all(cfg['seed'])
    rows = splits['train'][:ov['n_samples']]
    print(f'overfit test on {len(rows)} training samples: ' + ', '.join(r['sample_id'] for r in rows))
    ds = m3.M2Dataset(cfg, rows, norm)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    y = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    model = FNO2d(cfg['model']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ov['lr'])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ov['max_steps'], eta_min=1e-05)
             if ov.get('scheduler') == 'cosine' else None)
    hist = []
    t0 = time.perf_counter()
    loss = None
    for step in range(1, ov['max_steps'] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(x), y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if step % ov['eval_every'] == 0 or step == 1:
            hist.append({'step': step, 'l1': float(loss.detach()), 'seconds': time.perf_counter() - t0})
            print(f'step {step:4d} l1 {float(loss.detach()):.6f} ({time.perf_counter()-t0:.1f}s)', flush=True)
        if float(loss.detach()) < ov['target_l1']:
            break
    elapsed = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = model(x)
    final_l1 = float(loss.detach())
    run = m3.new_dir(cfg['results_root'], 'm4_overfit')
    torch.save({'model': model.state_dict(), 'config': cfg['model']}, run / 'overfit.pt')
    m3.write_json(run / 'm4_overfit.json',
                  {'samples': [r['sample_id'] for r in rows], 'history': hist, 'seconds': elapsed,
                   'final_l1': final_l1, 'target_l1': ov['target_l1'],
                   'passed': bool(final_l1 < ov['target_l1'])})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.4))
    for i in range(n):
        gt = to_ms(y[i, 0].cpu(), norm)
        pr = to_ms(pred[i, 0].cpu(), norm)
        for r_, img, ttl in [(0, gt, 'truth'), (1, pr, 'prediction')]:
            ax = axes[r_, i]
            im = ax.imshow(img, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
            ax.set_title(f"{rows[i]['sample_id']} {rows[i]['letter']} {ttl}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, label='m/s', shrink=0.8)
    fig.suptitle(f"Overfit test, {n} samples: final L1 {final_l1:.4f} in {elapsed:.0f}s")
    p = Path(cfg['figure_dir']) / 'm4_overfit.png'
    if p.exists():
        raise SystemExit(f'refusing to overwrite {p}')
    fig.savefig(p, dpi=cfg['preview']['dpi'])
    plt.close(fig)
    print('wrote', p)
    if not (final_l1 < ov['target_l1']):
        raise SystemExit(f'overfit test FAILED: final L1 {final_l1:.4f} >= {ov["target_l1"]}')
    print(f'overfit PASSED: final L1 {final_l1:.6f} in {elapsed:.1f}s')
    return run


def _resolve_run_dir(cfg):
    """Fresh timestamped run dir, or the directory of an interrupted run when resuming."""
    tr = cfg['train']
    rf = tr.get('resume_from')
    if rf and rf not in ('latest', True):
        return Path(rf)
    if tr.get('resume') and rf == 'latest':
        cands = sorted(Path(cfg['results_root']).glob(tr.get('run_prefix', 'm4_train') + '_*'),
                        key=lambda p: p.stat().st_mtime)
        for c in reversed(cands):
            if (c / 'last.pt').exists():
                return c
    return m3.new_dir(cfg['results_root'], tr.get('run_prefix', 'm4_train'))


def stage_train(cfg, splits, norm, device):
    tr = cfg['train']
    seed_all(cfg['seed'])
    gen = torch.Generator()
    gen.manual_seed(cfg['seed'])
    train_loader = m3.make_loader(cfg, splits['train'], norm, True, tr['batch_size'],
                                  tr['num_workers'], gen)
    val_loader = m3.make_loader(cfg, splits['validation'], norm, False, tr['batch_size'], 2)
    model = FNO2d(cfg['model']).to(device)
    total, parts = count_params(model)
    print(f'parameters {total:,}')
    opt = torch.optim.Adam(model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr['max_epochs'], eta_min=tr['eta_min'])
    run = _resolve_run_dir(cfg)
    log_path = run / 'm4_train_log.jsonl'
    last_path = run / 'last.pt'
    start_epoch, best, best_epoch, bad, hist, prior_s = 1, float('inf'), 0, 0, [], 0.0
    if tr.get('resume') and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        best, best_epoch, bad, hist = ck['best'], ck['best_epoch'], ck['bad'], ck['hist']
        prior_s = ck.get('elapsed_total', 0.0)
        print(f'RESUME {run.name} at epoch {start_epoch} (best {best:.5f} @ {best_epoch})', flush=True)
    ck_every = int(tr.get('checkpoint_every', 10))
    t_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, tr['max_epochs'] + 1):
        model.train()
        t0 = time.perf_counter()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        train_l1 = tot / nb
        t_train = time.perf_counter() - t0
        lr = opt.param_groups[0]['lr']
        sched.step()
        ev = evaluate(model, val_loader, device, norm)
        val_l1 = ev['l1']
        improved = val_l1 < best - tr['min_delta']
        if improved:
            best, best_epoch, bad = val_l1, epoch, 0
            torch.save({'model': model.state_dict(), 'config': cfg['model'], 'epoch': epoch,
                        'val_l1': val_l1, 'normalization': norm}, run / 'best.pt')
        else:
            bad += 1
        rec = {'epoch': epoch, 'train_l1': train_l1, 'val_l1': val_l1, 'lr': lr,
               'train_seconds': t_train, 'elapsed': time.perf_counter() - t_start,
               'best': improved, 'bad_epochs': bad}
        hist.append(rec)
        with log_path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(rec) + '\n')
        if epoch % ck_every == 0:
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'scheduler': sched.state_dict(), 'epoch': epoch, 'best': best,
                        'best_epoch': best_epoch, 'bad': bad, 'hist': hist,
                        'elapsed_total': prior_s + (time.perf_counter() - t_start),
                        'config': cfg['model'], 'normalization': norm}, last_path)
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"epoch {epoch:4d} train {train_l1:.5f} val {val_l1:.5f} lr {lr:.2e} "
                  f"{t_train:.1f}s bad {bad}", flush=True)
        if tr.get('early_stop', True) and bad >= tr['patience']:
            print(f'early stop at epoch {epoch} (best {best:.5f} @ {best_epoch})')
            break
    total_s = prior_s + (time.perf_counter() - t_start)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    ck = torch.load(run / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    ev = evaluate(model, val_loader, device, norm)
    agg = aggregate(ev['per'])
    val_rows = splits['validation']
    agg['n_samples'] = len(val_rows)
    print('validation metrics', json.dumps(agg, indent=2))

    # Reference points computed in m/s: a constant background image, and the pixelwise
    # mean of the training maps. Both are produced without any learning.
    true_ms = ev['true']
    const = torch.full_like(true_ms, cfg['speed']['max_ms'])
    cdiff = const - true_ms
    cwin = gaussian_window(cfg['eval']['ssim_window'], cfg['eval']['ssim_sigma'], const.device)
    true_norm = true_ms * norm['speed']['scale'] + norm['speed']['offset']
    baseline = {
        'description': 'constant prediction of 5800 m/s at every pixel; no learning',
        'relative_l2_rms_of_samples': float(np.sqrt(np.mean(
            (cdiff.flatten(1).norm(dim=1) / true_ms.flatten(1).norm(dim=1)).pow(2).numpy()))),
        'mae_ms': float(cdiff.abs().mean()),
        'rmse_ms': float((cdiff.flatten(1) ** 2).mean(dim=1).sqrt().mean()),
        'ssim': float(np.mean([ssim(const[i:i + 1], true_ms[i:i + 1], cwin, 4300.0)
                               for i in range(const.shape[0])])),
        'val_l1_normalized': float((1.0 - true_norm).abs().mean()),
    }
    print('constant-background baseline', json.dumps(baseline, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = [h['epoch'] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ep, [h['train_l1'] for h in hist], label='train L1')
    ax[0].plot(ep, [h['val_l1'] for h in hist], label='validation L1')
    ax[0].axvline(best_epoch, ls=':', c='k', label=f'best epoch {best_epoch}')
    ax[0].set(xlabel='epoch', ylabel='L1 on normalized speed', title='loss', yscale='log')
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [h['lr'] for h in hist], c='tab:green')
    ax[1].set(xlabel='epoch', ylabel='learning rate', title='cosine schedule')
    ax[1].grid(alpha=0.3)
    fig.suptitle(f'FNO2d baseline: stopped after {len(hist)} epochs, best validation L1 {best:.5f}')
    fig.tight_layout()
    p = Path(cfg['figure_dir']) / (cfg.get('figure_prefix', 'm4') + '_training_curves.png')
    if p.exists():
        raise SystemExit(f'refusing to overwrite {p}')
    fig.savefig(p, dpi=cfg['preview']['dpi'])
    plt.close(fig)

    nfig = min(cfg['eval']['n_figures'], len(val_rows))
    rng = np.random.default_rng(cfg['seed'])
    pick = sorted(rng.choice(len(val_rows), size=nfig, replace=False).tolist())
    figs = []
    for i in pick:
        r = val_rows[i]
        gt, pr = ev['true'][i, 0].numpy(), ev['pred'][i, 0].numpy()
        err = pr - gt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        im = axes[0].imshow(gt, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
        fig.colorbar(im, ax=axes[0], label='m/s')
        axes[0].set(title=f"{r['sample_id']} {r['letter']} truth", xlabel='Lateral (mm)', ylabel='Depth (mm)')
        im = axes[1].imshow(pr, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
        fig.colorbar(im, ax=axes[1], label='m/s')
        axes[1].set(title='prediction', xlabel='Lateral (mm)')
        lim = float(np.abs(err).max())
        im = axes[2].imshow(err, extent=[0, 40, 40, 0], vmin=-lim, vmax=lim, cmap='coolwarm')
        fig.colorbar(im, ax=axes[2], label='m/s')
        axes[2].set(title=f'error (rel L2 {ev["per"]["per_sample_rel_l2"][i]:.4f})', xlabel='Lateral (mm)')
        fig.suptitle(f"Validation sample, MAE {ev['per']['per_sample_mae_ms'][i]:.1f} m/s, "
                     f"SSIM {ev['per']['ssim'][i]:.4f}")
        fig.tight_layout()
        pp = Path(cfg['figure_dir']) / (cfg.get('figure_prefix', 'm4') + f'_val_{r["sample_id"]}.png')
        if pp.exists():
            raise SystemExit(f'refusing to overwrite {pp}')
        fig.savefig(pp, dpi=cfg['preview']['dpi'])
        plt.close(fig)
        figs.append(str(pp))

    out = {
        'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'tf32': bool(tr['tf32']), 'seed': cfg['seed'], 'parameters': total,
        'parameters_by_module': parts, 'batch_size': tr['batch_size'], 'lr': tr['lr'],
        'max_epochs': tr['max_epochs'], 'patience': tr['patience'],
        'epochs_run': len(hist), 'best_epoch': best_epoch, 'best_val_l1': best,
        'total_train_seconds': total_s, 'seconds_per_epoch': total_s / len(hist),
        'peak_allocated_gib': peak, 'validation': agg, 'validation_baseline': baseline,
        'validation_per_sample': {k: v for k, v in ev['per'].items()},
        'validation_sample_ids': [r['sample_id'] for r in val_rows],
        'history': hist, 'figures': figs,
        'test_set': 'not evaluated (reserved for M5)',
        'results_dir': str(run),
    }
    m3.write_json(run / 'm4_train_summary.json', out)
    m3.write_json(cfg['metrics_path'], out)
    print('wrote', run / 'm4_train_summary.json', 'and', cfg['metrics_path'])
    return out


def main():
    cfg = m3.cfg_load(sys.argv[1])
    stage = sys.argv[2]
    manifest, splits = m3.load_splits(cfg)
    norm = m3.norm_from_cfg(cfg)
    assert torch.cuda.is_available(), 'CUDA required'
    configure_tf32(cfg['train']['tf32'])
    device = torch.device('cuda')
    seed_all(cfg['seed'])
    if stage == 'bench':
        stage_bench(cfg, splits, norm, device)
    elif stage == 'overfit':
        stage_overfit(cfg, splits, norm, device)
    elif stage == 'train':
        stage_train(cfg, splits, norm, device)
    else:
        raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
    main()

"""M3: frozen-split loading, train-only normalization, amplitude statistics, previews.

Read-only with respect to the M2 dataset. Every run output goes to a new timestamped
directory under E:/FNO_research; only the small JSON/PNG artifacts named in the config
are written into the repository.

Stages:
    norm   - fit the FMC normalization on the 375 training samples, write the JSON
    stats  - report normalized/raw/scattered FMC amplitudes per split
    viz    - draw the speed map and the tx-32 B-scan before and after normalization
    load   - benchmark on-demand HDF5 loading per batch
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SPLIT_ORDER = ['train', 'validation', 'test']


def cfg_load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def new_dir(base, prefix):
    """Create a fresh timestamped output directory (never reuses an existing one)."""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path(base) / f'{prefix}_{stamp}'
    for i in range(1, 1000):
        cand = out if i == 1 else Path(f'{out}_{i:02d}')
        try:
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        except FileExistsError:
            continue
    raise RuntimeError('cannot create output directory')


def write_json(path, obj):
    """Write a new JSON file; refuse to overwrite an existing one."""
    p = Path(path)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sample_path(cfg, sid):
    return Path(cfg['dataset_dir']) / 'samples' / (sid + '.h5')


def uniform_path(cfg):
    return Path(cfg['dataset_dir']) / 'samples' / 'uniform.h5'


def load_splits(cfg):
    """Load the frozen manifest and group rows by split without reselecting anything."""
    manifest = json.loads(Path(cfg['manifest_path']).read_text(encoding='utf-8-sig'))
    rows = manifest['samples']
    splits = {k: [] for k in SPLIT_ORDER}
    for r in rows:
        splits[r['split']].append(r)
    assert len(rows) == sum(cfg['split_counts'].values()), (len(rows), cfg['split_counts'])
    for k in SPLIT_ORDER:
        assert len(splits[k]) == cfg['split_counts'][k], (k, len(splits[k]))
        assert all(r['split'] == k for r in splits[k])
    ids = [r['sample_id'] for r in rows]
    assert len(set(ids)) == len(ids)
    return manifest, splits


def read_fmc(cfg, sid):
    with h5py.File(sample_path(cfg, sid), 'r') as f:
        return f['fmc'][:]


def fit_speed_norm(cfg):
    lo, hi = cfg['speed']['norm_range']
    cmin, cmax = cfg['speed']['min_ms'], cfg['speed']['max_ms']
    return {
        'min_ms': cmin,
        'max_ms': cmax,
        'norm_range': [lo, hi],
        'scale': (hi - lo) / (cmax - cmin),
        'offset': lo - cmin * (hi - lo) / (cmax - cmin),
        'formula': f'c_n = {lo} + (c_ms - {cmin}) * {hi - lo} / {cmax - cmin}',
        'fixed_physical_range': True,
        'fitted_from_data': False,
    }


def fit_fmc_norm(cfg, splits):
    """Global min/max over training samples only; validation/test reuse it."""
    lo, hi = cfg['fmc_norm_range']
    t0 = time.perf_counter()
    mn, mx, count = np.inf, -np.inf, 0
    for r in splits['train']:
        a = read_fmc(cfg, r['sample_id'])
        mn = min(mn, float(a.min()))
        mx = max(mx, float(a.max()))
        count += a.size
    dt = time.perf_counter() - t0
    scale = (hi - lo) / (mx - mn)
    return {
        'min': mn,
        'max': mx,
        'norm_range': [lo, hi],
        'scale': scale,
        'offset': lo - mn * scale,
        'formula': 'x_n = lo + (x - min) * (hi - lo) / (max - min)',
        'fit_split': 'train',
        'fit_samples': len(splits['train']),
        'fit_elements': count,
        'fit_seconds': dt,
        'read_mib_per_s': count * 4 / 2 ** 20 / dt,
        'leakage_guard': 'validation and test never contribute to these values',
    }, (mn, mx, scale, lo - mn * scale)


def read_reference(cfg):
    """Uniform-medium FMC used as the defect-free reference for the scattering input."""
    with h5py.File(Path(cfg['scatter']['reference']), 'r') as f:
        return f['fmc'][:]


def fit_scatter_norm(cfg, splits):
    """Train-set min/max of (sample FMC - uniform FMC); one affine map to [-1, 1]."""
    lo, hi = cfg['fmc_norm_range']
    hom = read_reference(cfg)
    t0 = time.perf_counter()
    mn, mx, count = np.inf, -np.inf, 0
    sq = 0.0
    for r in splits['train']:
        s = read_fmc(cfg, r['sample_id']) - hom
        mn = min(mn, float(s.min()))
        mx = max(mx, float(s.max()))
        sq += float(np.sum(s.astype(np.float64) ** 2))
        count += s.size
    dt = time.perf_counter() - t0
    scale = (hi - lo) / (mx - mn)
    return {
        'mode': 'scatter',
        'reference': cfg['scatter']['reference'],
        'reference_sha256': file_sha(cfg['scatter']['reference']),
        'input': 'FMC - uniform FMC',
        'min': mn,
        'max': mx,
        'raw_rms': float(np.sqrt(sq / count)),
        'norm_rms': float(np.sqrt(sq / count) * scale),
        'norm_range': [lo, hi],
        'scale': scale,
        'offset': lo - mn * scale,
        'formula': 'x_n = lo + (x - ref - min) * (hi - lo) / (max - min)',
        'fit_split': 'train',
        'fit_samples': len(splits['train']),
        'fit_elements': count,
        'fit_seconds': dt,
        'read_mib_per_s': count * 4 / 2 ** 20 / dt,
        'leakage_guard': 'validation and test never contribute to these values',
        'no_fixed_gain': 'the mapping comes only from train-set min/max; no arbitrary multiplier',
    }, (mn, mx, scale, lo - mn * scale)


def stage_norm(cfg, splits):
    fmc, (mn, mx, scale, off) = fit_fmc_norm(cfg, splits)
    norm = {
        'seed': cfg['seed'],
        'dataset_dir': cfg['dataset_dir'],
        'manifest_path': cfg['manifest_path'],
        'manifest_sha256': file_sha(cfg['manifest_path']),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'fmc': fmc,
        'speed': fit_speed_norm(cfg),
        'purpose': 'M3 baseline normalization exactly as specified; no per-sample or per-split statistics',
    }
    write_json(cfg['normalization_path'], norm)
    print(f'FMC raw min {mn:.12g} max {mx:.12g} scale {scale:.12g} offset {off:.12g}')
    print(f'fit on {fmc["fit_samples"]} train samples, {fmc["fit_elements"]:,} elements, '
          f'{fmc["fit_seconds"]:.2f}s, {fmc["read_mib_per_s"]:.0f} MiB/s')
    print('wrote', cfg['normalization_path'])
    return norm


def stage_norm_scatter(cfg, splits):
    fmc, (mn, mx, scale, off) = fit_scatter_norm(cfg, splits)
    norm = {
        'seed': cfg['seed'],
        'dataset_dir': cfg['dataset_dir'],
        'manifest_path': cfg['manifest_path'],
        'manifest_sha256': file_sha(cfg['manifest_path']),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'fmc': fmc,
        'speed': fit_speed_norm(cfg),
        'purpose': 'M4 input: defect-free reference subtracted first, then train-set min/max affine map',
    }
    write_json(cfg['scatter']['normalization_path'], norm)
    print(f'scatter raw min {mn:.12g} max {mx:.12g} scale {scale:.12g} offset {off:.12g}')
    print(f'scatter raw RMS {fmc["raw_rms"]:.6g} -> normalized RMS {fmc["norm_rms"]:.6g}')
    print(f'fit on {fmc["fit_samples"]} train samples, {fmc["fit_elements"]:,} elements, '
          f'{fmc["fit_seconds"]:.2f}s, {fmc["read_mib_per_s"]:.0f} MiB/s')
    print('wrote', cfg['scatter']['normalization_path'])
    return norm


def norm_from_cfg(cfg):
    sc = cfg.get('scatter')
    p = Path(sc['normalization_path']) if (sc and sc.get('enabled')) else Path(cfg['normalization_path'])
    if not p.exists():
        raise SystemExit(f'missing normalization file, run the matching norm stage first: {p}')
    norm = json.loads(p.read_text(encoding='utf-8-sig'))
    assert norm['fmc']['fit_split'] == 'train'
    assert norm['manifest_sha256'] == file_sha(cfg['manifest_path'])
    return norm


def stage_stats(cfg, splits, norm):
    """Amplitude statistics of the saved (un-normalized) FMC and its normalized copy."""
    scale, off = norm['fmc']['scale'], norm['fmc']['offset']
    hom = read_fmc(cfg, 'uniform') if uniform_path(cfg).exists() else None
    rows = []
    t0 = time.perf_counter()
    for split in SPLIT_ORDER:
        rec = dict(split=split, samples=len(splits[split]), raw_count=0, raw_sum_sq=0.0,
                   raw_min=np.inf, raw_max=-np.inf, norm_count=0, norm_sum_sq=0.0,
                   norm_min=np.inf, norm_max=-np.inf, scat_count=0, scat_sum_sq=0.0,
                   scat_norm_sum_sq=0.0, scat_max_abs=0.0, direct_count=0, direct_sum_sq=0.0)
        for r in splits[split]:
            a = read_fmc(cfg, r['sample_id']).astype(np.float64)
            rec['raw_count'] += a.size
            rec['raw_sum_sq'] += float(np.sum(a * a))
            rec['raw_min'] = min(rec['raw_min'], float(a.min()))
            rec['raw_max'] = max(rec['raw_max'], float(a.max()))
            n = a * scale + off
            rec['norm_count'] += n.size
            rec['norm_sum_sq'] += float(np.sum(n * n))
            rec['norm_min'] = min(rec['norm_min'], float(n.min()))
            rec['norm_max'] = max(rec['norm_max'], float(n.max()))
            if hom is not None:
                s = a - hom
                rec['scat_count'] += s.size
                rec['scat_sum_sq'] += float(np.sum(s * s))
                rec['scat_norm_sum_sq'] += float(np.sum((s * scale) ** 2))
                rec['scat_max_abs'] = max(rec['scat_max_abs'], float(np.abs(s).max()))
            # direct-arrival energy reference: same elements, but the uniform field itself
            if hom is not None:
                rec['direct_count'] += hom.size
                rec['direct_sum_sq'] += float(np.sum(hom.astype(np.float64) ** 2))
        rows.append(rec)
    elapsed = time.perf_counter() - t0

    def rms(sq, n):
        return float(np.sqrt(sq / n)) if n else float('nan')

    summary = {'per_split': {}, 'definitions': {
        'fmc_raw_rms': 'RMS of the saved FMC; the dataset is not normalized',
        'fmc_norm_rms': 'RMS after the train-fitted affine map to [-1, 1]',
        'scatter_rms': 'RMS of (sample FMC - uniform FMC); the stored data are not subtracted',
        'scatter_norm_rms': 'RMS of the same difference after the affine map (scale only)',
        'direct_rms': 'RMS of the uniform reference FMC, i.e. the direct-arrival level',
    }}
    tot = {}
    for rec in rows:
        s = {
            'samples': rec['samples'],
            'fmc_raw': {'min': rec['raw_min'], 'max': rec['raw_max'], 'rms': rms(rec['raw_sum_sq'], rec['raw_count'])},
            'fmc_norm': {'min': rec['norm_min'], 'max': rec['norm_max'], 'rms': rms(rec['norm_sum_sq'], rec['norm_count'])},
        }
        if rec['scat_count']:
            s['scatter'] = {'max_abs': rec['scat_max_abs'], 'rms': rms(rec['scat_sum_sq'], rec['scat_count']),
                            'norm_rms': rms(rec['scat_norm_sum_sq'], rec['scat_count']),
                            'ratio_to_direct': rms(rec['scat_sum_sq'], rec['scat_count']) / rms(rec['direct_sum_sq'], rec['direct_count'])}
            s['uniform_reference_rms'] = rms(rec['direct_sum_sq'], rec['direct_count'])
        summary['per_split'][rec['split']] = s
        for k, v in [('raw_sum_sq', rec['raw_sum_sq']), ('raw_count', rec['raw_count']),
                     ('norm_sum_sq', rec['norm_sum_sq']), ('norm_count', rec['norm_count']),
                     ('scat_sum_sq', rec['scat_sum_sq']), ('scat_count', rec['scat_count']),
                     ('scat_norm_sum_sq', rec['scat_norm_sum_sq']),
                     ('direct_sum_sq', rec['direct_sum_sq']), ('direct_count', rec['direct_count'])]:
            tot[k] = tot.get(k, 0) + rec[k]
    summary['all'] = {
        'samples': sum(r['samples'] for r in rows),
        'fmc_raw': {'rms': rms(tot['raw_sum_sq'], tot['raw_count'])},
        'fmc_norm': {'rms': rms(tot['norm_sum_sq'], tot['norm_count'])},
        'scatter': {'rms': rms(tot['scat_sum_sq'], tot['scat_count']),
                    'norm_rms': rms(tot['scat_norm_sum_sq'], tot['scat_count']),
                    'ratio_to_direct': rms(tot['scat_sum_sq'], tot['scat_count']) / rms(tot['direct_sum_sq'], tot['direct_count'])},
        'uniform_reference_rms': rms(tot['direct_sum_sq'], tot['direct_count']),
    }
    summary['normalization'] = norm['fmc']
    summary['speed_normalization'] = norm['speed']
    summary['stats_seconds'] = elapsed
    summary['note'] = ('The scatter rows subtract the single uniform reference FMC only to report amplitude '
                       'levels; no file is modified and the training input keeps the direct arrival.')
    write_json(cfg['stats_path'], summary)

    print(f'{"split":<11}{"raw rms":>13}{"norm rms":>13}{"scat rms":>13}{"scat/direct":>13}')
    for k in SPLIT_ORDER + ['all']:
        s = summary['per_split'].get(k, summary['all'])
        print(f'{k:<11}{s["fmc_raw"]["rms"]:>13.6g}{s["fmc_norm"]["rms"]:>13.6g}'
              f'{s["scatter"]["rms"]:>13.6g}{s["scatter"]["ratio_to_direct"]:>13.6g}')
    print(f'stats over {summary["all"]["samples"]} samples in {elapsed:.2f}s -> {cfg["stats_path"]}')
    return summary


class M2Dataset(Dataset):
    """On-demand reader: returns normalized FMC (64, 1000, 64) and speed (1, 200, 200)."""

    def __init__(self, cfg, rows, norm):
        self.cfg = cfg
        self.rows = rows
        self.sid = [r['sample_id'] for r in rows]
        self.scale = norm['fmc']['scale']
        self.offset = norm['fmc']['offset']
        self.mode = norm['fmc'].get('mode', 'full')
        self.cscale = norm['speed']['scale']
        self.coffset = norm['speed']['offset']
        self.fmc_shape = (cfg['model']['in_channels'], cfg['model']['t_in'], cfg['model']['r_in'])
        self.out_shape = (cfg['model']['out_channels'], cfg['model']['t_out'], cfg['model']['r_out'])
        # Reference field is loaded once per worker process, not per sample.
        self.hom = None
        if self.mode == 'scatter':
            self.hom = np.ascontiguousarray(read_reference(cfg).transpose(0, 2, 1), dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with h5py.File(sample_path(self.cfg, self.sid[i]), 'r') as f:
            fmc = f['fmc'][:]
            c200 = f['c_200'][:]
        # (tx, rx, t) -> (tx, t, rx): channels are transmitters, grid is T x R
        x = np.ascontiguousarray(fmc.transpose(0, 2, 1), dtype=np.float32)
        if self.hom is not None:
            x -= self.hom
        x *= np.float32(self.scale)
        x += np.float32(self.offset)
        y = np.ascontiguousarray(c200 * np.float32(self.cscale) + np.float32(self.coffset),
                                 dtype=np.float32)[None, :, :]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(cfg, rows, norm, shuffle, batch_size=None, num_workers=None, generator=None):
    lc = cfg['loader']
    nw = lc['num_workers'] if num_workers is None else num_workers
    return DataLoader(
        M2Dataset(cfg, rows, norm), batch_size=batch_size or lc['batch_size'], shuffle=shuffle,
        num_workers=nw,
        prefetch_factor=lc['prefetch_factor'] if nw > 0 else None,
        pin_memory=lc['pin_memory'], drop_last=False, persistent_workers=False,
        generator=generator)


def stage_load(cfg, splits, norm):
    """Benchmark on-demand loading: per-batch wall time and effective throughput."""
    out = {}
    generator = torch.Generator()
    generator.manual_seed(cfg['seed'])
    for split in ['train', 'validation']:
        lc = cfg['loader']
        ds = M2Dataset(cfg, splits[split], norm)
        t0 = time.perf_counter()
        x, y = ds[0]
        single = time.perf_counter() - t0
        assert tuple(x.shape) == ds.fmc_shape, (x.shape, ds.fmc_shape)
        assert tuple(y.shape) == ds.out_shape, (y.shape, ds.out_shape)
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        # Both maps come from training statistics; unseen validation samples may sit
        # slightly outside [-1, 1], which is recorded rather than clipped.
        xmin, xmax = float(x.min()), float(x.max())
        assert tuple(y.shape) == ds.out_shape
        assert float(y.min()) >= -1.0001 and float(y.max()) <= 1.0001
        loader = make_loader(cfg, splits[split], norm, shuffle=False, generator=generator)
        times = []
        t0 = time.perf_counter()
        nb = 0
        for xb, yb in loader:
            times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            nb += 1
        wall = sum(times)
        out[split] = {
            'batches': nb, 'batch_size': lc['batch_size'], 'num_workers': lc['num_workers'],
            'samples': len(ds), 'first_item_seconds': single,
            'input_min': xmin, 'input_max': xmax, 'input_mode': ds.mode,
            'input_rms': float(x.pow(2).mean().sqrt()),
            'batch_seconds': times, 'mean_batch_seconds': float(np.mean(times)),
            'first_batch_seconds': times[0], 'wall_seconds': wall,
            'samples_per_second': len(ds) / wall,
            'mib_per_second': len(ds) * (ds.fmc_shape[0] * ds.fmc_shape[1] * ds.fmc_shape[2] * 4) / 2 ** 20 / wall,
        }
        print(f'{split}: {nb} batches, first {times[0]:.3f}s, mean {np.mean(times):.3f}s, '
              f'{out[split]["samples_per_second"]:.1f} samples/s, {out[split]["mib_per_second"]:.0f} MiB/s')
    run = new_dir(cfg['results_root'], 'm3_load')
    write_json(run / 'm3_loader_benchmark.json', out)
    print('wrote', run / 'm3_loader_benchmark.json')
    return out


def stage_viz(cfg, splits, norm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pv = cfg['preview']
    rows = [r for k in SPLIT_ORDER for r in splits[k]]
    rng = np.random.default_rng(pv['seed'])
    picked = sorted(rng.choice(len(rows), size=pv['count'], replace=False).tolist())
    tx = pv['tx_one_based'] - 1
    hom = read_fmc(cfg, 'uniform')[tx].astype(np.float64)
    scale, off = norm['fmc']['scale'], norm['fmc']['offset']
    t_axis = np.arange(cfg['model']['t_in']) * 2e-08 * 1e6
    produced = []
    for i in picked:
        r = rows[i]
        with h5py.File(sample_path(cfg, r['sample_id']), 'r') as f:
            c200 = f['c_200'][:]
            raw = f['fmc'][tx].astype(np.float64)
        cn = c200 * norm['speed']['scale'] + norm['speed']['offset']
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        im = axes[0].imshow(c200, extent=[0, 40, 40, 0], vmin=1500, vmax=5800, cmap='viridis')
        fig.colorbar(im, ax=axes[0], label='m/s')
        axes[0].set(title=f"{r['sample_id']} {r['letter']} speed", xlabel='Lateral (mm)', ylabel='Depth (mm)')
        im = axes[1].imshow(cn, extent=[0, 40, 40, 0], vmin=-1, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=axes[1], label='normalized')
        axes[1].set(title='speed normalized [-1,1]', xlabel='Lateral (mm)')
        lim = float(np.abs(raw).max())
        im = axes[2].imshow(raw, extent=[t_axis[0], t_axis[-1], 64, 1], aspect='auto',
                           vmin=-lim, vmax=lim, cmap='seismic')
        fig.colorbar(im, ax=axes[2], label='raw pressure')
        axes[2].set(title=f'Tx {pv["tx_one_based"]} B-scan, raw', xlabel='Time (us)', ylabel='Receiver')
        xn = raw * scale + off
        im = axes[3].imshow(xn, extent=[t_axis[0], t_axis[-1], 64, 1], aspect='auto',
                            vmin=-1, vmax=1, cmap='seismic')
        fig.colorbar(im, ax=axes[3], label='normalized (full [-1,1])')
        axes[3].set(title=f'normalized, actual {float(xn.min()):.3g}..{float(xn.max()):.3g}',
                    xlabel='Time (us)', ylabel='Receiver')
        fig.suptitle('M3 normalization: affine map fitted on training min/max only; '
                     f'raw range {float(raw.min()):.4g}..{float(raw.max()):.4g}')
        fig.tight_layout()
        p = Path(cfg['figure_dir']) / f'm3_norm_{r["sample_id"]}.png'
        if p.exists():
            raise SystemExit(f'refusing to overwrite {p}')
        fig.savefig(p, dpi=pv['dpi'])
        plt.close(fig)
        produced.append(str(p))
        print('wrote', p)
    print('scatter check (display only): uniform tx32 max abs', float(np.abs(hom).max()))
    return produced


def main():
    cfg = cfg_load(sys.argv[1])
    stage = sys.argv[2]
    manifest, splits = load_splits(cfg)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    if stage == 'norm':
        stage_norm(cfg, splits)
    elif stage == 'norm_scatter':
        stage_norm_scatter(cfg, splits)
    elif stage == 'stats':
        stage_stats(cfg, splits, norm_from_cfg(cfg))
    elif stage == 'viz':
        stage_viz(cfg, splits, norm_from_cfg(cfg))
    elif stage == 'load':
        stage_load(cfg, splits, norm_from_cfg(cfg))
    else:
        raise SystemExit(f'unknown stage {stage}')


if __name__ == '__main__':
    main()

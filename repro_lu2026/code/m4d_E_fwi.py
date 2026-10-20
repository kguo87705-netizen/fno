"""M4d-E stage 3 (final): gated FWI initial-value pilot with a scattering-field loss.

Loss (all groups, all steps):
    r = [d_obs - F_obs(c_bg)] - [F_inv(c) - F_inv(c_bg)]      c_bg = uniform 5800 m/s
    loss = mean(r^2) over the 16 shots of the iteration, diagonal excluded
F_obs is the M2 Deepwave run (0.05 mm, 2.5 ns), F_inv the grid used by the inversion.
F_inv(c_bg) is computed once per grid and cached.

Optimisation: Adam on the velocity, projection to [1500, 5800] m/s after every step
(outside the autograd graph), 16 shots per iteration following one shared shot schedule.

Stages
    3a  forward mismatch at the true velocity, accuracy 4 and 8, all 64 shots, plus the
        training-mean-velocity initial residual
    3b  code check on the 0.05 mm generation grid (inverse crime), 10 iterations
    3c  common learning rate on one training sample, 3 rates x 3 inits x 20 iterations
    3d  one validation sample, A/B/C 20 iterations plus unrefined D
    3e  extension to the three displayed validation samples with the iteration budget
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

CODE = r'E:\FNO_research\repos\fno_light\repro_lu2026\code'
sys.path.insert(0, CODE)
import deepwave
from source_match import waves
from m2_common import write_new_json
import m3_data as m3
import m4_fno as m4

BASE = r'E:\FNO_research\repos\fno_light\repro_lu2026'
CFG = BASE + r'\configs\m4b_fno.json'
LOCAL_RUN = Path(r'E:\FNO_research\results\m4c_local_20261020_170451')
OUTROOT = Path(r'E:\FNO_research\results')
VAL = ['m2_000378', 'm2_000401', 'm2_000405']
TRAIN_SAMPLE = 'm2_000011'
F0 = 2.25e6
SHIFT = 1.3333333333333332e-6
GAIN = 0.9907679913977262
N_ELEM = 64
SHOTS_PER_ITER = 16
SCHEDULE_SEED = 20261020
BLOCKS = 25
GRID01 = {'name': '0.1 mm / 5 ns', 'dx': 1e-4, 'dt': 5e-9, 'nt': 4000, 'stride': 4,
          'pml': 20, 'z': 2, 'pitch': 6, 'fx': 11, 'rep': 2}
GRID005 = {'name': '0.05 mm / 2.5 ns', 'dx': 5e-5, 'dt': 2.5e-9, 'nt': 8000, 'stride': 8,
           'pml': 40, 'z': 4, 'pitch': 12, 'fx': 22, 'rep': 4}


def shot_schedule():
    g = np.random.default_rng(SCHEDULE_SEED)
    return [[g.permutation(N_ELEM)[k * SHOTS_PER_ITER:(k + 1) * SHOTS_PER_ITER].tolist()
             for k in range(4)] for _ in range(BLOCKS)]


def shots_for(schedule, it):
    return schedule[(it - 1) // 4][(it - 1) % 4]


def make_wave(g):
    q = {'dt_s': g['dt'], 'source_frequency_hz': F0, 'source_shift_s': SHIFT,
         'max_vel_m_s': 5800.0, 'dx_m': g['dx']}
    t = np.arange(g['nt']) * g['dt']
    a = np.pi * F0 * (t - SHIFT)
    r = (1 - 2 * a * a) * np.exp(-a * a)
    return waves(q, r)['forward_difference'] * (1e-4 / g['dx'])


def locations(g):
    return torch.tensor([[g['z'], g['fx'] + j * g['pitch']] for j in range(N_ELEM)],
                        dtype=torch.long, device='cuda')


def simulate(v, g, wave, acc, shots=None):
    loc = locations(g)
    sel = torch.arange(N_ELEM) if shots is None else torch.as_tensor(shots, device='cuda')
    n = len(sel)
    w = torch.as_tensor(wave, device='cuda')[None, None, :].repeat(n, 1, 1)
    res = deepwave.scalar(
        v, g['dx'], g['dt'], source_amplitudes=w,
        source_locations=loc[sel][:, None, :].contiguous(),
        receiver_locations=loc[None, :, :].repeat(n, 1, 1),
        accuracy=acc, pml_width=[g['pml']] * 4, pml_freq=F0, max_vel=5800.0,
        survey_pad=None, python_backend=False, storage_mode='none')
    parts = list(res) if isinstance(res, (tuple, list)) else [res]
    cands = [x for x in parts if hasattr(x, 'shape') and x.ndim == 3 and x.shape[-1] == g['nt']]
    assert cands, [getattr(x, 'shape', None) for x in parts]
    y = cands[0]
    return y[..., ::g['stride']].contiguous() * GAIN


def offdiag():
    m = torch.ones(N_ELEM, N_ELEM, dtype=torch.bool, device='cuda')
    m.fill_diagonal_(False)
    return m


def to_grid(v200, g):
    return torch.tensor(np.repeat(np.repeat(v200, g['rep'], 0), g['rep'], 1),
                        dtype=torch.float32, device='cuda')


def sample_data(sid):
    cfg = m3.cfg_load(CFG)
    with h5py.File(m3.sample_path(cfg, sid), 'r') as f:
        return f['fmc'][:].astype(np.float32), f['c_200'][:]


def uniform_obs():
    cfg = m3.cfg_load(CFG)
    with h5py.File(m3.uniform_path(cfg), 'r') as f:
        return f['fmc'][:].astype(np.float32)


def mean_map200():
    cfg = m3.cfg_load(CFG)
    _, splits = m3.load_splits(cfg)
    maps = []
    for r in splits['train']:
        with h5py.File(m3.sample_path(cfg, r['sample_id']), 'r') as f:
            maps.append(f['c_200'][:])
    return np.mean(np.stack(maps), 0).astype(np.float32)


def model_pred200(sid):
    cfg = m3.cfg_load(CFG)
    cfg['model'] = dict(cfg['model'])
    cfg['model']['mapping'] = 'local'
    norm = m3.norm_from_cfg(cfg)
    _, splits = m3.load_splits(cfg)
    rows = [r for sp in ['train', 'validation'] for r in splits[sp]]
    idx = [r['sample_id'] for r in rows].index(sid)
    ck = torch.load(LOCAL_RUN / 'best.pt', map_location='cuda', weights_only=False)
    model = m4.FNO2d(cfg['model']).to('cuda')
    model.load_state_dict(ck['model'])
    model.eval()
    ds = m3.M2Dataset(cfg, [rows[idx]], norm)
    x, _ = ds[0]
    with torch.no_grad():
        y = model(x[None].cuda())
    return np.clip(m4.to_ms(y[0, 0].cpu(), norm).numpy().astype(np.float32), 1500.0, 5800.0)


def vel_metrics(v400, truth200, g):
    pred = v400.detach().reshape(200, g['rep'], 200, g['rep']).mean(dim=(1, 3)).cpu().numpy()
    tm = truth200.astype(np.float64)
    d = pred - tm
    mask = tm < 5000.0
    inter = float(((pred < 5000) & mask).sum())
    return {'relative_l2': float(np.linalg.norm(d) / np.linalg.norm(tm)),
            'mae_ms': float(np.abs(d).mean()),
            'defect_mae_ms': float(np.abs(d)[mask].mean()) if mask.any() else None,
            'dice': inter / max(float((pred < 5000).sum() + mask.sum()), 1.0),
            'iou': inter / max(float(((pred < 5000) | mask).sum()), 1.0)}


def ed_all_shots(v, g, wave, acc, obs_scat, ref_bg):
    with torch.no_grad():
        sim = simulate(v.detach(), g, wave, acc)
        r = (sim - ref_bg) - obs_scat
        m = offdiag()
        return float(r[m].norm() / obs_scat[m].norm())


def fwi(v0, g, wave, acc, obs_scat, ref_bg, truth, iters, lr, schedule, tag, ckpts=(0, 5, 10, 20)):
    v = torch.nn.Parameter(v0.clone().detach())
    # The scattering MSE is ~1e-8, so its gradient is ~1e-13; Adam's default eps=1e-8 would
    # swamp the update. eps is a numerical detail here, not a scientific parameter.
    opt = torch.optim.Adam([v], lr=lr, eps=1e-16)
    m = offdiag()
    hist = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    init_met = vel_metrics(v, truth, g)
    ed0 = ed_all_shots(v, g, wave, acc, obs_scat, ref_bg)
    hist.append({'iteration': 0, 'E_d_all64': ed0, 'E_c_rel_l2': init_met['relative_l2'],
                 'mae_ms': init_met['mae_ms'], 'defect_mae_ms': init_met['defect_mae_ms'],
                 'dice': init_met['dice'], 'iou': init_met['iou'], 'seconds': 0.0,
                 'shots': 'all 64'})
    print(f'   {tag} it 0 E_d {ed0:.4f} E_c {init_met["relative_l2"]:.4f}', flush=True)
    for it in range(1, iters + 1):
        sh = shots_for(schedule, it)
        opt.zero_grad(set_to_none=True)
        sim = simulate(v, g, wave, acc, sh)
        sub_obs = obs_scat[sh]
        sub_ref = ref_bg[sh]
        # diagonal = pairs whose transmitter is the same element as the receiver;
        # for a shot subset the mask is (n_shots, 64), not (64, 64).
        sh_t = torch.as_tensor(sh, device='cuda')
        mask_sub = sh_t[:, None] != torch.arange(N_ELEM, device='cuda')[None, :]
        loss = ((((sim - sub_ref) - sub_obs) ** 2)[mask_sub]).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v.clamp_(1500.0, 5800.0)
        met = vel_metrics(v, truth, g)
        rec = {'iteration': it, 'train16_loss': float(loss.detach()), 'E_c_rel_l2': met['relative_l2'],
               'seconds': time.perf_counter() - t0, 'shots': sh}
        if it in ckpts or (it % 20 == 0):
            rec['E_d_all64'] = ed_all_shots(v, g, wave, acc, obs_scat, ref_bg)
            rec.update({k: met[k] for k in ['mae_ms', 'defect_mae_ms', 'dice', 'iou']})
            print(f'   {tag} it {it:3d} E_d {rec["E_d_all64"]:.4f} E_c {met["relative_l2"]:.4f} '
                  f'dice {met["dice"]:.3f} ({rec["seconds"]:.0f}s)', flush=True)
        hist.append(rec)
    per_it = (time.perf_counter() - t0) / iters
    return v.detach(), hist, per_it, torch.cuda.max_memory_allocated() / 2 ** 30


def stage_3a():
    g = GRID01
    wave = make_wave(g)
    sid = VAL[0]
    obs, truth = sample_data(sid)
    obs = torch.tensor(obs, device='cuda')
    obs_scat = obs - torch.tensor(uniform_obs(), device='cuda')
    m = offdiag()
    res = {'sample': sid, 'grid': g['name'], 'source_max_abs': float(np.abs(wave).max()),
           'definitions': 'E_scat and E_total are forward mismatch at the true velocity, not an FWI '
                          'residual floor; denominator is the observed scattering norm; diagonal excluded',
           'trials': {}}
    for acc in [4, 8]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_bg = simulate(torch.full((400, 400), 5800.0, device='cuda'), g, wave, acc)
        torch.cuda.synchronize()
        t_bg = time.perf_counter() - t0
        t0 = time.perf_counter()
        sim_true = simulate(to_grid(truth, g), g, wave, acc)
        torch.cuda.synchronize()
        t_true = time.perf_counter() - t0
        sim_mean = simulate(to_grid(mean_map200(), g), g, wave, acc)
        e = {'forward_seconds_64shots_true': t_true, 'forward_seconds_64shots_uniform': t_bg,
             'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
             'E_scat_true_velocity': float(((sim_true - ref_bg)[m]
                                            - obs_scat[m]).norm() / obs_scat[m].norm()),
             'E_total_true_velocity': float((sim_true[m] - obs[m]).norm()
                                            / obs_scat[m].norm()),
             'E_scat_mean_velocity': float(((sim_mean - ref_bg)[m]
                                            - obs_scat[m]).norm() / obs_scat[m].norm())}
        res['trials'][f'accuracy{acc}'] = e
        print(f'acc {acc}', json.dumps(e, indent=1), flush=True)
    best = min(res['trials'].items(), key=lambda kv: kv[1]['E_scat_true_velocity'])
    res['chosen_accuracy'] = int(best[0].replace('accuracy', ''))
    gate = {'E_scat_true_velocity': best[1]['E_scat_true_velocity'],
            'E_total_true_velocity': best[1]['E_total_true_velocity'],
            'initial_residual_mean_velocity': best[1]['E_scat_mean_velocity']}
    gate['ratio_initial_to_forward_mismatch'] = (gate['initial_residual_mean_velocity']
                                                 / gate['E_scat_true_velocity'])
    res['gate'] = gate
    if gate['E_scat_true_velocity'] <= 0.10:
        res['main_grid'] = '0.1 mm'
    elif gate['E_scat_true_velocity'] <= 0.25:
        res['main_grid'] = '0.1 mm (usable; report the forward mismatch and its ratio to the initial residual)'
    else:
        res['main_grid'] = '0.05 mm (same-grid inversion, 0.1 mm forward mismatch > 0.25)'
    run = m3.new_dir(OUTROOT, 'm4d_E_fwi_3a')
    write_new_json(run / 'grid_floor.json', res)
    print('GATE', json.dumps(gate, indent=1))
    print('main grid:', res['main_grid'])
    print('wrote', run / 'grid_floor.json')





def stage_3b():
    """Inverse-crime FWI on the generation grid: code correctness only."""
    g = GRID005
    wave = make_wave(g)
    acc = 8
    sid = VAL[0]
    obs, truth = sample_data(sid)
    obs = torch.tensor(obs, device='cuda')
    obs_scat = obs - torch.tensor(uniform_obs(), device='cuda')
    n = 800
    ref_bg = simulate(torch.full((n, n), 5800.0, device='cuda'), g, wave, acc)
    v0 = to_grid(mean_map200(), g)
    v, hist, per_it, peak = fwi(v0, g, wave, acc, obs_scat, ref_bg, truth, 10, 10.0,
                               shot_schedule(), '3b', ckpts=(0, 5, 10))
    ed = [(h['iteration'], h['E_d_all64']) for h in hist if 'E_d_all64' in h]
    ec = [(h['iteration'], h['E_c_rel_l2']) for h in hist]
    ok_loss = all(ed[i + 1][1] <= ed[i][1] for i in range(len(ed) - 1))
    ok_vel = ec[-1][1] < ec[0][1]
    out = {'sample': sid, 'grid': g['name'], 'accuracy': acc, 'lr': 10.0, 'iters': 10,
           'per_iteration_s': per_it, 'peak_allocated_gib': peak,
           'E_d_all64': ed, 'E_c': ec, 'loss_decreased': ok_loss,
           'velocity_error_decreased': ok_vel,
           'note': 'inverse crime: F_inv is the generation grid, code check only'}
    run = m3.new_dir(OUTROOT, 'm4d_E_fwi_3b')
    write_new_json(run / 'inverse_crime_check.json', out)
    print('3b', json.dumps({k: out[k] for k in ['per_iteration_s', 'peak_allocated_gib',
                                                'loss_decreased', 'velocity_error_decreased']}))
    print('wrote', run / 'inverse_crime_check.json')
    if not (ok_loss and ok_vel):
        raise SystemExit('3b FAILED: stop and report')


def stage_3c():
    g = GRID01
    wave = make_wave(g)
    sid = TRAIN_SAMPLE
    obs, truth = sample_data(sid)
    obs = torch.tensor(obs, device='cuda')
    obs_scat = obs - torch.tensor(uniform_obs(), device='cuda')
    ref_bg = simulate(torch.full((400, 400), 5800.0, device='cuda'), g, wave, 8)
    inits = {'A_uniform': torch.full((400, 400), 5800.0, device='cuda'),
             'B_mean_image': to_grid(mean_map200(), g),
             'C_fno_local': to_grid(model_pred200(sid), g)}
    sched = shot_schedule()
    out = {'sample': sid, 'grid': g['name'], 'accuracy': 8, 'iters': 20, 'trials': {}}
    for lr in [1, 10, 100]:
        for name, v0 in inits.items():
            v, hist, per_it, peak = fwi(v0, g, wave, 8, obs_scat, ref_bg, truth, 20, lr, sched,
                                        f'3c {lr:g} {name}', ckpts=(0, 20))
            e0 = hist[0]['E_c_rel_l2']
            e20 = hist[-1]['E_c_rel_l2']
            diverged = bool(not np.isfinite(e20) or e20 > e0)
            out['trials'][f'{lr:g}|{name}'] = {
                'lr': lr, 'init': name, 'E_c_0': e0, 'E_c_20': e20,
                'ratio': (e20 / e0) if e0 else None, 'diverged': diverged,
                'E_d_0': hist[0]['E_d_all64'], 'E_d_20': hist[-1].get('E_d_all64'),
                'per_iteration_s': per_it, 'peak_allocated_gib': peak}
            print(f'lr {lr:g} {name}: E_c {e0:.4f} -> {e20:.4f} ratio {e20/e0:.4f} '
                  f'diverged {diverged}', flush=True)
    # Selection: drop rates where any init diverged, then minimise the worst-group ratio.
    ok = {}
    for lr in [1, 10, 100]:
        rows = [v for k, v in out['trials'].items() if v['lr'] == lr]
        if any(r['diverged'] for r in rows):
            continue
        ok[lr] = max(r['ratio'] for r in rows)
    if not ok:
        raise SystemExit('3c FAILED: every learning rate diverged')
    chosen = min(ok, key=ok.get)
    out['chosen_lr'] = chosen
    out['worst_group_ratio'] = {f'{k:g}': v for k, v in ok.items()}
    out['per_iteration_s'] = out['trials'][f'{chosen:g}|B_mean_image']['per_iteration_s']
    run = m3.new_dir(OUTROOT, 'm4d_E_fwi_3c')
    write_new_json(run / 'lr_choice.json', out)
    print('chosen lr', chosen, 'worst-group ratios', {f'{k:g}': round(v, 4) for k, v in ok.items()})
    print('wrote', run / 'lr_choice.json')


def stage_3d(iters, lr):
    g = GRID01
    wave = make_wave(g)
    acc = 8
    sid = VAL[0]
    obs, truth = sample_data(sid)
    obs = torch.tensor(obs, device='cuda')
    obs_scat = obs - torch.tensor(uniform_obs(), device='cuda')
    ref_bg = simulate(torch.full((400, 400), 5800.0, device='cuda'), g, wave, acc)
    fno = model_pred200(sid)
    inits = {'A_uniform': torch.full((400, 400), 5800.0, device='cuda'),
             'B_mean_image': to_grid(mean_map200(), g),
             'C_fno_local': to_grid(fno, g)}
    sched = shot_schedule()
    out = {'sample': sid, 'grid': g['name'], 'accuracy': acc, 'iters': iters, 'lr': lr,
           'groups': {}}
    for name, v0 in inits.items():
        v, hist, per_it, peak = fwi(v0, g, wave, acc, obs_scat, ref_bg, truth, iters, lr, sched,
                                    name, ckpts=(0, 5, 10, 20))
        out['groups'][name] = {'history': hist, 'per_iteration_s': per_it,
                               'peak_allocated_gib': peak}
    d = vel_metrics(to_grid(fno, g), truth, g)
    out['groups']['D_no_refinement'] = {'initial': d, 'note': 'FNO prediction, no FWI'}
    run = m3.new_dir(OUTROOT, 'm4d_E_fwi_3d')
    write_new_json(run / 'single_sample.json', out)
    print('wrote', run / 'single_sample.json')


def stage_3e(iters, lr):
    g = GRID01
    wave = make_wave(g)
    acc = 8
    ref_bg = simulate(torch.full((400, 400), 5800.0, device='cuda'), g, wave, acc)
    mean_grid = to_grid(mean_map200(), g)
    sched = shot_schedule()
    out = {'grid': g['name'], 'accuracy': acc, 'iters': iters, 'lr': lr, 'samples': {}}
    for sid in VAL:
        obs, truth = sample_data(sid)
        obs = torch.tensor(obs, device='cuda')
        obs_scat = obs - torch.tensor(uniform_obs(), device='cuda')
        fno = model_pred200(sid)
        entry = {'groups': {}}
        for name, v0 in [('A_uniform', torch.full((400, 400), 5800.0, device='cuda')),
                         ('B_mean_image', mean_grid),
                         ('C_fno_local', to_grid(fno, g))]:
            v, hist, per_it, peak = fwi(v0, g, wave, acc, obs_scat, ref_bg, truth, iters, lr,
                                        sched, f'3e {sid} {name}', ckpts=(0, 5, 10, 20))
            entry['groups'][name] = {'history': hist, 'per_iteration_s': per_it,
                                     'peak_allocated_gib': peak}
        entry['groups']['D_no_refinement'] = {'initial': vel_metrics(to_grid(fno, g), truth, g)}
        out['samples'][sid] = entry
        run = m3.new_dir(OUTROOT, 'm4d_E_fwi_3e')
        write_new_json(run / f'{sid}.json', out)
        print('wrote', run / f'{sid}.json')


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    st = sys.argv[1]
    if st == '3a':
        stage_3a()
    elif st == '3b':
        stage_3b()
    elif st == '3c':
        stage_3c()
    elif st == '3d':
        stage_3d(int(sys.argv[2]), float(sys.argv[3]))
    elif st == '3e':
        stage_3e(int(sys.argv[2]), float(sys.argv[3]))
    else:
        raise SystemExit('unknown stage')

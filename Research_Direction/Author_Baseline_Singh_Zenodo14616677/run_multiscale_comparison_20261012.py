"""Authorized two-cutoff, 15 low-pass + 20 full-band author FWI comparison."""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
import time
import json
import threading
from pathlib import Path
import numpy as np
import torch
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = Path(__file__).resolve().parent / 'singh_zenodo/singh_zenodo'
OUT = ROOT / '.build/singh-multiscale-author-20261012'
ACTIVE_CUTOFF = None
CACHE = {}

def lowpass(x, dt):
    if ACTIVE_CUTOFF is None:
        return x
    # Zero-extend each trace, multiply by a real even response, then crop.
    # This linear operator is self-adjoint: L = P^T F^-1 H F P.
    nfft = 1 << (4*x.shape[-1]-1).bit_length()
    key = (ACTIVE_CUTOFF, dt, nfft, str(x.device), x.dtype)
    if key not in CACHE:
        freq = np.fft.rfftfreq(nfft, dt)
        sos = signal.butter(4, ACTIVE_CUTOFF, fs=1/dt, output='sos')
        _, response = signal.sosfreqz(sos, worN=freq, fs=1/dt)
        CACHE[key] = torch.as_tensor(abs(response)**2, dtype=x.dtype, device=x.device)
    return torch.fft.irfft(torch.fft.rfft(x, n=nfft, dim=-1)*CACHE[key], n=nfft, dim=-1)[..., :x.shape[-1]]

def before_epoch(ns):
    global ACTIVE_CUTOFF
    ACTIVE_CUTOFF = ns['cutoff_hz'] if ns['epoch'] < 15 else None
    if time.time() >= DEADLINE:
        raise TimeoutError('30-minute computation budget exhausted')

def finish(ns):
    torch.cuda.synchronize()
    seconds = time.perf_counter()-ns['start0']
    pred = ns['gammaPred'].detach().cpu().numpy().squeeze()
    truth = ns['gamma'].cpu().numpy().squeeze()
    error = (pred.astype(float)-truth.astype(float))**2
    assert np.isfinite(pred).all() and np.isfinite(ns['historyCost']).all()
    assert len(ns['historyCost']) == 35
    assert pred.min() >= ns['gamma0']*.999 and pred.max() <= 1
    row = dict(case=int(ns['case']), cutoff_hz=ns['cutoff_hz'], epochs=35,
        lowpass_steps=15, fullband_steps=20, material_mse=float(error[1:-1,1:-1].mean()),
        inner_mse=float(error[2:-2,2:-2].mean()), fit_seconds=seconds,
        last_preupdate_author_cost=float(ns['historyCost'][-1]))
    folder = Path(ns['outputFolder'])
    np.savez_compressed(folder/'final_fields.npz', truth=truth, reconstruction=pred,
        history_cost=ns['historyCost'], history_gamma=ns['historyGamma'])
    ns['final_metrics'] = row
    print('FINAL '+json.dumps(row), flush=True)

if __name__ == '__main__':
    torch.set_num_threads(4)
    sys.path.insert(0, str(AUTHOR))
    import AdjointMethod
    adjoint = (AUTHOR/'AdjointMethod.py').read_text(encoding='utf-8-sig')
    anchor = '    # cost function\n'
    assert anchor in adjoint
    adjoint = adjoint.replace(anchor,
        '    residual = lowpass(U[:, selx, sely, 1:], dt) - lowpass(um, dt)\n\n'+anchor, 1)
    assert adjoint.count('(U[:, selx, sely, 1:] - um) ** 2') == 1
    adjoint = adjoint.replace('(U[:, selx, sely, 1:] - um) ** 2', 'residual ** 2')
    assert adjoint.count('-(U[:, selx, sely,1:] - um)') == 1
    adjoint = adjoint.replace('-(U[:, selx, sely,1:] - um)', '-lowpass(residual, dt)')
    AdjointMethod.__dict__['lowpass'] = lowpass
    exec(compile(adjoint, str(AUTHOR/'AdjointMethod.py'), 'exec'), AdjointMethod.__dict__)
    source = (AUTHOR/'conventionalFWI.py').read_text(encoding='utf-8-sig')
    source = source.replace('cases = np.linspace(1, 4, 4, dtype=int)', 'cases = np.array([selected_case], dtype=int)')
    source = source.replace('destinationFolder = "data_casestudy"', 'destinationFolder = outputFolder')
    source = source.replace('if destinationFolder not in os.listdir(os.getcwd()):', 'if not os.path.isdir(destinationFolder):')
    for filename in ('material', 'measurement', 'source.h5'):
        source = source.replace(f'destinationFolder + "/{filename}', f'inputFolder + "/{filename}')
    source = source.replace('    for epoch in range(epochs):', '    for epoch in range(epochs):\n        before_epoch(globals())')
    source = source[:source.index('    # Plotting and saving figures over iterations')]+'    finish(globals())\n'
    OUT.mkdir(parents=True, exist_ok=True)
    # Small filter-only checks: no model fitting or additional solver calls.
    ACTIVE_CUTOFF = 250000
    a = torch.randn(2, 2000)
    b = torch.randn(2, 2000)
    assert torch.allclose((lowpass(a, 3e-8)*b).sum(), (a*lowpass(b, 3e-8)).sum(), atol=2e-5, rtol=2e-5)
    ACTIVE_CUTOFF = None
    assert lowpass(a, 3e-8) is a
    START = time.time()
    DEADLINE = START+1800
    def timeout():
        print('STOP: 30-minute computation deadline', flush=True)
        os._exit(124)
    watchdog = threading.Timer(1800, timeout)
    watchdog.daemon = True
    watchdog.start()
    results = []
    failures = 0
    for cutoff in [250000, 350000]:
        for case in range(1,5):
            if DEADLINE-time.time() < 160:
                print('STOP: insufficient remaining budget', flush=True)
                sys.exit(124)
            folder = OUT/f'fc{cutoff//1000}k'/f'case{case}'
            folder.mkdir(parents=True, exist_ok=True)
            ns = dict(__file__=str(AUTHOR/'conventionalFWI.py'), __name__='__main__',
                selected_case=case, cutoff_hz=cutoff, inputFolder=str(AUTHOR/'data_casestudy'),
                outputFolder=str(folder), before_epoch=before_epoch, finish=finish)
            print(f'RUN fc={cutoff} case={case} remaining={DEADLINE-time.time():.1f}s', flush=True)
            case_start = time.perf_counter()
            try:
                exec(compile(source, ns['__file__'], 'exec'), ns)
            except Exception as exc:
                failures += 1
                print(f'ERROR {failures}: {type(exc).__name__}: {exc}', flush=True)
                # Do not skip a failed case and silently compare incomplete groups.
                raise
            row = ns['final_metrics']
            row['case_wall_seconds'] = time.perf_counter()-case_start
            results.append(row)
            (folder/'metrics.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
            (OUT/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print('DONE '+json.dumps(row), flush=True)
            del ns
            torch.cuda.empty_cache()
    print(f'TOTAL_COMPUTE_SECONDS {time.time()-START:.3f}', flush=True)

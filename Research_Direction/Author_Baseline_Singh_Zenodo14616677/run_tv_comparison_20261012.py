"""One authorized three-weight TV comparison, retaining author update statements."""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import json
import argparse
from pathlib import Path
import sys
import time
import threading
import numpy as np
import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = Path(__file__).resolve().parent / 'singh_zenodo/singh_zenodo'
OUT = ROOT / '.build/singh-tv-author-20261012'
EPS = 1e-3
START = None
DEADLINE = None
RESULTS = []

def smooth_tv(field):
    material = field[..., 1:-1, 1:-1]
    gx = functional.pad(material[..., 1:, :] - material[..., :-1, :], (0, 0, 0, 1))
    gy = functional.pad(material[..., :, 1:] - material[..., :, :-1], (0, 1, 0, 0))
    return torch.sqrt(gx.square() + gy.square() + EPS ** 2).sum()

def add_tv(ns):
    if time.time() >= DEADLINE:
        raise TimeoutError('30 minute computation budget reached')
    if ns['epoch'] == 0:
        ns['initial_data_cost'] = float(ns['cost'].detach().cpu())
        ns['tv_lambda'] = ns['k_value'] * ns['initial_data_cost'] / TV_REF
    if not torch.isfinite(ns['cost']) or not torch.isfinite(ns['gradient']).all():
        raise RuntimeError('Nonfinite author data cost or gradient')
    if ns['k_value']:
        penalty = smooth_tv(ns['gammaPred'])
        tv_gradient, = torch.autograd.grad(penalty, ns['gammaPred'])
        ns['gammaPred'].grad.add_(tv_gradient * (ns['tv_lambda'] * ns['costScaling']))

def finish(ns):
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - ns['start0']
    final = ns['gammaPred'].detach()
    with torch.no_grad():
        wave = ns['forwardSolver'].forwardNSteps(ns['u0'].clone(), ns['u1'].clone(),
            final, ns['F'], ns['Nx'], ns['Ny'], ns['N'], ns['numberOfSources'], ns['device'])
        residual = wave[:, ns['selx'], ns['sely'], 1:] - ns['um'][0].to(ns['device'])
        cost = .5 * ns['dx'] * ns['dy'] * torch.trapezoid(residual.square(), dx=ns['dt'], dim=-1).sum() / ns['numberOfSources']
    prediction = final.cpu().numpy().squeeze()
    truth = ns['gamma'].cpu().numpy().squeeze()
    error = (prediction.astype(np.float64) - truth.astype(np.float64)) ** 2
    row = dict(case=int(ns['case']), k=ns['k_value'], tv_lambda=ns['tv_lambda'],
        J0=ns['initial_data_cost'], TV_ref=TV_REF, material_mse=float(error[1:-1, 1:-1].mean()),
        inner_mse=float(error[2:-2, 2:-2].mean()), final_time_integral_cost=float(cost.cpu()),
        final_tv=float(smooth_tv(final).cpu()), fit_seconds=fit_seconds)
    if not np.isfinite(list(row.values())).all():
        raise RuntimeError('Nonfinite final result')
    folder = Path(ns['outputFolder'])
    np.savez_compressed(folder / 'final_fields.npz', truth=truth, reconstruction=prediction)
    if ns['k_value'] == 0:
        old = torch.load(ROOT / f'.build/author-singh-case{int(ns["case"])}-35epochs-20261008/fields.pt', weights_only=True)
        row['max_difference_from_historical'] = float(np.max(np.abs(prediction - old['reconstruction'].numpy().squeeze())))
    ns['final_metrics'] = row
    print('FINAL ' + json.dumps(row), flush=True)

def main():
    global START, TV_REF
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', nargs='+', type=float, choices=(.1, 1., 10.), default=[1.])
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    global DEADLINE
    DEADLINE = args.deadline if args.deadline is not None else time.time() + 1800
    def stop_at_deadline():
        print('STOP: round 30-minute deadline reached', flush=True)
        os._exit(124)
    watchdog = threading.Timer(max(0, DEADLINE - time.time()), stop_at_deadline)
    watchdog.daemon = True
    watchdog.start()
    torch.set_num_threads(4)
    sys.path.insert(0, str(AUTHOR))
    reference = []
    for case in range(1, 5):
        saved = torch.load(ROOT / f'.build/author-singh-case{case}-35epochs-20261008/fields.pt', weights_only=True)
        reference.append(float(smooth_tv(saved['reconstruction'].double())))
    TV_REF = float(np.mean(reference))
    OUT.mkdir(parents=True, exist_ok=True)
    print('TV_REF ' + json.dumps(dict(per_case=reference, mean=TV_REF)), flush=True)
    source = (AUTHOR / 'conventionalFWI.py').read_text(encoding='utf-8-sig')
    source = source.replace('cases = np.linspace(1, 4, 4, dtype=int)', 'cases = np.array([selected_case], dtype=int)')
    source = source.replace('destinationFolder = "data_casestudy"', 'destinationFolder = outputFolder')
    source = source.replace('if destinationFolder not in os.listdir(os.getcwd()):', 'if not os.path.isdir(destinationFolder):')
    for filename in ('material', 'measurement', 'source.h5'):
        source = source.replace(f'destinationFolder + "/{filename}', f'inputFolder + "/{filename}')
    source = source.replace('        # setting the gradient at the senors', '        add_tv(globals())\n\n        # setting the gradient at the senors')
    source = source.replace('    # Plotting and saving figures over iterations', '    finish(globals())\n\n    # Plotting and saving figures over iterations')
    START = time.perf_counter()
    # The middle weight runs first, so a budget problem leaves its full four cases.
    jobs = ([(0., 1)] if args.timing else []) + [(k, c) for k in args.k for c in range(1, 5)]
    results_path = OUT / 'results.json'
    if results_path.exists():
        RESULTS.extend(json.loads(results_path.read_text(encoding='utf-8')))
    for k, case in jobs:
        if any(r['k'] == k and r['case'] == case for r in RESULTS):
            print(f'SKIP completed k={k:g} case={case}', flush=True)
            continue
        elapsed = time.perf_counter() - START
        if DEADLINE - time.time() < 100:
            print('STOP insufficient remaining budget', flush=True)
            break
        folder = OUT / (f'k{k:g}' if k else 'conventional_timing') / f'case{case}'
        folder.mkdir(parents=True, exist_ok=True)
        print(f'RUN k={k:g} case={case} elapsed={elapsed:.1f}s', flush=True)
        ns = dict(__file__=str(AUTHOR / 'conventionalFWI.py'), __name__='__main__',
            selected_case=case, k_value=k, inputFolder=str(AUTHOR / 'data_casestudy'),
            outputFolder=str(folder), add_tv=add_tv, finish=finish)
        case_start = time.perf_counter()
        exec(compile(source, ns['__file__'], 'exec'), ns)
        row = ns['final_metrics']
        row['case_wall_seconds'] = time.perf_counter() - case_start
        (folder / 'metrics.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        RESULTS.append(row)
        (OUT / 'results.json').write_text(json.dumps(RESULTS, indent=2), encoding='utf-8')
        print('DONE ' + json.dumps(row), flush=True)
        del ns
        torch.cuda.empty_cache()
    print(f'TOTAL_COMPUTE_SECONDS {time.perf_counter() - START:.3f}', flush=True)

if __name__ == '__main__':
    main()

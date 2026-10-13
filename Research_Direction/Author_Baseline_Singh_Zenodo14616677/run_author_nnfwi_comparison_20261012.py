"""Run released NN FWI without changing its optimizer or model settings."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = Path(__file__).resolve().parent / 'singh_zenodo' / 'singh_zenodo'
OUT = ROOT / '.build' / 'singh-nnfwi-author-20261012'

def finish_training(ns):
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - ns['start0']
    with torch.no_grad():
        final = torch.ones_like(ns['gammaPred'])
        final[:, :, 2:-2, 2:-2] = ns['model'](ns['inputData'])
        wave = ns['forwardSolver'].forwardNSteps(
            ns['u0'].clone(), ns['u1'].clone(), final, ns['F'],
            ns['Nx'], ns['Ny'], ns['N'], ns['numberOfSources'], ns['device'])
        residual = wave[:, ns['selx'], ns['sely'], 1:] - ns['um'][0].to(ns['device'])
        cost = .5 * ns['dx'] * ns['dy'] * torch.trapezoid(
            residual.square(), dx=ns['dt'], dim=-1).sum() / ns['numberOfSources']
    prediction = final[0, 0].cpu().numpy()
    truth = ns['gamma'][0, 0].cpu().numpy()
    error = (prediction.astype(np.float64) - truth.astype(np.float64)) ** 2
    metrics = dict(case=int(ns['case']), updates=35, initial_lr=.001,
                   material_mse=float(error[1:-1, 1:-1].mean()),
                   inner_mse=float(error[2:-2, 2:-2].mean()),
                   final_time_integral_cost=float(cost.cpu()), fit_seconds=fit_seconds)
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError('Nonfinite final result')
    folder = Path(ns['outputFolder'])
    np.savez_compressed(folder / 'final_fields.npz', truth=truth, reconstruction=prediction)
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('FINAL ' + json.dumps(metrics), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True, choices=range(1, 5))
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = OUT / f'case{args.case}'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(AUTHOR))
    source = (AUTHOR / 'NNbasedFWI.py').read_text(encoding='utf-8-sig')
    source = source.replace('cases = np.linspace(1, 4, 4, dtype=int)', f'cases = np.array([{args.case}], dtype=int)')
    source = source.replace('destinationFolder = "data_casestudy"', 'destinationFolder = outputFolder')
    source = source.replace('if destinationFolder not in os.listdir(os.getcwd()):', 'if not os.path.isdir(destinationFolder):')
    for filename in ('material', 'gradient', 'measurement', 'source.h5'):
        source = source.replace(f'destinationFolder + "/{filename}', f'inputFolder + "/{filename}')
    source = source.replace('    # Plotting and saving figures over iterations', '    finish_training(globals())\n\n    # Plotting and saving figures over iterations')
    namespace = dict(__file__=str(AUTHOR / 'NNbasedFWI.py'), __name__='__main__',
                     inputFolder=str(AUTHOR / 'data_casestudy'), outputFolder=str(output),
                     finish_training=finish_training)
    started = time.perf_counter()
    exec(compile(source, namespace['__file__'], 'exec'), namespace)
    print(f'CASE_WALL_SECONDS {time.perf_counter() - started:.3f}', flush=True)

if __name__ == '__main__':
    main()

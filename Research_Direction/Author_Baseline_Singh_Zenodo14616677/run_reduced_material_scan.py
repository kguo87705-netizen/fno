"""Nine preregistered contrasts with known case1 geometry; no optimization."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import pandas as pd
import torch
import FiniteDifferencePyTorchConv as FD
import Utilities
from discrete_reference import DiscreteReference, time_cost, variable_mask


ROOT = Path(__file__).resolve().parent
PRIOR = Path('.build/author-observation-control-20261008')
OUT = Path('.build/author-reduced-material-20261008')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = pd.read_csv(ROOT / 'settings2D.csv').iloc[0]
    nx, ny, n, ns = (int(cfg[k]) for k in ['Nx', 'Ny', 'N', 'numberOfSources'])
    dx, dy, dt = float(cfg['Lx']/nx), float(cfg['Ly']/ny), float(cfg['dt'])
    solver = FD.FiniteDifference(dt, dx, dy, float(cfg['c']), float(cfg['rho']), device)
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg['distanceBetweenSources']), ns)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg['distanceBetweenSensors']), sx)
    ref = DiscreteReference(solver, rx, ry)
    force = torch.tensor(pd.read_hdf(ROOT/'data_casestudy/source.h5').values).reshape(ns, 1, nx+3, ny+3, n).to(device)
    frozen = torch.load(PRIOR/'frozen_observations.pt', weights_only=True, map_location=device)
    return ref, force, frozen, (dx, dy, dt), variable_mask(frozen['truth'], rx, ry, sx, sy)


def metrics(gamma, pred, truth, frozen, spacing):
    result = {'physical_material_half_mse': float(.5*(gamma[..., 1:-1, 1:-1]-truth[..., 1:-1, 1:-1]).square().mean())}
    for name in ['released', 'self_consistent']:
        target = frozen[name]
        result[name] = {'cost': float(time_cost(pred, target, *spacing)),
                        'waveform_relative_l2': float((pred-target).double().norm()/target.double().norm())}
    return result


def check_sources():
    for name, info in json.loads((ROOT/'source_manifest.json').read_text())['files'].items():
        assert sha(ROOT/name) == info['sha256'], name
    previous = json.loads((PRIOR/'summary.json').read_text())
    for name, digest in previous['protocol']['input_hashes'].items():
        assert sha(ROOT/name) == digest, name


def run():
    check_sources()
    assert not OUT.exists(), 'Keep existing results; use a new output directory for a new protocol.'
    plan_path = PRIOR/'next_reduced_material_protocol.json'
    plan = json.loads(plan_path.read_text())
    assert plan['status'] == 'planned_not_run'
    values = plan['contrast_values']
    assert values == [1e-5, 1e-4, .001, .01, .05, .1, .3, .7, 1.]
    OUT.mkdir()
    protocol = {'status': 'fixed_before_forward_evaluation', 'prior_plan': plan,
                'prior_plan_sha256': sha(plan_path), 'script_sha256': sha(Path(__file__)),
                'frozen_observations_sha256': sha(PRIOR/'frozen_observations.pt'),
                'reference_sha256': sha(ROOT/'discrete_reference.py'),
                'input_hashes': json.loads((PRIOR/'summary.json').read_text())['protocol']['input_hashes'],
                'dtype': 'float32 same as frozen observations', 'optimization_steps': 0,
                'scope': 'Case1 known-geometry development scan; no operator validation/test or blind reconstruction.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    ref, force, frozen, spacing, mask = setup()
    truth = frozen['truth']
    support = truth < .5
    assert not (support & (mask == 0)).any(), 'Known defect intersects fixed cells.'
    start = time.perf_counter()
    records, fields, predictions = [], [], []
    with torch.no_grad():
        truth_prediction = ref.traces(truth, force, False)
        assert torch.equal(truth_prediction, frozen['self_consistent'])
        for a in values:
            gamma = torch.ones_like(truth)
            gamma[support] = a
            gamma[mask == 0] = truth[mask == 0]
            pred = ref.traces(gamma, force, False)
            assert torch.isfinite(pred).all()
            record = {'contrast': a, **metrics(gamma, pred, truth, frozen, spacing)}
            records.append(record)
            fields.append(gamma.cpu())
            predictions.append(pred.cpu())
            print(json.dumps(record), flush=True)
    torch.save({'fields': torch.cat(fields), 'predictions': torch.stack(predictions),
                'support': support.cpu(), 'truth_prediction': truth_prediction.cpu()}, OUT/'states.pt')
    report = {'status': 'completed', 'records': records, 'elapsed_seconds': time.perf_counter()-start,
              'support_physical_cells': int(support[..., 1:-1, 1:-1].sum()),
              'truth_defect_values': torch.unique(truth[support]).cpu().tolist(),
              'truth_reference': metrics(truth, truth_prediction, truth, frozen, spacing),
              'best_sampled': {name: min(records, key=lambda r: r[name]['cost'])['contrast'] for name in ['released', 'self_consistent']},
              'interpretation': 'Sampled minima only; no continuous optimum, geometry inference or full-field FWI claim.'}
    (OUT/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def verify():
    check_sources()
    protocol = json.loads((OUT/'protocol.json').read_text())
    for path, key in [(Path(__file__), 'script_sha256'), (PRIOR/'next_reduced_material_protocol.json', 'prior_plan_sha256'),
                      (PRIOR/'frozen_observations.pt', 'frozen_observations_sha256'), (ROOT/'discrete_reference.py', 'reference_sha256')]:
        assert sha(path) == protocol[key], str(path)
    for name, digest in protocol['input_hashes'].items():
        assert sha(ROOT/name) == digest
    report = json.loads((OUT/'summary.json').read_text())
    saved = torch.load(OUT/'states.pt', weights_only=True, map_location='cpu')
    ref, force, frozen, spacing, mask = setup()
    truth = frozen['truth']
    assert torch.equal(saved['support'], (truth < .5).cpu())
    assert [r['contrast'] for r in report['records']] == protocol['prior_plan']['contrast_values']
    assert saved['fields'].shape[0] == saved['predictions'].shape[0] == 9
    assert torch.isfinite(saved['fields']).all() and torch.isfinite(saved['predictions']).all()
    maxdiff = 0.
    with torch.no_grad():
        p = ref.traces(truth, force, False)
        assert torch.equal(p.cpu(), saved['truth_prediction']) and torch.equal(p, frozen['self_consistent'])
        assert metrics(truth, p, truth, frozen, spacing) == report['truth_reference']
        for i, record in enumerate(report['records']):
            gamma = saved['fields'][i:i+1].to(truth.device)
            support = truth < .5
            assert torch.equal(gamma[support], torch.full_like(gamma[support], record['contrast']))
            assert torch.equal(gamma[~support], truth[~support])
            assert torch.equal(gamma[mask == 0], truth[mask == 0])
            pred = ref.traces(gamma, force, False)
            assert torch.equal(pred.cpu(), saved['predictions'][i])
            vals = metrics(gamma, pred, truth, frozen, spacing)
            diffs = [abs(vals['physical_material_half_mse']-record['physical_material_half_mse'])]
            diffs += [abs(vals[n][k]-record[n][k]) for n in ['released', 'self_consistent'] for k in vals[n]]
            maxdiff = max(maxdiff, *diffs)
            assert max(diffs) == 0
            print('verified contrast', record['contrast'], flush=True)
    for name in ['released', 'self_consistent']:
        assert report['best_sampled'][name] == min(report['records'], key=lambda r: r[name]['cost'])['contrast']
    checks = {'status': 'passed', 'points_recomputed': 9, 'predictions': 'bit_identical',
              'max_metric_difference': maxdiff, 'truth_forward': 'bit_identical_to_frozen',
              'fixed_cells_and_known_support': 'passed', 'seven_original_files': 'unchanged',
              'protocol_and_input_hashes': 'passed', 'optimization_steps': 0,
              'scope': protocol['scope']}
    (OUT/'verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else run()

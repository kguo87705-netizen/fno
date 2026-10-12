"""Preregistered case1 known-support scalar optimization, four fixed35 arms."""
from pathlib import Path
import json
import time
import argparse
import torch
import pandas as pd
from run_reduced_material_scan import ROOT, PRIOR, setup, sha, check_sources
from discrete_reference import time_cost

OUT = Path('.build/author-scalar-material-control-20261008')
PLAN = Path('.build/author-reduced-material-20261008/next_scalar_optimization_protocol.json')


def field(a, truth):
    return torch.where(truth < .5, a, torch.ones_like(truth))


def values(a, gamma, pred, target, truth, spacing):
    return {'contrast': float(a.detach()), 'cost': float(time_cost(pred, target, *spacing).detach()),
            'waveform_relative_l2': float((pred.detach()-target).double().norm()/target.double().norm()),
            'material_half_mse': float(.5*(gamma.detach()-truth).square().mean()),
            'physical_material_half_mse': float(.5*(gamma.detach()[..., 1:-1, 1:-1]-truth[..., 1:-1, 1:-1]).square().mean())}


def audit(ref, force, frozen, spacing):
    ref.solver.double()
    force = force.double()
    truth = frozen['truth'].double()
    records = []
    for name in ['released', 'self_consistent']:
        for initial in ['uniform', 'near_truth']:
            a0 = 1. if initial == 'uniform' else float((.95*frozen['truth']+.05)[frozen['truth'] < .5][0])
            a = torch.tensor(a0, dtype=torch.float64, device=truth.device, requires_grad=True)
            target = frozen[name].double()
            loss = time_cost(ref.traces(field(a, truth), force), target, *spacing)*1e12
            grad = float(torch.autograd.grad(loss, a)[0])
            estimates = []
            with torch.no_grad():
                for relative in [1e-3, 5e-4]:
                    eps = a0*relative
                    plus = time_cost(ref.traces(field(a+eps, truth), force, False), target, *spacing)*1e12
                    minus = time_cost(ref.traces(field(a-eps, truth), force, False), target, *spacing)*1e12
                    fd = float((plus-minus)/(2*eps))
                    estimates.append({'epsilon': eps, 'finite_difference': fd,
                                      'relative_error': abs(fd-grad)/max(abs(fd), abs(grad), 1e-12)})
            assert estimates[-1]['relative_error'] < 1e-4, (name, initial, grad, estimates)
            record = {'observation': name, 'initial': initial, 'contrast': a0,
                      'scaled_ad_gradient': grad, 'checks': estimates,
                      'precision': 'promoted float32 stencil, float64 arithmetic; frozen float32 targets promoted',
                      'bounds': 'positive unprojected differences; a=1 positive side exceeds optimization bound'}
            records.append(record)
            print('gradient audit '+json.dumps(record), flush=True)
    return records


def run():
    check_sources()
    assert not OUT.exists(), 'Never overwrite prior results.'
    plan = json.loads(PLAN.read_text())
    assert plan['status'] == 'planned_not_run'
    OUT.mkdir()
    protocol = {'status': 'fixed_before_gradient_audit_and_training', 'plan': plan,
                'plan_sha256': sha(PLAN), 'script_sha256': sha(Path(__file__)),
                'reference_sha256': sha(ROOT/'discrete_reference.py'),
                'setup_script_sha256': sha(ROOT/'run_reduced_material_scan.py'),
                'frozen_observations_sha256': sha(PRIOR/'frozen_observations.pt'),
                'prior_summary_sha256': sha(PRIOR/'summary.json'),
                'audit': {'relative_epsilons': [1e-3, 5e-4], 'required_small_epsilon_relative_error': 1e-4},
                'input_hashes': json.loads((PRIOR/'summary.json').read_text())['protocol']['input_hashes']}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    ref, force, frozen, spacing, mask = setup()
    truth = frozen['truth']
    assert not ((truth < .5) & (mask == 0)).any()
    initial_checks = {}
    for name in ['released', 'self_consistent']:
        for initial in ['uniform', 'near_truth']:
            key = name+'_'+initial
            old_path = PRIOR/key/'states.pt'
            old = torch.load(old_path, weights_only=True, map_location=truth.device)['history'][0:1]
            a = truth.new_tensor(1.) if initial == 'uniform' else (.95*truth+.05)[truth < .5][0]
            assert torch.equal(field(a, truth), old), key
            initial_checks[key] = {'bit_identical': True, 'states_sha256': sha(old_path)}
    (OUT/'initial_checks.json').write_text(json.dumps(initial_checks, indent=2), encoding='utf-8')
    gradients = audit(ref, force, frozen, spacing)
    (OUT/'gradient_audit.json').write_text(json.dumps({'status': 'passed', 'records': gradients}, indent=2), encoding='utf-8')
    del ref, force, frozen, truth
    ref, force, frozen, spacing, mask = setup()
    truth = frozen['truth']
    lower = float(pd.read_csv(ROOT/'settings2D.csv').iloc[0]['gamma0'])
    results = {}
    for name in ['released', 'self_consistent']:
        for initial in ['uniform', 'near_truth']:
            key = name+'_'+initial
            a0 = truth.new_tensor(1.) if initial == 'uniform' else (.95*truth+.05)[truth < .5][0]
            a = torch.nn.Parameter(a0.clone())
            opt = torch.optim.Adam([a], lr=.02)
            records, contrasts = [], []
            start = time.perf_counter()
            for update in range(36):
                opt.zero_grad(set_to_none=True)
                gamma = field(a, truth)
                if update == 35:
                    with torch.no_grad():
                        pred = ref.traces(gamma, force, False)
                        loss = time_cost(pred, frozen[name], *spacing)
                else:
                    pred = ref.traces(gamma, force)
                    loss = time_cost(pred, frozen[name], *spacing)
                assert torch.isfinite(loss)
                record = {'update': update, **values(a, gamma, pred, frozen[name], truth, spacing)}
                contrasts.append(a.detach().cpu().clone())
                if update < 35:
                    (loss*1e12).backward()
                    assert torch.isfinite(a.grad)
                    record['scaled_gradient_before_clip'] = float(a.grad)
                    torch.nn.utils.clip_grad_norm_([a], 1e-5)
                    record['gradient_after_clip'] = float(a.grad)
                    opt.step()
                    with torch.no_grad():
                        a.clamp_(lower, 1.)
                records.append(record)
                if update % 5 == 0:
                    print(key+' '+json.dumps(record), flush=True)
            arm = OUT/key
            arm.mkdir()
            torch.save({'contrasts': torch.stack(contrasts), 'prediction_final': pred.detach().cpu(),
                        'final_optimizer': opt.state_dict()}, arm/'states.pt')
            result = {'records': records, 'elapsed_seconds': time.perf_counter()-start,
                      'final_cost_over_initial': records[-1]['cost']/records[0]['cost'],
                      'final_material_mse_over_initial': records[-1]['material_half_mse']/records[0]['material_half_mse'],
                      'cost_increases': sum(b['cost'] > a['cost'] for a,b in zip(records, records[1:]))}
            (arm/'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            results[key] = result
    (OUT/'summary.json').write_text(json.dumps({'status': 'completed', 'results': results,
        'scope': plan['interpretation']}, indent=2), encoding='utf-8')


def verify():
    check_sources()
    p = json.loads((OUT/'protocol.json').read_text())
    for path,key in [(PLAN,'plan_sha256'),(Path(__file__),'script_sha256'),(ROOT/'discrete_reference.py','reference_sha256'),
                     (ROOT/'run_reduced_material_scan.py','setup_script_sha256'),(PRIOR/'frozen_observations.pt','frozen_observations_sha256'),
                     (PRIOR/'summary.json','prior_summary_sha256')]:
        assert sha(path) == p[key]
    for name,digest in p['input_hashes'].items():
        assert sha(ROOT/name) == digest
    initial_checks = json.loads((OUT/'initial_checks.json').read_text())
    assert json.loads((OUT/'gradient_audit.json').read_text())['status'] == 'passed'
    r = json.loads((OUT/'summary.json').read_text())
    assert r['status'] == 'completed'
    ref,force,frozen,spacing,mask = setup()
    truth = frozen['truth']
    checks = {}
    for key,arm in r['results'].items():
        data = torch.load(OUT/key/'states.pt', weights_only=True, map_location=truth.device)
        assert data['contrasts'].shape == (36,) and torch.isfinite(data['contrasts']).all()
        for state in data['final_optimizer']['state'].values():
            assert int(state['step']) == 35
            assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
        assert sha(PRIOR/key/'states.pt') == initial_checks[key]['states_sha256']
        old = torch.load(PRIOR/key/'states.pt', weights_only=True, map_location=truth.device)['history'][0:1]
        assert torch.equal(field(data['contrasts'][0],truth),old)
        name = 'self_consistent' if key.startswith('self_consistent') else 'released'
        for i in range(36):
            assert float(data['contrasts'][i]) == arm['records'][i]['contrast']
        for i in [0,35]:
            a = data['contrasts'][i]
            gamma = field(a,truth)
            assert torch.equal(gamma[mask == 0], truth[mask == 0])
            with torch.no_grad():
                pred = ref.traces(gamma,force,False)
            if i == 35:
                assert torch.equal(pred,data['prediction_final'])
            for k,v in values(a,gamma,pred,frozen[name],truth,spacing).items():
                assert v == arm['records'][i][k], (key,i,k)
        checks[key] = {'initial_fields': 'bit_identical_to_free_pixels', 'endpoints': 'metrics_exact',
                       'final_prediction': 'bit_identical', 'finite_adam_steps':35}
    report = {'status':'passed','arms':checks,'sources_and_hashes':'passed','gradient_audit':'passed',
              'scope':r['scope']}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    verify() if args.verify else run()

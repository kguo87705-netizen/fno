"""Frozen four-state, observation-conditioned acquisition mechanism experiment."""
from pathlib import Path
import itertools
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import FiniteDifferencePyTorchConv as FD
import Utilities
from run_tx_design_value_20261012 import sha, dump, mask_for

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
OUT = ROOT / '.build/active-second-shot-20261012'
PLAN = ROOT / '.build/route-evidence-audit-20261012/selected_minimal_experiment.json'
BG = ROOT / '.build/tx-design-value-20261012/background.pt'
PAIRS = list(itertools.combinations(range(4), 2))
SECONDS = [0, 2, 3]


def logpost(ll):
    shifted = ll - np.max(ll, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def choose_action(initial_logposterior, hypothetical_ll):
    """Only first-shot posterior and frozen dictionary enter this policy."""
    p = np.exp(initial_logposterior)
    risks = np.empty((len(p), 3))
    for j, ll in enumerate(hypothetical_ll):
        for start in range(0, len(p), 128):
            end = min(start+128, len(p))
            post = np.exp(logpost(initial_logposterior[start:end, None, None, :] + ll[None]))
            high = post[..., 1] + post[..., 3]
            conditional_risk = np.minimum(high, 1-high).mean(axis=-1)
            risks[start:end, j] = np.sum(p[start:end] * conditional_risk, axis=-1)
    return np.asarray(SECONDS)[risks.argmin(axis=1)], risks


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError('Preserve previous output; no automatic rerun.')
    OUT.mkdir(parents=True)
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert sha(ROOT / '.build/route-evidence-audit-20261012/summary.json') == plan['upstream_audit_sha256']
    oldmanifest = json.loads((BG.parent/'data_manifest.json').read_text(encoding='utf-8'))
    assert sha(BG) == oldmanifest['background_sha256']
    inputs = [Path(__file__), PLAN, BG, CODE/'run_tx_design_value_20261012.py', CODE/'settings2D.csv',
              CODE/'data_casestudy/source.h5', CODE/'FiniteDifferencePyTorchConv.py', CODE/'Utilities.py']
    protocol = dict(plan=plan, status='frozen_before_simulation',
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
        implementation=dict(state_order='x outer, gamma inner; low/high labels 0/1',
            inference='exact Gaussian likelihood in full row-span SVD sufficient statistics; rank threshold smax*1e-12',
            evaluation_rng='numpy default_rng SeedSequence([20261012,18,state,seed]); source order 0..3',
            integration_rng='numpy default_rng(2026101218); second-source order 0,2,3; state then draw then coordinate',
            action_ties='first minimum, source order 0,2,3', label_ties='lower material/left location',
            bootstrap='10000 stratified paired replicates, common indices across methods, quantile .05/6, numpy linear',
            posterior_validation='16 full waveform noise draws, seed 2026101220; compare normalized full and projected likelihood',
            oracle='per-state best among three common-first-shot fixed pairs; also per-trial optimistic oracle; truth-informed only'))
    dump(OUT/'protocol.json', protocol)
    ledger = dict(status='running', entries=[], physical_calls_attempted=0, network_forwards=0, training_updates=0)
    dump(OUT/'ledger.json', ledger)

    def checktime():
        if time.perf_counter()-started > 600:
            raise TimeoutError('Frozen total wall time budget exceeded; no retries.')

    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg = pd.read_csv(CODE/'settings2D.csv').iloc[0]
        nx, ny, n = [int(cfg[k]) for k in ['Nx', 'Ny', 'N']]
        sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg['distanceBetweenSources']), 4)
        rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg['distanceBetweenSensors']), sx)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        force = torch.tensor(pd.read_hdf(CODE/'data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).to(device)
        dump(OUT/'environment.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
             device=str(device), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
             source_dtype=str(force.dtype), sources=list(zip(sx,sy)), receivers=list(zip(rx,ry))))
        states = [dict(x=x, y=.028, a=.006, b=.003, gamma=g) for x in [.038,.062] for g in [.4999,.5001]]

        def simulate(state, source, role):
            checktime()
            assert ledger['physical_calls_attempted'] < 18
            entry = dict(state=state, source=source, role=role, status='started')
            ledger['entries'].append(entry)
            ledger['physical_calls_attempted'] += 1
            dump(OUT/'ledger.json', ledger)
            t0 = time.perf_counter()
            try:
                geom = states[state]
                mask = mask_for(geom,nx,ny,float(cfg['Lx']),float(cfg['Ly']))
                assert mask.any() and not mask[[0,-1],:].any() and not mask[:,[0,-1]].any()
                material = torch.where(mask,torch.tensor(geom['gamma']),torch.tensor(1.))
                gamma = torch.ones(1,1,nx+3,ny+3,device=device)
                gamma[0,0,1:-1,1:-1] = material.to(device)
                initial = torch.zeros_like(gamma)
                solver = FD.FiniteDifference(float(cfg['dt']),float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['c']),float(cfg['rho']),device)
                with torch.no_grad():
                    wave = solver.forwardNSteps(initial,initial.clone(),gamma,force[source:source+1],nx,ny,n,1,device)
                    if device.type=='cuda': torch.cuda.synchronize()
                    trace = wave[0,rx,ry,1:].cpu().contiguous().clone()
                assert trace.shape==(24,2000) and torch.isfinite(trace).all()
                entry.update(status='completed',seconds=time.perf_counter()-t0)
                return trace
            except Exception as e:
                entry.update(status='failed',error=repr(e),seconds=time.perf_counter()-t0)
                raise
            finally:
                dump(OUT/'ledger.json',ledger)

        data, manifest = [], []
        for h, state in enumerate(states):
            traces = torch.stack([simulate(h,s,'dictionary') for s in range(4)])
            file = OUT/f'state_{h}.pt'
            torch.save(dict(state=state,traces=traces),file)
            manifest.append(dict(file=file.name,sha256=sha(file)))
            data.append(traces.numpy().astype(np.float64))
            print(f'dictionary state {h} complete; calls={ledger["physical_calls_attempted"]}',flush=True)
        replays = []
        for h,s in [(0,1),(3,3)]:
            trace = simulate(h,s,'replay')
            equal = np.array_equal(trace.numpy(),data[h][s])
            replays.append(dict(state=h,source=s,bit_equal=equal))
            assert equal
        dump(OUT/'data_manifest.json',dict(records=manifest,replays=replays))
        ledger['physics_seconds_including_preparation'] = time.perf_counter()-started
        dump(OUT/'ledger.json',ledger)
        print('16 dictionary calls and 2 replays passed; analyzing',flush=True)
        bg = torch.load(BG,weights_only=True).numpy().astype(np.float64)
        rms = float(np.sqrt(np.mean(bg**2)))
        assert abs(rms-.2552473836595172)<1e-12
        sigma = .01*rms
        waves = np.stack(data).reshape(4,4,-1)
        coords, bases, diagnostics = [], [], []
        validation_rng = np.random.default_rng(2026101220)
        posterior_errors = []
        for s in range(4):
            whitened = (waves[:,s]-waves[0,s])/sigma
            u, sv, vt = np.linalg.svd(whitened,full_matrices=False)
            rank = int(np.count_nonzero(sv > sv[0]*1e-12))
            q = u[:,:rank]*sv[:rank]
            basis = vt[:rank]
            reconstruction = float(np.max(np.abs(q@basis-whitened)))
            assert reconstruction<1e-7
            coords.append(q); bases.append(basis)
            diagnostics.append(dict(source=s,rank=rank,singular_values=sv.tolist(),reconstruction_max_abs=reconstruction,
                material_mahalanobis_by_location=[float(np.linalg.norm(q[1]-q[0])),float(np.linalg.norm(q[3]-q[2]))]))
            for h in range(4):
                noise = validation_rng.normal(size=whitened.shape[1])
                obs = whitened[h]+noise
                full = -.5*np.sum((obs[None]-whitened)**2,axis=1)
                reduced = -.5*np.sum((obs@basis.T-q)**2,axis=1)
                posterior_errors.append(float(np.max(np.abs(np.exp(logpost(full))-np.exp(logpost(reduced))))))
        assert max(posterior_errors)<1e-8
        # Potential outcomes paired across methods. Truth is used only to generate evaluation observations.
        ll = np.empty((4,1000,4,4))
        observations = np.zeros((4,1000,4,3))
        for h in range(4):
            for k,seed in enumerate(plan['evaluation_seeds']):
                rng = np.random.default_rng(np.random.SeedSequence([20261012,18,h,seed]))
                for s,q in enumerate(coords):
                    y = q[h]+rng.normal(size=q.shape[1])
                    observations[h,k,s,:len(y)] = y
                    ll[h,k,s] = -.5*np.sum((y-q)**2,axis=1)
        integration_rng = np.random.default_rng(2026101218)
        hypothetical = []
        for s in SECONDS:
            q = coords[s]
            y = q[:,None,:]+integration_rng.normal(size=(4,512,q.shape[1]))
            hypothetical.append(-.5*np.sum((y[:,:,None,:]-q[None,None,:,:])**2,axis=-1))
        firstlog = logpost(ll[:,:,1,:].reshape(-1,4))
        action, risks = choose_action(firstlog,hypothetical)
        assert np.array_equal(choose_action(firstlog[::-1],hypothetical)[0][::-1],action)
        flatll = ll.reshape(-1,4,4)
        adaptivepost = np.exp(logpost(flatll[:,1]+flatll[np.arange(4000),action])).reshape(4,1000,4)
        posts = [np.exp(logpost(ll[:,:,a]+ll[:,:,b])) for a,b in PAIRS]+[adaptivepost,np.exp(firstlog).reshape(4,1000,4)]
        posts = np.stack(posts,axis=2)
        high = posts[...,1]+posts[...,3]
        right = posts[...,2]+posts[...,3]
        truth_material = (np.arange(4)%2)[:,None,None]
        truth_location = (np.arange(4)//2)[:,None,None]
        errors = ((high>.5)!=truth_material)
        location_errors = ((right>.5)!=truth_location)
        rates = errors.mean(axis=(0,1))
        differences = errors[:,:,:6].astype(float)-errors[:,:,6,None]
        boot_rng = np.random.default_rng(2026101219)
        boot = np.zeros((10000,6))
        for start in range(0,10000,200):
            checktime()
            for h in range(4):
                indices = boot_rng.integers(0,1000,size=(200,1000))
                boot[start:start+200] += differences[h][indices].mean(axis=1)/4
        lower = np.quantile(boot,.05/6,axis=0)
        reduction = rates[:6]-rates[6]
        relative = np.divide(reduction,rates[:6],out=np.zeros(6),where=rates[:6]>0)
        gate_each = (reduction>=.01)&(relative>=.2)&(lower>0)
        common = [PAIRS.index(tuple(sorted([1,s]))) for s in SECONDS]
        common_errors = errors[:,:,common]
        state_best = common_errors.mean(axis=1).argmin(axis=1)
        state_oracle = float(np.mean([common_errors[h,:,state_best[h]].mean() for h in range(4)]))
        trial_oracle = float(common_errors.all(axis=-1).mean())
        names = [f'fixed_{a}_{b}' for a,b in PAIRS]+['adaptive','source1_only']
        metrics = {name:dict(material_error_rate=float(rates[j]),position_error_rate=float(location_errors[:,:,j].mean()),
                    material_brier=float(np.mean((high[:,:,j]-truth_material[:,:,0])**2)),
                    error_by_state=errors[:,:,j].mean(axis=1).tolist()) for j,name in enumerate(names)}
        comparisons = [dict(fixed=names[j],absolute_reduction=float(reduction[j]),relative_reduction=float(relative[j]),
                           simultaneous_lower_bound=float(lower[j]),gate_pass=bool(gate_each[j])) for j in range(6)]
        calibration = {}
        for j,name in enumerate(names):
            bins = []
            for low,up in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                sel = (high[:,:,j]>=low)&((high[:,:,j]<up) if up<1 else (high[:,:,j]<=up))
                if sel.any():
                    truth_broadcast = np.broadcast_to(truth_material[:,:,0],(4,1000))
                    bins.append(dict(low=float(low),upper=float(up),count=int(sel.sum()),
                         mean_predicted=float(high[:,:,j][sel].mean()),observed_high=float(truth_broadcast[sel].mean())))
            calibration[name] = bins
        chosenrisk = risks[np.arange(4000),[SECONDS.index(s) for s in action]]
        summary = dict(status='completed',scientific_gate_pass=bool(gate_each.all()),sigma=sigma,background_rms=rms,
             metrics=metrics,comparisons=comparisons,
             action_frequencies={str(s):int(np.sum(action==s)) for s in SECONDS},
             action_frequencies_by_state=[{str(s):int(np.sum(action.reshape(4,1000)[h]==s)) for s in SECONDS} for h in range(4)],
             mean_predicted_chosen_risk=float(chosenrisk.mean()),realized_adaptive_error=float(rates[6]),
             oracle=dict(state_informed_error_rate=state_oracle,state_best_sources=[SECONDS[j] for j in state_best],
                         trial_informed_error_rate=trial_oracle,deployable=False),
             diagnostics=diagnostics,calibration=calibration,
             scope='four matched-model hypotheses, 4000 conditional noise trials, four states only; no continuous inversion or hardware validation',
             old_continuous_gate='unchanged and failed; this experiment cannot override it')
        dump(OUT/'summary.json',summary)
        np.savez_compressed(OUT/'trial_results.npz',observations=observations,source_loglikelihood=ll,posteriors=posts,
             actions=action.reshape(4,1000),predicted_risks=risks.reshape(4,1000,3),material_errors=errors,
             location_errors=location_errors,bootstrap_reductions=boot,**{f'coords_{s}':q for s,q in enumerate(coords)})
        # Verify from saved files, with likelihood and all decisions independently recomputed.
        saved = np.load(OUT/'trial_results.npz')
        checks = dict(input_hashes=all(sha(ROOT/p)==v for p,v in protocol['input_sha256'].items()),
            waveform_hashes=all(sha(OUT/r['file'])==r['sha256'] for r in manifest),
            replay_bit_equal=all(r['bit_equal'] for r in replays),
            full_likelihood_posterior_max_abs=max(posterior_errors),
            reduced_likelihood_max_abs=max(float(np.max(np.abs(-.5*np.sum((saved['observations'][:,:,s,:q.shape[1],None]-q.T)**2,axis=2)-ll[:,:,s]))) for s,q in enumerate(coords)),
            posterior_normalization_max_abs=float(np.max(np.abs(saved['posteriors'].sum(axis=-1)-1))),
            fixed_decisions_match=all(np.array_equal(np.exp(logpost(ll[:,:,a]+ll[:,:,b]))[:,:,1::2].sum(axis=-1)>.5,high[:,:,j]>.5) for j,(a,b) in enumerate(PAIRS)),
            truth_free_policy_permutation_invariant=True,physical_attempts=ledger['physical_calls_attempted'],
            failed_physical_calls=sum(e['status']=='failed' for e in ledger['entries']))
        assert checks['input_hashes'] and checks['waveform_hashes'] and checks['replay_bit_equal'] and checks['fixed_decisions_match']
        assert checks['reduced_likelihood_max_abs']<1e-8 and checks['posterior_normalization_max_abs']<1e-12
        checktime()
        ledger.update(status='completed',total_wall_seconds=time.perf_counter()-started,
             solver_and_extract_seconds=sum(e['seconds'] for e in ledger['entries']))
        dump(OUT/'ledger.json',ledger)
        checks.update(status='passed',scope='hashes, fresh replays, Gaussian sufficient statistics, posterior decisions, saved arrays; scientific gate separate')
        dump(OUT/'verification.json',checks)
        print(json.dumps(dict(gate=summary['scientific_gate_pass'],metrics=metrics,comparisons=comparisons,
             actions=summary['action_frequencies'],oracle=summary['oracle'],wall=ledger['total_wall_seconds']),indent=2),flush=True)
    except Exception as e:
        ledger.update(status='failed',error=repr(e),wall_seconds_before_failure=time.perf_counter()-started)
        dump(OUT/'ledger.json',ledger)
        dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_automatic_retries=True))
        raise


if __name__=='__main__':
    main()

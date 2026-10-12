"""Frozen synthetic source-gain robustness check, reusing physical waveforms."""
from pathlib import Path
import itertools
import json
import time
import traceback
import numpy as np
import torch
from run_tx_design_value_20261012 import sha, dump

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT/'.build/active-second-shot-20261012'
OUT = ROOT/'.build/gain-nuisance-20261012'
PAIRS = list(itertools.combinations(range(4),2))+[(0,1,2,3)]
METHODS = ['assume_unit_gain','profile_per_source_gain','oracle_true_gain']
PATTERNS = {'nominal':[1.]*4,'common_low':[.9975]*4,'common_high':[1.0025]*4,
            'alternating':[.9975,1.0025,.9975,1.0025],
            'alternating_reverse':[1.0025,.9975,1.0025,.9975]}


def classify(scores):
    # Profile likelihood is a score, not a calibrated posterior probability.
    return np.argmin(scores,axis=-1)


def main():
    start = time.perf_counter()
    if OUT.exists(): raise FileExistsError('Preserve old results; no automatic rerun')
    OUT.mkdir(parents=True)
    summary = json.loads((UP/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((UP/'data_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(UP/r['file'])==r['sha256'] for r in manifest['records'])
    files = [Path(__file__),Path(__file__).with_name('run_tx_design_value_20261012.py'),
             UP/'summary.json',UP/'data_manifest.json']+[UP/r['file'] for r in manifest['records']]
    protocol = dict(status='frozen_before_evaluation',question='Does unknown source amplitude bias a fixed physical material classifier, and does a simple bounded gain fit remove it?',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        prior_states='same four exact physical hypotheses, uniform',fixed_primary_sources=[1,2],
        comparisons=[list(p) for p in PAIRS],full_four_source_role='extra acquisition budget reference, not same-cost competitor',
        truth_gain_patterns=PATTERNS,inference_gain_bounds=[.995,1.005],
        gain_model='separate scalar multiplicative gain for each complete raw source trace; additive Gaussian noise follows gain',
        units='dimensionless amplitude; arbitrary synthetic stress range, not measured instrument specification',
        noise_sigma=summary['sigma'],noise_model='unchanged iid per waveform sample; exact sufficient-statistic simulation',
        noise_seeds=list(range(1000,2000)),rng='default_rng SeedSequence([20261012,21,state,seed]); source order 0..3; same potential noise across methods and gain patterns',
        inference='joint state minimum squared residual; unit gain, per-source bounded least squares gain, and supplied true gain oracle; no state truth in unit/profile functions',
        metrics=['material_error_rate','position_error_rate','worst_gain_material_error','gain_estimation_error'],
        bootstrap=dict(replicates=10000,seed=2026101222,method='paired stratified by four states; same indices across all five patterns and contrasts',
            family='10 primary fixed12 contrasts: unit minus profile and profile minus oracle at five gains',
            simultaneous_quantiles=[.0025,.9975]),
        decision=dict(unit_gain_failure='any stress pattern worsens unit-gain material error over its nominal rate by at least .01',
            simple_fit_adequate='profile error no more than .005 above oracle for every pattern (descriptive screening tolerance, not hardware standard)',
            no_route_reversal='synthetic calibration stress cannot establish active selection gain, neural necessity, continuous geometry resolution or hardware performance'),
        physical_calls=0,network_forwards=0,training_updates=0,wall_seconds_cap=180,
        cost_boundary='reuse previously counted 18 source calls; waveform loading, analysis and bootstrap timed separately; no acquisition savings claim')
    dump(OUT/'protocol.json',protocol)
    ledger=dict(status='running',new_physical_calls=0,network_forwards=0,training_updates=0,upstream_physical_calls=18)
    dump(OUT/'ledger.json',ledger)
    try:
        torch.set_num_threads(4)
        w=np.stack([torch.load(UP/f'state_{h}.pt',weights_only=True)['traces'].numpy().astype(float) for h in range(4)]).reshape(4,4,-1)/summary['sigma']
        coordinates=[]; bases=[]; noise=[]; diag=[]
        for s in range(4):
            u,sv,vt=np.linalg.svd(w[:,s],full_matrices=False)
            assert sv[-1]>sv[0]*1e-12
            q=u*sv; coordinates.append(q); bases.append(vt)
            assert np.max(np.abs(q@vt-w[:,s]))<1e-7
            rows=[]
            for l in range(2):
                delta=q[2*l+1]-q[2*l]; mean=(q[2*l+1]+q[2*l])/2
                residual=delta-mean*(delta@mean)/(mean@mean)
                rows.append(dict(material_distance=float(np.linalg.norm(delta)),
                    gain_direction_cosine=float(delta@mean/(np.linalg.norm(delta)*np.linalg.norm(mean))),
                    material_distance_after_gain_projection=float(np.linalg.norm(residual))))
            diag.append(dict(source=s,material_by_location=rows,
                minimum_cross_location_distance=float(min(np.linalg.norm(q[a]-q[b]) for a in [0,1] for b in [2,3]))))
        q=np.stack(coordinates) # source, hypothesis, coordinate
        z=np.empty((4,1000,4,4))
        for h in range(4):
            for k,seed in enumerate(protocol['noise_seeds']):
                z[h,k]=np.random.default_rng(np.random.SeedSequence([20261012,21,h,seed])).normal(size=(4,4))
        gains=np.array(list(PATTERNS.values()))
        obs=q.transpose(1,0,2)[:,None,None,:,:]*gains[None,None,:,:,None]+z[:,:,None,:,:]
        norm=(q*q).sum(axis=-1)
        fit=np.clip(np.einsum('hnpsc,stc->hnpst',obs,q)/norm[None,None,None,:,:],.995,1.005)
        unit=((obs[:,:,:,:,None,:]-q[None,None,None,:,:,:])**2).sum(axis=-1)
        prof=((obs[:,:,:,:,None,:]-fit[:,:,:,:,:,None]*q[None,None,None,:,:,:])**2).sum(axis=-1)
        oracle=((obs[:,:,:,:,None,:]-gains[None,None,:,:,None,None]*q[None,None,None,:,:,:])**2).sum(axis=-1)
        scores=np.stack([unit,prof,oracle],axis=3) # h, seed, gain, method, source, hypothesis
        decisions=np.stack([classify(scores[:,:,:,:,list(pair),:].sum(axis=4)) for pair in PAIRS],axis=-1)
        material_errors=decisions%2!=(np.arange(4)%2)[:,None,None,None,None]
        position_errors=decisions//2!=(np.arange(4)//2)[:,None,None,None,None]
        rates=material_errors.mean(axis=(0,1)); position_rates=position_errors.mean(axis=(0,1))
        # Primary fixed pair only, to avoid selecting a winner on stress evaluation.
        e=material_errors[:,:,:,:,3].astype(float)
        contrast=np.stack([e[:,:,:,0]-e[:,:,:,1],e[:,:,:,1]-e[:,:,:,2]],axis=-1).reshape(4,1000,10)
        boot=np.zeros((10000,10)); rng=np.random.default_rng(2026101222)
        for b in range(0,10000,200):
            if time.perf_counter()-start>180: raise TimeoutError('Frozen budget exceeded')
            for h in range(4):
                idx=rng.integers(0,1000,(200,1000))
                boot[b:b+200]+=contrast[h][idx].mean(axis=1)/4
        lower,upper=np.quantile(boot,[.0025,.9975],axis=0)
        gain_selected=fit[np.arange(4)[:,None,None,None],np.arange(1000)[None,:,None,None],
            np.arange(5)[None,None,:,None],np.arange(4)[None,None,None,:],decisions[:,:,:,1,3,None]]
        # Full-space independent checks: 20 fresh noise observations, all candidate gains.
        vrng=np.random.default_rng(2026101223); maxgain=0.; maxscore=0.; correct=True
        for h in range(4):
            for p in range(5):
                total_full=np.zeros((3,4)); total_projected=np.zeros((3,4))
                for s in [1,2]:
                    fullobs=gains[p,s]*w[h,s]+vrng.normal(size=w.shape[-1])
                    y=fullobs@bases[s].T
                    directfit=np.clip(w[:,s]@fullobs/(w[:,s]**2).sum(axis=-1),.995,1.005)
                    smallfit=np.clip(q[s]@y/norm[s],.995,1.005)
                    maxgain=max(maxgain,float(np.max(np.abs(directfit-smallfit))))
                    for m,amps in enumerate([np.ones(4),directfit,np.full(4,gains[p,s])]):
                        ds=((fullobs[None,:]-amps[:,None]*w[:,s])**2).sum(axis=-1)
                        rs=((y[None,:]-amps[:,None]*q[s])**2).sum(axis=-1)
                        maxscore=max(maxscore,float(np.max(np.abs((ds-ds[0])-(rs-rs[0])))))
                        total_full[m]+=ds;total_projected[m]+=rs
                correct=correct and np.array_equal(total_full.argmin(axis=-1),total_projected.argmin(axis=-1))
        assert maxgain<1e-10 and maxscore<1e-5 and correct
        case_metrics={name:{METHODS[m]:{str(list(pair)):dict(material_error=float(rates[p,m,j]),position_error=float(position_rates[p,m,j]))
            for j,pair in enumerate(PAIRS)} for m in range(3)} for p,name in enumerate(PATTERNS)}
        primary={METHODS[m]:dict(error_by_gain=rates[:,m,3].tolist(),worst_error=float(rates[:,m,3].max())) for m in range(3)}
        degrade=rates[1:,0,3]-rates[0,0,3]
        excess=rates[:,1,3]-rates[:,2,3]
        result=dict(status='completed',primary_fixed12=primary,metrics=case_metrics,
            contrasts=[dict(pattern=list(PATTERNS)[j//2],comparison=['unit_minus_profile','profile_minus_oracle'][j%2],
                difference=float(contrast[:,:,j].mean()),simultaneous_lower=float(lower[j]),simultaneous_upper=float(upper[j])) for j in range(10)],
            diagnosis=diag,unit_gain_failure=bool(np.any(degrade>=.01)),simple_fit_adequate=bool(np.all(excess<=.005)),
            fitted_gain_mae_primary=float(np.mean(np.abs(gain_selected-gains[None,None,:,:]))),
            interpretation='synthetic source amplitude stress only; no physical, active-selection or neural performance claim',
            limits='same four discrete hypotheses and exact matched waveforms; 4000 paired noise draws reused across five gain patterns; profile score is not calibrated posterior')
        dump(OUT/'summary.json',result)
        np.savez_compressed(OUT/'trial_results.npz',observations=obs,coordinates=q,gains=gains,source_scores=scores,
            fitted_gain=fit,decisions=decisions,material_errors=material_errors,position_errors=position_errors,bootstrap=boot)
        saved=np.load(OUT/'trial_results.npz')
        for j,pair in enumerate(PAIRS):
            assert np.array_equal(saved['source_scores'][:,:,:,:,list(pair),:].sum(axis=4).argmin(axis=-1),saved['decisions'][...,j])
        assert np.array_equal(saved['material_errors'].mean(axis=(0,1)),rates)
        checks=dict(status='passed',input_hashes=all(sha(ROOT/p)==v for p,v in protocol['input_sha256'].items()),
            full_space_gain_fit_max_abs=maxgain,full_space_relative_score_max_abs=maxscore,
            full_space_state_decisions_match=bool(correct),saved_pair_decisions_and_rates_match=True,
            array_shapes={k:list(saved[k].shape) for k in saved.files},new_physical_calls=0)
        assert checks['input_hashes']
        dump(OUT/'verification.json',checks)
        ledger.update(status='completed',wall_seconds=time.perf_counter()-start)
        dump(OUT/'ledger.json',ledger)
        print(json.dumps(dict(primary=primary,unit_gain_failure=result['unit_gain_failure'],simple_fit_adequate=result['simple_fit_adequate'],verification=checks,wall=ledger['wall_seconds']),indent=2))
    except Exception as err:
        ledger.update(status='failed',error=repr(err),wall_seconds=time.perf_counter()-start)
        dump(OUT/'ledger.json',ledger)
        dump(OUT/'failure.json',dict(error=repr(err),traceback=traceback.format_exc(),no_auto_retry=True))
        raise


if __name__=='__main__': main()

"""Independent saved-output check; no physics calls, fitting, or policy changes."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'.build/active-second-shot-20261012'
started = time.perf_counter()
summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
z = np.load(OUT/'trial_results.npz')
waves = np.stack([torch.load(OUT/f'state_{h}.pt',weights_only=True)['traces'].numpy().astype(float) for h in range(4)])
errors = z['material_errors']
rows = []
for j,(a,b) in enumerate(itertools.combinations(range(4),2)):
    post = z['posteriors'][:,:,j]
    decision = (post[:,:,1]+post[:,:,3])>.5
    direct_error = decision != (np.arange(4)%2)[:,None]
    assert np.array_equal(direct_error,errors[:,:,j])
    assert float(direct_error.mean())==summary['metrics'][f'fixed_{a}_{b}']['material_error_rate']
    delta = waves[1::2]-waves[::2]
    distances = np.sqrt(np.sum(delta[:,[a,b]]**2,axis=(1,2,3)))/summary['sigma']
    analytic_known_location_risk = float(np.mean([.5*math.erfc(d/(2*math.sqrt(2))) for d in distances]))
    rows.append(dict(pair=[a,b],known_location_equal_prior_gaussian_risk=analytic_known_location_risk))
adaptive = errors[:,:,6]
baseline = errors[:,:,3]
helped = int(np.sum(baseline & ~adaptive))
harmed = int(np.sum(~baseline & adaptive))
n = helped+harmed
one_sided_p = sum(math.comb(n,k) for k in range(helped,n+1))/2**n
lower = np.quantile(z['bootstrap_reductions'],.05/6,axis=0)
assert np.allclose(lower,[r['simultaneous_lower_bound'] for r in summary['comparisons']],rtol=0,atol=1e-14)
dist = np.array([[np.linalg.norm(waves[2*l+1,s]-waves[2*l,s])/summary['sigma'] for s in range(4)] for l in range(2)])
best_second = np.array([0,2,3])[dist[:,[0,2,3]].argmax(axis=1)]
knownloc_optimal_risk = float(np.mean([.5*math.erfc(math.hypot(dist[l,1],dist[l,s])/(2*math.sqrt(2))) for l,s in enumerate(best_second)]))
result = dict(status='passed',kind='posthoc_saved_data_verification_and_theoretical_diagnostic_not_a_new_gate',
    no_new_physical_calls=0,no_network_forwards=0,no_training_updates=0,
    fixed_metrics_and_saved_bootstrap_reproduced=True,
    paired_main_baseline=dict(fixed_errors=int(baseline.sum()),adaptive_errors=int(adaptive.sum()),
        fixed_wrong_adaptive_right=helped,fixed_right_adaptive_wrong=harmed,
        exact_one_sided_mcnemar_p=one_sided_p,bonferroni_threshold=.05/6),
    known_location_theory=dict(assumptions='posthoc analytic Gaussian equal-prior binary material classification, location supplied; cannot replace frozen unknown-location policy or gate',
        fixed_pairs=rows,best_second_by_location=best_second.tolist(),
        ideal_common_first_expected_error=knownloc_optimal_risk,
        expected_gain_vs_fixed12=rows[3]['known_location_equal_prior_gaussian_risk']-knownloc_optimal_risk),
    gate_saturation=dict(observed_fixed12_error=float(baseline.mean()),required_absolute_reduction=.01,
        absolute_gate_impossible_even_if_zero_adaptive_error=bool(baseline.mean()<.01),
        interpretation='observed baseline leaves less than one percentage point of error; frozen gate unchanged; no general impossibility claim'),
    seconds=time.perf_counter()-started)
(OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

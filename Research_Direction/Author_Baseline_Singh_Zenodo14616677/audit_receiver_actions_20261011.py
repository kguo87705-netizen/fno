import json, hashlib, time
from pathlib import Path
import torch
R=Path.cwd(); A=R/'.build/phase-a-array-interface-20261011-r2'; O=R/'.build/receiver-action-evidence-audit-20261011'
assert not O.exists(); O.mkdir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
files=list(A.glob('*.pt'))+[A/'protocol.json',A/'ledger.json',A/'verification.json',R/'Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_phase_b_frozen_r2_subarray_diagnostic_20261011.py']
hashes={str(p.relative_to(R)):sha(p) for p in files};dump('protocol.json',{'purpose':'audit saved outputs only; no new solver or model calls','hashes':hashes})
t=time.perf_counter(); rows=[]
for m in ['background','low_contrast','high_contrast']:
 f=torch.load(A/f'full_aperture__{m}.pt',weights_only=True,map_location='cpu')
 row={'material':m,'gamma_range':[float(f['gamma'].min()),float(f['gamma'].max())]}
 for name in ['left_half','right_half']:
  d=torch.load(A/f'{name}__{m}.pt',weights_only=True,map_location='cpu');idx=d['action']['rx_elements']
  row[name]={'same_full_trace':torch.equal(f['traces_full'],d['traces_full']),'same_wavefield':torch.equal(f['wavefield_snapshots'],d['wavefield_snapshots']),'exact_receiver_slice':torch.equal(f['traces_full'][idx],d['traces_selected']),'same_gamma':torch.equal(f['gamma'],d['gamma'])}
  assert all(row[name].values())
 rows.append(row)
summary={'status':'completed','rows':rows,'new_physical_calls':0,'new_network_forwards':0,'conclusions':['receiver masking only, no transmit action change','two distinct material inputs, not six independent neural samples','no ensemble or noise replicates: predictive uncertainty uncalibrated','subset cannot add observations after full aperture already observed','phase A physical_seconds includes processing and saving; not solver-only timing','phase A 9 generation plus 3 verification calls per run; two runs consumed 24 calls, not 18','peak argmax shift is not validated echo TOF','gamma .48 has larger distance from background 1 than .52: low/high contrast names misleading'],'audit_seconds':time.perf_counter()-t}
dump('summary.json',summary)
# Prospective selection diagnostic. Outcomes cannot be inferred from saved two-state summaries.
protocol={'status':'planned_not_run','title':'equal-channel receiver selection rank consistency','purpose':'whether frozen R2 preserves physical ranking of left vs right reception for material discrimination','scope':'source0, known center support, discrete material hypothesis discrimination; no blind reconstruction or acquisition savings','hypotheses':[.48,.52],'pilot_channels':[5,18],'remaining_budget':11,'left_action':list(range(0,5))+list(range(6,12)),'right_action':list(range(12,18))+list(range(19,24)),'total_selected_channels':13,'target_materials':[.46,.50,.54],'source':'source0','noise':{'model':'iid Gaussian additive on full record generated once per target and seed','std':'0.01 * RMS of saved full homogeneous trace; same scalar for all channels','seeds':list(range(20))},'methods':['fixed_left','fixed_right','frozen_R2_score','nominal_secant_score','physical_score_oracle_upper_bound','full24_reference'], 'score':'sum of squared difference between hypothesis traces on incremental channels; independent equal variance Gaussian noise assumption; same 11 channels per action; tie chooses left','policy_note':'two hypotheses fixed independent of pilot: static design, NOT adaptive policy; pilot is reserved for later extension, not used to claim adaptation','nominal':'center .5 trace plus original .3/.7 secant times (a-.5); calibration costs separately reported','selection':'freeze selected action before reading target observations; select material hypothesis with minimal unweighted SSE on pilot+chosen channels','evaluation':['per-target absolute material error','agreement with full24 choice','paired method error differences over common noise seeds','R2 vs physical action score ranking','actual vs logical compute ledger'],'interpretation':'targets outside or between hypotheses test bounded discrimination; no claim exact material recovery; action disagreement prevents moving to active loop; agreement only qualifies this two-state observation design','budget':{'new_physical_calls_max':3,'new_network_forwards_max':2,'training_updates':0,'max_seconds':300},'verification':'save predicted hypothesis traces and noisy observations; rederive scores and choices from raw data; hash all consumed PT and source files before run; no posthoc thresholds','cost':'existing calibration files reused with historical cost unknown; cached offline channel masks do not equal hardware acquisition savings','stop':'if shape, source coordinate, background or hashes mismatch save failure and stop; no extra runs to chase success'}
dump('next_protocol.json',protocol)
assert set(protocol['pilot_channels']).isdisjoint(protocol['left_action']);assert set(protocol['pilot_channels']).isdisjoint(protocol['right_action']);assert len(protocol['left_action'])==len(protocol['right_action'])==11
for p,h in hashes.items():assert sha(R/p)==h
dump('verification.json',{'status':'passed','checks':['all 6 half-array outputs exact masks of full outputs','same gamma and full wavefields','13 source hashes unchanged','pilot disjoint and equal action cost'],'limitation':'audit of saved outputs only, not stage A complete acceptance'})
print(json.dumps(summary,ensure_ascii=False,indent=2))

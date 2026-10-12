"""Read-only comparison of simple local material-response baselines."""
import argparse, json
from pathlib import Path
import torch
from run_reduced_material_scan import sha

ROOT=Path(__file__).resolve().parent
SRC=Path('.build/fno-known-support-cross-depth-material-response-20261011')
OUT=Path('.build/simple-separable-response-baseline-20261011')
PLAN=Path('.build/route-evidence-audit-20261011/next_experiment_protocol.json')

def metrics(pred, ref):
    p,r=pred.double().reshape(-1),ref.double().reshape(-1)
    return {'relative_l2':float((p-r).norm()/r.norm()),'amplitude_ratio':float(p.norm()/r.norm()),'cosine':float(torch.dot(p,r)/(p.norm()*r.norm()))}

def main():
    assert not OUT.exists(); OUT.mkdir()
    inputs=[SRC/'raw.pt',SRC/'summary.json',SRC/'protocol.json',PLAN]
    protocol={'status':'frozen_read_only','purpose':'separate constant nominal slope from depth-scalar oracle calibration',
              'inputs_sha256':{str(p):sha(p) for p in inputs},
              'baselines':['zero_response','nominal_constant_secant','depth_scalar_oracle_projection'],
              'oracle_warning':'depth scalar is fit independently to each physical target and is diagnostic only; not a predictive test',
              'scope':'same known 449-pixel support, source0, used development depths -1,+1; no new labels, training or final test'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    raw=torch.load(SRC/'raw.pt',weights_only=True,map_location='cpu')
    out={'status':'completed','rows':[]}
    for depth,entry in raw['depths'].items():
        for eps,e in entry['epsilons'].items():
            for space in ['field','trace']:
                ref=e['physical'][space]
                nominal=raw['baseline_derivative'][space]
                zero=torch.zeros_like(ref)
                alpha=float(torch.dot(ref.double().reshape(-1),nominal.double().reshape(-1))/nominal.double().reshape(-1).dot(nominal.double().reshape(-1)))
                oracle=alpha*nominal
                for name,pred,extra in [('zero_response',zero,{}),('nominal_constant_secant',nominal,{}),('depth_scalar_oracle_projection',oracle,{'alpha':alpha})]:
                    out['rows'].append({'depth':int(depth),'eps':float(eps),'space':space,'baseline':name,**metrics(pred,ref),**extra})
    out['interpretation']={'predictive_baseline':'nominal_constant_secant','oracle_only':'depth_scalar_oracle_projection',
                           'decision_rule':'If oracle projection <=20% while nominal >20%, amplitude-only correction may explain the gap; otherwise scalar separability is insufficient.'}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

def verify():
    p=json.loads((OUT/'protocol.json').read_text())
    for f,h in p['inputs_sha256'].items(): assert sha(Path(f))==h
    s=json.loads((OUT/'summary.json').read_text()); assert s['status']=='completed' and len(s['rows'])==24
    (OUT/'verification.json').write_text(json.dumps({'status':'passed','rows':len(s['rows']),'read_only':True},indent=2))
    print('Verification passed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args();verify() if a.verify else main()

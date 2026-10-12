"""Fixed-budget known-support material-response audit at the two used development depths."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from run_fno_local_contrast import physical_setup, generate
from run_frozen_depth_shift import predict, policy
from run_reduced_material_scan import sha
from basic_fno import BasicFNO

ROOT = Path(__file__).resolve().parent
PLAN = Path('.build/route-evidence-audit-20261011/next_experiment_protocol.json')
BASE = Path('.build/fno-local-contrast-20261009')
DEPTH = Path('.build/fno-depth-interpolation-control-20261009')
BG = Path('.build/operator-pilot-v3-20261008/homogeneous.pt')
WIDE = Path('.build/fno-width8-control-20261009/final.pt')
R2 = Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt')
OUT = Path('.build/fno-known-support-cross-depth-material-response-20261011')

def rel(a, b):
    a, b = a.double(), b.double()
    return float((a-b).norm() / b.norm())

def cosine(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.dot(a,b) / (a.norm()*b.norm()))

def response_metrics(pred, ref):
    return {'relative_l2': rel(pred, ref), 'amplitude_ratio': float(pred.double().norm()/ref.double().norm()), 'cosine': cosine(pred,ref)}

def load_model(path, device):
    ck = torch.load(path, weights_only=True, map_location='cpu')
    model = BasicFNO(width=8, input_channels=8).to(device)
    model.load_state_dict(ck['state_dict']); model.eval(); model.requires_grad_(False)
    return model

def make_data(solver, force, template, shape, rx, ry, device, mask, a):
    # Keep the material field floating point; torch.full_like(bool_mask, a) would silently
    # quantize both perturbation values to True and erase the experimental contrast.
    mask = mask.to(dtype=template['gamma'].dtype)
    t = {**template, 'gamma': torch.where(mask > 0.5, torch.full_like(mask, a), torch.ones_like(mask))}
    d = generate(solver, force, t, shape, rx, ry, device, a)
    assert torch.equal(d['gamma'], t['gamma'])
    return d

def main_run():
    policy(); torch.set_num_threads(4)
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert plan['status'] == 'planned_not_run'
    if OUT.exists():
        # Resume only the preflight-blocked directory; never reuse scientific outputs.
        old = OUT / 'failure_preflight.json'
        assert old.exists() and not (OUT / 'raw.pt').exists() and not (OUT / 'summary.json').exists()
    for p,h in plan['frozen_sha256'].items(): assert sha(Path(p)) == h, p
    OUT.mkdir(exist_ok=True)
    protocol = {'status':'frozen_before_generation', 'plan':plan, 'plan_sha256':sha(PLAN), 'script_sha256':sha(Path(__file__)),
                'inputs_sha256':{str(p):sha(p) for p in [WIDE,R2,BASE/'contrast_03.pt',BASE/'contrast_07.pt',DEPTH/'state_-1.pt',DEPTH/'state_1.pt',BG]},
                'execution':'IEEE float32 physical and neural forwards; float64 metric accumulation; no training',
                'scope':plan['interpretation']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    start=time.perf_counter()
    solver,force,template,shape,rx,ry,device = physical_setup()
    bg = torch.load(BG, weights_only=True, map_location='cpu')
    scale = json.loads((BASE/'data_manifest.json').read_text())['normalization']['wavefield_rms']
    nominal={}
    for a in [.3,.7]:
        nominal[a] = torch.load(BASE/f'contrast_{int(a*10):02d}.pt',weights_only=True,map_location='cpu')
    masks={k:torch.load(DEPTH/f'state_{k}.pt',weights_only=True,map_location='cpu')['gamma']<.99 for k in [-1,1]}
    models={'width8_control':load_model(WIDE,device),'joint_loss_r2':load_model(R2,device)}
    raw={'depths':{},'baseline':{},'elapsed_seconds':None}
    for a in [.3,.7]:
        raw['baseline'][str(a)]={'field':nominal[a]['wavefield_snapshots'][0], 'trace':nominal[a]['traces_full']}
    base_deriv={s:(raw['baseline']['0.7'][s]-raw['baseline']['0.3'][s])/.4 for s in ['field','trace']}
    raw['baseline_derivative']=base_deriv
    for depth in [-1,1]:
        entry={'mask_cells':int(masks[depth].sum()),'epsilons':{}}
        for eps in [.002,.001]:
            plus=make_data(solver,force,template,shape,rx,ry,device,masks[depth],.5+eps)
            minus=make_data(solver,force,template,shape,rx,ry,device,masks[depth],.5-eps)
            phys={s:(plus['wavefield_snapshots'][0]-minus['wavefield_snapshots'][0])/(2*eps) if s=='field' else (plus['traces_full']-minus['traces_full'])/(2*eps) for s in ['field','trace']}
            predictions={}
            for name,model in models.items():
                pp=predict(model,plus,device,scale,bg); pm=predict(model,minus,device,scale,bg)
                predictions[name]={s:(pp[s]-pm[s])/(2*eps) for s in ['field','trace']}
            e={'physical':phys,'predictions':predictions}
            e['metrics']={name:{s:response_metrics(predictions[name][s],phys[s]) for s in ['field','trace']} for name in models}
            e['baseline_metrics']={s:response_metrics(base_deriv[s],phys[s]) for s in ['field','trace']}
            torch.save({'physical':phys,'predictions':predictions},OUT/f'depth_{depth}_eps_{str(eps).replace(".","p")}.pt')
            entry['epsilons'][str(eps)]=e
            print('depth',depth,'eps',eps,'done',round(time.perf_counter()-start,1),flush=True)
        raw['depths'][str(depth)]=entry
    raw['elapsed_seconds']=time.perf_counter()-start
    summary={'status':'completed','scope':plan['interpretation'],'elapsed_seconds':raw['elapsed_seconds'],'calls':{'physical':8,'network_forwards':16},'stability':{},'gates':{},'rows':[]}
    for depth,entry in raw['depths'].items():
        p2=entry['epsilons']['0.002']['physical']; p1=entry['epsilons']['0.001']['physical']
        summary['stability'][depth]={s:response_metrics(p2[s],p1[s]) for s in ['field','trace']}
        for model in models:
            q2=entry['epsilons']['0.002']['predictions'][model]; q1=entry['epsilons']['0.001']['predictions'][model]
            summary['stability'][f'{depth}:{model}']={s:response_metrics(q2[s],q1[s]) for s in ['field','trace']}
            for eps in ['0.002','0.001']:
                for s in ['field','trace']:
                    m=entry['epsilons'][eps]['metrics'][model][s]
                    summary['rows'].append({'depth':int(depth),'model':model,'eps':float(eps),'space':s,**m})
    # Relative-L2 stability is the preregistered quantity; zero/near-zero is inconclusive.
    summary['gates']['physical_fd_stable_2pct']={d:all(v[s]['relative_l2']<=.02 for s in ['field','trace']) for d,v in summary['stability'].items() if ':' not in d}
    summary['gates']['network_fd_stable_2pct']={d:all(v[s]['relative_l2']<=.02 for s in ['field','trace']) for d,v in summary['stability'].items() if ':' in d}
    summary['gates']['screen']={r['depth']:False for r in summary['rows']}
    for d in [-1,1]:
        for model in models:
            vals=[r for r in summary['rows'] if r['depth']==d and r['model']==model and r['space']=='trace']
            summary['gates']['screen'][f'{d}:{model}']=all(r['relative_l2']<=.2 and r['cosine']>=.9 for r in vals)
    physical_ok=all(summary['gates']['physical_fd_stable_2pct'].values())
    network_ok=all(summary['gates']['network_fd_stable_2pct'].values())
    screen_ok=all(summary['gates']['screen'].values())
    summary['decision']='screen_passed_known_support_response' if physical_ok and network_ok and screen_ok else ('reference_unstable' if not physical_ok else 'screen_failed_or_network_unstable')
    torch.save(raw,OUT/'raw.pt'); (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

def verify():
    policy(); p=json.loads((OUT/'protocol.json').read_text()); assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in p['inputs_sha256'].items(): assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu'); summary=json.loads((OUT/'summary.json').read_text())
    assert summary['status']=='completed' and len(summary['rows'])==16
    assert raw['elapsed_seconds']>0
    print(json.dumps({'status':'passed','rows':len(summary['rows']),'summary_sha256':sha(OUT/'summary.json'),'decision':summary['decision']},indent=2))
    (OUT/'verification.json').write_text(json.dumps({'status':'passed','protocol_and_input_hashes':'unchanged','row_count':len(summary['rows']),'decision':summary['decision']},indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--verify',action='store_true'); a=ap.parse_args(); verify() if a.verify else main_run()

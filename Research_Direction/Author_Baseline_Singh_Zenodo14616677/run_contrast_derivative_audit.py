"""Frozen contrast tangent and physical-step audit; no training or selection."""
from pathlib import Path
import argparse
import json
import time
import torch
import torch.nn.functional as F
from basic_fno import BasicFNO
from run_fno_local_contrast import samples, physical_setup
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from discrete_reference import DiscreteReference, time_cost
from run_reduced_material_scan import sha, check_sources

ROOT=Path(__file__).resolve().parent
BASE=Path('.build/fno-local-contrast-20261009')
WIDE=Path('.build/fno-width8-control-20261009')
OUT=Path('.build/fno-contrast-derivative-20261009')
PLAN=WIDE/'next_contrast_derivative_protocol.json'
EPS=[.001,.0005]

def f32(a): return float(torch.tensor(a,dtype=torch.float32))
def dump(path,obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def metrics(p,r):
    p,r=p.double().flatten(),r.double().flatten()
    return {'relative_l2':float((p-r).norm()/r.norm()),'norm_ratio':float(p.norm()/r.norm()),
            'cosine':float(torch.dot(p,r)/(p.norm()*r.norm()))}
def cost(p,target): return time_cost(p.double(),target.double(),.1/255,.05/127,30e-9)*1e12

def summarize(raw):
    rows=[];pooled={}
    for method,states in raw['methods'].items():
        ps=[];rs=[]
        for key,v in states.items():
            phys=raw['physical'][key];responses=[]
            for j,e in enumerate(EPS):
                lo,hi=v['endpoints'][j]
                pj=(v['pairs'][j][1].double()-v['pairs'][j][0].double())/(hi-lo)
                rj=(phys['pairs'][j][1]-phys['pairs'][j][0])/(hi-lo)
                ownfd=float((cost(v['pairs'][j][1],raw['target'])-cost(v['pairs'][j][0],raw['target']))/(hi-lo))
                physicalfd=float((cost(phys['pairs'][j][1],raw['target'])-cost(phys['pairs'][j][0],raw['target']))/(hi-lo))
                responses.append({'epsilon':e,**metrics(pj,rj),'own_loss_fd':ownfd,
                    'own_ad_fd_relative_error':abs(ownfd-v['gradient'])/max(abs(v['gradient']),1e-15),
                    'physical_ad_fd_relative_error':abs(physicalfd-phys['gradient'])/max(abs(phys['gradient']),1e-15)})
                if j==1:ps.append(pj);rs.append(rj)
            def tangent(z,j):
                lo,hi=z['endpoints'][j]
                return (z['pairs'][j][1].double()-z['pairs'][j][0].double())/(hi-lo)
            trials=[]
            for t in v['trials']:
                ratio=float(cost(raw['step_predictions'][str(t['a'])],raw['target'])/cost(phys['center'],raw['target']))
                trials.append({**t,'physical_cost_ratio':ratio,'distance_to_target':abs(t['a']-.5)})
            rows.append({'method':method,'state':float(key),'responses':responses,
                'two_epsilon_relative_change':metrics(tangent(v,0),tangent(v,1))['relative_l2'],
                'physical_two_epsilon_relative_change':metrics(tangent(phys,0),tangent(phys,1))['relative_l2'],
                'proxy_loss':float(cost(v['center'],raw['target'])),'physical_loss':float(cost(phys['center'],raw['target'])),
                'proxy_gradient':v['gradient'],'physical_gradient':phys['gradient'],
                'gradient_ratio':v['gradient']/phys['gradient'],'trials':trials})
        pooled[method]=metrics(torch.cat(ps),torch.cat(rs))
    return {'status':'completed','rows':rows,'pooled_epsilon_0005':pooled,'elapsed_seconds':raw['elapsed_seconds'],
            'scope':'Known geometry, source0, two development states and one scalar contrast direction; no recovery or speedup claim'}

def run():
    torch.set_num_threads(4);check_sources()
    plan=json.loads(PLAN.read_text())
    for p,h in plan['frozen_files_sha256'].items():assert sha(Path(p))==h,p
    assert not OUT.exists(),'Never overwrite experiments.'
    OUT.mkdir()
    deps=['basic_fno.py','run_fno_local_contrast.py','time_feature_input.py','operator_data.py',
          'time_reconstruction.py','discrete_reference.py','FiniteDifferencePyTorchConv.py','Utilities.py','run_reduced_material_scan.py']
    dump(OUT/'protocol.json',{'plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
        'dependency_sha256':{n:sha(ROOT/n) for n in deps},'status':'fixed_before_run',
        'execution_details':{'objective_accumulation':'float64 for all methods, including after float32 proxy output; background added after promotion',
         'numerical_qa':'Report both epsilon results; do not retune original 20% / 0.9 response screens',
         'resource_cap':'No training, two physical AD evaluations, twelve proxy endpoint evaluations plus six scalar AD evaluations; physical steps cached; stop on OOM',
         'zero_response_baseline':'Relative response error 1, norm ratio 0, cosine undefined; constant predictor loss gradient zero'}})
    start=time.perf_counter();manifest,groups=samples();s=groups['train'][0];scale=manifest['normalization']['wavefield_rms']
    bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')
    target=groups['train'][1]['traces_full'][None].double()
    solver,force,template,shape,rx,ry,device=physical_setup()
    assert shape[:2]==(255,127),shape
    solver=solver.double();force=force.double();reference=DiscreteReference(solver,rx,ry)
    support=(template['gamma']<.99).to(device)
    def physical(a,grad=False):
        av=torch.tensor(a,dtype=torch.float64,device=device,requires_grad=grad)
        with torch.set_grad_enabled(grad):
            gamma=F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.)
            pred=reference.traces(gamma,force)
            derivative=float(torch.autograd.grad(cost(pred,target.to(device)),av)[0]) if grad else None
        return pred.detach().cpu(),derivative
    raw={'target':target,'physical':{},'methods':{},'step_predictions':{}}
    for state in [.4,.6]:
        a=f32(state);endpoints=[(f32(state-e),f32(state+e)) for e in EPS]
        center,g=physical(a,True);pairs=[]
        for lo,hi in endpoints:pairs.append((physical(lo)[0],physical(hi)[0]))
        raw['physical'][str(a)]={'center':center,'gradient':g,'pairs':pairs,'endpoints':endpoints}
        print('physical completed',a,g,flush=True)
    ids=s['receiver_indices'].to(device);times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
    background=bg['traces_full'][0:1].to(device).double()
    # Compute receiver slices of the exact frozen normalized field interpolation.
    training=[(x['target'][0][ids.cpu()[:,0],ids.cpu()[:,1],:]-
        bg['wavefield_snapshots'][0][ids.cpu()[:,0],ids.cpu()[:,1],:]/scale).to(device) for x in groups['train']]
    for method,path in [('width4',BASE/'final.pt'),('width8',WIDE/'final.pt'),('piecewise_linear',None)]:
        model=None
        if path:
            ck=torch.load(path,weights_only=True,map_location='cpu')
            model=BasicFNO(width=ck['config']['width'],input_channels=8).to(device)
            model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False);del ck
        def proxy(a,grad=False):
            av=torch.tensor(a,dtype=torch.float32,device=device,requires_grad=grad)
            with torch.set_grad_enabled(grad):
                if model is not None:
                    gamma=torch.where(support,av,1.)
                    prediction=model(encode_time_features(s,gamma=gamma,mode='zero')[None])[0,0]
                    sparse=prediction[ids[:,0],ids[:,1],:]
                else:
                    lo,hi=(0,1) if a<=.5 else (1,2)
                    w=(av-groups['train'][lo]['contrast'])/.2
                    sparse=(1-w)*training[lo]+w*training[hi]
                scatter=cubic_time_reconstruction(sparse*scale,s['time_seconds'].to(device),times)
                pred=scatter[None].double()+background
                derivative=float(torch.autograd.grad(cost(pred,target.to(device)),av)[0]) if grad else None
            return pred.detach().cpu(),derivative
        states={}
        for key,p in raw['physical'].items():
            a=float(key);center,g=proxy(a,True)
            pairs=[(proxy(lo)[0],proxy(hi)[0]) for lo,hi in p['endpoints']]
            trials=[]
            for step in [.02,.01]:
                candidate=f32(max(.3,min(.7,a-step*(1 if g>0 else -1 if g<0 else 0))))
                if str(candidate) not in raw['step_predictions']:raw['step_predictions'][str(candidate)]=physical(candidate)[0]
                trials.append({'requested_step':step,'a':candidate,'actual_step':candidate-a})
            states[key]={'center':center,'gradient':g,'pairs':pairs,'endpoints':p['endpoints'],'trials':trials}
            print(method,a,g,flush=True)
        raw['methods'][method]=states
        del model
        torch.cuda.empty_cache()
    raw['elapsed_seconds']=time.perf_counter()-start
    torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summarize(raw))
    print(json.dumps(summarize(raw)['pooled_epsilon_0005']),flush=True)

def verify():
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['script_sha256'] and sha(PLAN)==p['plan_sha256']
    for name,h in p['dependency_sha256'].items():assert sha(ROOT/name)==h
    for name,h in p['plan']['frozen_files_sha256'].items():assert sha(Path(name))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')
    assert summarize(raw)==json.loads((OUT/'summary.json').read_text())
    assert len(raw['methods'])==3 and len(raw['physical'])==2
    for row in summarize(raw)['rows']:
        assert len(row['trials'])==2 and len(row['responses'])==2
        assert all(torch.isfinite(torch.tensor(v)) for v in [row['proxy_gradient'],row['physical_gradient']])
    dump(OUT/'verification.json',{'status':'passed','immutable_inputs':'14 SHA checks passed',
        'code_dependencies':'SHA checks passed','metrics':'exactly recomputed from saved endpoint/center/step tensors',
        'coverage':'all 3 methods, 2 states, 2 epsilons and 2 steps reported',
        'raw_sha256':sha(OUT/'raw.pt'),'summary_sha256':sha(OUT/'summary.json'),
        'limitation':'Numerical AD/finite-difference residuals are reported in summary; passed here means artifact consistency, not physical accuracy acceptance'})
    print('Verification passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    verify() if args.verify else run()

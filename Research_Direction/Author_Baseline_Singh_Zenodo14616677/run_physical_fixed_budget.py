"""Exactly8 updates per route; caches isolated by method and observation."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_contrast_derivative_audit import ROOT,cost,dump
from run_physical_single_step import proposal
from run_fno_local_contrast import physical_setup
from discrete_reference import DiscreteReference
from run_reduced_material_scan import sha
OLD=Path('.build/fno-physical-single-step-control-20261009')
VALID=Path('.build/fno-frozen-new-contrast-validation-20261009')
OUT=Path('.build/fno-physical-fixed-budget-control-20261009')
PLAN=OLD/'next_fixed_budget_protocol.json'
def summarize(raw):
    routes=[];totals={}
    for route in raw['routes']:
        key=route['target'];obs=raw['observations'][key];records=[]
        for state in route['states']:
            a=state['a'];pred=route['cache'][str(a)];e=pred-obs
            records.append({**state,'cost':float(cost(pred,obs)),
                'contrast_error':abs(a-float(key)),'material_half_mse':.5*(a-float(key))**2*raw['support_fraction'],
                'full_receiver_error':float(e.norm()/obs.norm()),'scatter_receiver_error':float(e.norm()/(obs-raw['background']).norm())})
        cost_increases=[i for i in range(1,9) if records[i]['cost']>records[i-1]['cost']]
        result={'method':route['method'],'target':float(key),'records':records,'cost_increase_updates':cost_increases,
                'actual_physical_calls':len(route['cache']),'cache_hits':route['cache_hits'],'elapsed_seconds':route['elapsed_seconds'],
                'clips':sum(abs(s['raw_delta'])>.02 for s in route['states'][1:])}
        routes.append(result)
        totals.setdefault(route['method'],{'physical_calls':0,'cache_hits':0,'physical_route_seconds':0.})
        t=totals[route['method']];t['physical_calls']+=len(route['cache']);t['cache_hits']+=route['cache_hits'];t['physical_route_seconds']+=route['elapsed_seconds']
    for m,t in totals.items():
        stage=raw['first_stage_seconds'].get(m,0.) if m!='fixed05' else 0.
        t['historical_first_stage_grid_seconds']=stage
        t['descriptive_online_seconds_sum']=t['physical_route_seconds']+stage
    return {'status':'completed','routes':routes,'method_totals':totals,'logical_physical_calls':384,
        'actual_physical_calls':sum(len(r['cache']) for r in raw['routes']),'elapsed_seconds':raw['elapsed_seconds'],
        'primary_update':8,'evaluation_updates':[0,1,2,4,8],
        'offline_costs':{'network_training_and_four_evaluations_seconds':532.9938059000124,
                        'training_data_generation_seconds':None,'linear_training_reference_generation_seconds':None},
        'timing_limit':'First-stage historical run plus current sequential route sums, not an independently measured full pipeline benchmark; offline missing costs unknown',
        'scope':'Fixed8 known-support scalar updates on repeatedly used diagnostic observations; not blind FWI or independent final testing'}
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());previous=json.loads((OLD/'protocol.json').read_text());frozen={**previous['frozen_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    for n,h in previous['dependencies'].items():assert sha(ROOT/n)==h
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'status':'fixed_before_run','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
        'proposal_script_sha256':sha(Path(__file__).with_name('run_physical_single_step.py')),'frozen_sha256':frozen,'dependencies':previous['dependencies'],
        'execution':'Same float64 physical solver; cache local to each route; all8 steps, no convergence stopping',
        'GPU':torch.cuda.get_device_name(),'torch':torch.__version__})
    start=time.perf_counter();observations=torch.load(VALID/'observations.pt',weights_only=True,map_location='cpu')
    prior=json.loads((VALID/'summary.json').read_text());single=torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu')
    solver,force,template,shape,rx,ry,device=physical_setup();solver.double();force=force.double();ref=DiscreteReference(solver,rx,ry)
    support=(template['gamma']<.99).to(device)
    raw={'observations':observations,'background':single['background'],'support_fraction':single['support_fraction'],'routes':[],
         'first_stage_seconds':prior['method_seconds']}
    for key,obs in observations.items():
        for method in ['width8','piecewise_linear','fixed05']:
            a=.5 if method=='fixed05' else next(x['selected_a'] for x in prior['rows'] if str(x['target'])==key and x['method']==method)
            route={'target':key,'method':method,'cache':{},'cache_hits':0,'states':[{'update':0,'a':a}]}
            def evaluate(value):
                if time.perf_counter()-start>600:raise TimeoutError('10-minute cap')
                if str(value) in route['cache']:route['cache_hits']+=1;return route['cache'][str(value)]
                with torch.no_grad():
                    av=torch.tensor(value,device=device,dtype=torch.float64)
                    pred=ref.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu()
                assert torch.isfinite(pred).all();route['cache'][str(value)]=pred;return pred
            torch.cuda.synchronize();t=time.perf_counter()
            for update in range(1,9):
                lo,hi=a-.002,a+.002;center,minus,plus=evaluate(a),evaluate(lo),evaluate(hi)
                delta,clipped,a1,norm=proposal(center,minus,plus,obs,a);evaluate(a1)
                route['states'].append({'update':update,'a':a1,'previous_a':a,'lo':lo,'hi':hi,'raw_delta':delta,
                    'clipped_delta':clipped,'response_time_norm':norm,'boundary_clipped':a+clipped!=a1})
                if update==1:
                    s=next(x for x in single['trials'] if x['target']==key and x['method']==method)
                    assert a1==s['a1'] and torch.equal(route['cache'][str(a1)],single['cache'][str(a1)])
                a=a1
            torch.cuda.synchronize();route['elapsed_seconds']=time.perf_counter()-t
            raw['routes'].append(route);torch.save(route,OUT/f"route_{method}_{key}.pt")
            print(method,key,'final',a,'calls',len(route['cache']),'seconds',route['elapsed_seconds'],flush=True)
    raw['elapsed_seconds']=time.perf_counter()-start;torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summarize(raw))
def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    assert sha(Path(__file__).with_name('run_physical_single_step.py'))==p['proposal_script_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert summarize(raw)==json.loads((OUT/'summary.json').read_text())
    assert len(raw['routes'])==12
    for r in raw['routes']:
        assert len(r['states'])==9 and len(r['cache'])+r['cache_hits']==32
        for s in r['states'][1:]:
            vals=proposal(r['cache'][str(s['previous_a'])],r['cache'][str(s['lo'])],r['cache'][str(s['hi'])],raw['observations'][r['target']],s['previous_a'])
            assert list(vals)==[s['raw_delta'],s['clipped_delta'],s['a'],s['response_time_norm']]
    solver,force,template,shape,rx,ry,device=physical_setup();solver.double();force=force.double();ref=DiscreteReference(solver,rx,ry);support=(template['gamma']<.99).to(device)
    with torch.no_grad():
        for r in raw['routes']:
            av=torch.tensor(r['states'][-1]['a'],device=device,dtype=torch.float64)
            again=ref.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu()
            assert torch.equal(again,r['cache'][str(float(av))])
    dump(OUT/'verification.json',{'status':'passed','all96_updates':'formulas exactly recomputed','all_metrics':'exact cached recomputation',
        'all12_final_predictions':'bit identical physical regeneration','first_update':'prior single-step parity',
        'inputs_dependencies':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt')})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

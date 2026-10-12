"""One preregistered physical Gauss-Newton scalar update; no selection."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_contrast_derivative_audit import ROOT,cost,dump
from run_fno_local_contrast import physical_setup
from discrete_reference import DiscreteReference
from run_reduced_material_scan import sha
OLD=Path('.build/fno-frozen-new-contrast-validation-20261009')
OUT=Path('.build/fno-physical-single-step-control-20261009')
PLAN=OLD/'next_physical_step_protocol.json'
def inner(a,b):return torch.trapezoid(a*b,dx=30e-9,dim=-1).sum()
def proposal(center,lo,hi,obs,a0):
    j=(hi-lo)/.004;den=inner(j,j)
    assert torch.isfinite(den) and den>0
    delta=-float(inner(j,center-obs)/den);clipped=max(-.02,min(.02,delta));a1=max(.3,min(.7,a0+clipped))
    return delta,clipped,a1,float(den.sqrt())
def summarize(raw):
    rows=[]
    for trial in raw['trials']:
        obs=raw['observations'][trial['target']];a0=trial['a0'];a1=trial['a1']
        center=raw['cache'][str(a0)];final=raw['cache'][str(a1)]
        e0=center-obs;e1=final-obs;truth=float(trial['target']);c0=float(cost(center,obs));c1=float(cost(final,obs))
        rows.append({**trial,'target':truth,'cost_before':c0,'cost_after':c1,'cost_ratio':c1/c0,
            'contrast_error_before':abs(a0-truth),'contrast_error_after':abs(a1-truth),
            'material_half_mse_before':.5*(a0-truth)**2*raw['support_fraction'],
            'material_half_mse_after':.5*(a1-truth)**2*raw['support_fraction'],
            'full_receiver_error_before':float(e0.norm()/obs.norm()),'full_receiver_error_after':float(e1.norm()/obs.norm()),
            'scatter_receiver_error_before':float(e0.norm()/(obs-raw['background']).norm()),
            'scatter_receiver_error_after':float(e1.norm()/(obs-raw['background']).norm()),
            'step_clipped':abs(trial['raw_delta'])>.02,'boundary_clipped':a0+trial['clipped_delta']!=a1,
            'cost_decreased':c1<c0,'material_improved':abs(a1-truth)<abs(a0-truth)})
    return {'status':'completed','rows':rows,'logical_physical_calls':48,'actual_unique_physical_calls':len(raw['cache']),
       'cache_hits':raw['cache_hits'],'elapsed_seconds':raw['elapsed_seconds'],
       'first_stage_seconds':raw['first_stage_seconds'],
       'scope':'One scalar physical step, known geometry, used development observations. No complete optimization or total-cost claim.'}
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());previous=json.loads((OLD/'protocol.json').read_text());frozen={**previous['frozen_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    for n,h in previous['dependencies'].items():assert sha(ROOT/n)==h
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'status':'fixed_before_update','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
       'frozen_sha256':frozen,'dependencies':previous['dependencies'],'execution':'float64 promoted original coefficients; four CPU threads; cache shared scalar states only within this run',
       'GPU':torch.cuda.get_device_name(),'torch':torch.__version__})
    start=time.perf_counter();old=torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu');summary=json.loads((OLD/'summary.json').read_text())
    solver,force,template,shape,rx,ry,device=physical_setup();solver.double();force=force.double();ref=DiscreteReference(solver,rx,ry)
    support=(template['gamma']<.99).to(device)
    raw={'observations':old['observations'],'background':old['background'],'support_fraction':old['support_fraction'],
         'cache':{},'trials':[],'cache_hits':0,'first_stage_seconds':summary['method_seconds']}
    def evaluate(a):
        if time.perf_counter()-start>300:raise TimeoutError('5-minute cap')
        if str(a) in raw['cache']:raw['cache_hits']+=1;return raw['cache'][str(a)]
        with torch.no_grad():
            av=torch.tensor(a,device=device,dtype=torch.float64)
            pred=ref.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu()
        assert torch.isfinite(pred).all();raw['cache'][str(a)]=pred;return pred
    for key,obs in raw['observations'].items():
        for method in ['width8','piecewise_linear','fixed05']:
            a0=.5 if method=='fixed05' else next(x['selected_a'] for x in summary['rows'] if str(x['target'])==key and x['method']==method)
            lo,hi=a0-.002,a0+.002;center,minus,plus=evaluate(a0),evaluate(lo),evaluate(hi)
            delta,clipped,a1,norm=proposal(center,minus,plus,obs,a0);evaluate(a1)
            raw['trials'].append({'method':method,'target':key,'a0':a0,'lo':lo,'hi':hi,'a1':a1,'raw_delta':delta,'clipped_delta':clipped,'response_time_norm':norm})
            print(method,key,a0,'->',a1,flush=True)
    raw['elapsed_seconds']=time.perf_counter()-start;torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summarize(raw))
def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert summarize(raw)==json.loads((OUT/'summary.json').read_text())
    for t in raw['trials']:
        delta,clipped,a1,norm=proposal(raw['cache'][str(t['a0'])],raw['cache'][str(t['lo'])],raw['cache'][str(t['hi'])],raw['observations'][t['target']],t['a0'])
        assert [delta,clipped,a1,norm]==[t['raw_delta'],t['clipped_delta'],t['a1'],t['response_time_norm']]
    assert len(raw['trials'])==12 and len(raw['cache'])+raw['cache_hits']==48
    solver,force,template,shape,rx,ry,device=physical_setup();solver.double();force=force.double();ref=DiscreteReference(solver,rx,ry);support=(template['gamma']<.99).to(device)
    with torch.no_grad():
        # Recompute every actual post-update result, including sub-grid states.
        for t in raw['trials']:
            av=torch.tensor(t['a1'],device=device,dtype=torch.float64)
            pred=ref.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu()
            assert torch.equal(pred,raw['cache'][str(t['a1'])])
    dump(OUT/'verification.json',{'status':'passed','all_metrics_and_steps':'exact recomputation','all12_post_update_physical_predictions':'bit identical regenerated',
        'inputs_dependencies':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'limitation':'Same stencil; no independent physics or network Jacobian test'})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

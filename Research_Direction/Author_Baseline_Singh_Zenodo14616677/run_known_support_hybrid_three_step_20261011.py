"""Fixed three-step known-support hybrid-vs-physical scalar recovery comparison."""
import argparse,json,time
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy
from run_reduced_material_scan import sha

ROOT=Path(__file__).resolve().parent
RESP=Path('.build/fno-known-support-cross-depth-material-response-20261011')
DEPTH=Path('.build/fno-depth-interpolation-control-20261009')
BASE=Path('.build/fno-local-contrast-20261009')
OUT=Path('.build/fno-known-support-hybrid-three-step-20261011')
PLAN=Path('.build/route-evidence-audit-20261011/next_experiment_protocol.json')

def cost(x,y): return float((x.double()-y.double()).norm()/y.double().norm())
def step(j,r,clip=.02):
    j,r=j.double().reshape(-1),r.double().reshape(-1)
    return max(-clip,min(clip,float(torch.dot(j,r)/torch.dot(j,j))))

def main():
    policy(); assert not OUT.exists(); OUT.mkdir(); start=time.perf_counter()
    frozen=[RESP/'raw.pt',RESP/'summary.json',RESP/'protocol.json',DEPTH/'state_-1.pt',DEPTH/'state_1.pt',BASE/'contrast_03.pt',BASE/'contrast_07.pt',PLAN]
    protocol={'status':'frozen_before_run','target_a':.502,'initial_a':.5,'iterations':3,'eps_physical_fd':.001,'clip':.02,
      'methods':['frozen_joint_loss_r2','nominal_constant_secant','recomputed_physical_fd'],
      'stopping':'fixed three iterations; no true-material stopping or target-based threshold selection',
      'cost':'count every new physical forward; cached proxy direction incurs zero new physical calls but its prior generation is reported separately',
      'scope':'known 449-pixel support, source0, used development depths -1,+1; not blind or independent final test',
      'inputs_sha256':{str(p):sha(p) for p in frozen}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    raw=torch.load(RESP/'raw.pt',weights_only=True,map_location='cpu');solver,force,template,shape,rx,ry,device=physical_setup()
    nominal={a:torch.load(BASE/f'contrast_{int(a*10):02d}.pt',weights_only=True,map_location='cpu')['traces_full'] for a in [.3,.7]}; jnom=(nominal[.7]-nominal[.3])/.4
    report={'status':'completed','depths':{},'elapsed_seconds':None}
    for depth in [-1,1]:
        state=torch.load(DEPTH/f'state_{depth}.pt',weights_only=True,map_location='cpu'); mask=(state['gamma']<.99); mf=mask.to(dtype=template['gamma'].dtype)
        material=lambda a: torch.where(mf>0.5,torch.full_like(mf,a),torch.ones_like(mf))
        center=state['physical']['trace']; target=generate(solver,force,{**template,'gamma':material(.502)},shape,rx,ry,device,.502)['traces_full'];
        proxy=raw['depths'][str(depth)]['epsilons']['0.001']['predictions']['joint_loss_r2']['trace']; rows=[]
        for method in protocol['methods']:
            a=.5; trace=center; physical_calls=1; hist=[{'iteration':0,'a':a,'cost':cost(trace,target),'physical_calls':physical_calls}]
            for it in range(1,4):
                residual=target-trace
                if method=='frozen_joint_loss_r2': j=proxy
                elif method=='nominal_constant_secant': j=jnom
                else:
                    plus=generate(solver,force,{**template,'gamma':material(a+.001)},shape,rx,ry,device,a+.001)['traces_full']; minus=generate(solver,force,{**template,'gamma':material(a-.001)},shape,rx,ry,device,a-.001)['traces_full']; j=(plus-minus)/.002; physical_calls+=2
                da=step(j,residual); newa=a+da
                d=generate(solver,force,{**template,'gamma':material(newa)},shape,rx,ry,device,newa)['traces_full']; physical_calls+=1
                a,trace=newa,d; hist.append({'iteration':it,'a':a,'step':da,'cost':cost(trace,target),'physical_calls':physical_calls,'response_source':'frozen_proxy' if method=='frozen_joint_loss_r2' else ('nominal_saved_secant' if method=='nominal_constant_secant' else 'new_physical_fd')})
            rows.append({'method':method,'history':hist,'final_cost':hist[-1]['cost'],'final_a':a,'final_true_material_error':abs(a-.502),'physical_calls':physical_calls})
        report['depths'][str(depth)]={'rows':rows}
    report['elapsed_seconds']=time.perf_counter()-start; (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

def verify():
    p=json.loads((OUT/'protocol.json').read_text());
    for f,h in p['inputs_sha256'].items(): assert sha(Path(f))==h
    s=json.loads((OUT/'summary.json').read_text()); assert s['status']=='completed' and all(len(x['rows'])==3 for x in s['depths'].values())
    (OUT/'verification.json').write_text(json.dumps({'status':'passed','depths':list(s['depths']),'iterations':3,'inputs':'unchanged'},indent=2)); print('Verification passed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args();verify() if a.verify else main()

"""One-step known-support material recovery trial using frozen response directions."""
import argparse,json,time
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import predict,policy
from run_reduced_material_scan import sha

ROOT=Path(__file__).resolve().parent
RESP=Path('.build/fno-known-support-cross-depth-material-response-20261011')
DEPTH=Path('.build/fno-depth-interpolation-control-20261009')
BASE=Path('.build/fno-local-contrast-20261009')
OUT=Path('.build/fno-known-support-scalar-recovery-trial-20261011')
PLAN=Path('.build/route-evidence-audit-20261011/next_experiment_protocol.json')

def cost(x,y):
    return float((x.double()-y.double()).norm()/(y.double().norm()))
def ls_step(j,r,lo=-.02,hi=.02):
    j,r=j.double().reshape(-1),r.double().reshape(-1)
    return max(lo,min(hi,float(torch.dot(j,r)/torch.dot(j,j))))

def main():
    policy(); assert not OUT.exists(); OUT.mkdir()
    frozen=[RESP/'raw.pt',RESP/'summary.json',RESP/'protocol.json',DEPTH/'state_-1.pt',DEPTH/'state_1.pt',BASE/'contrast_03.pt',BASE/'contrast_07.pt',PLAN]
    protocol={'status':'frozen_before_trial','target_material':.502,'initial_material':.5,'steps_clipped_to':[-.02,.02],
      'methods':['frozen_width8_response','frozen_joint_loss_r2_response','nominal_constant_secant','physical_response_oracle'],
      'evaluation':'physical receiver relative L2 to target waveform; target waveform used only as observation for residual step, true a reported descriptively',
      'scope':'known 449-pixel support, source0, depths -1,+1; one scalar trial step; not blind recovery or independent test',
      'inputs_sha256':{str(p):sha(p) for p in frozen}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    raw=torch.load(RESP/'raw.pt',weights_only=True,map_location='cpu'); solver,force,template,shape,rx,ry,device=physical_setup()
    masks={k:torch.load(DEPTH/f'state_{k}.pt',weights_only=True,map_location='cpu')['gamma']<.99 for k in [-1,1]}
    nominal={a:torch.load(BASE/f'contrast_{int(a*10):02d}.pt',weights_only=True,map_location='cpu')['traces_full'] for a in [.3,.7]}
    jnom=(nominal[.7]-nominal[.3])/.4
    models={'width8':raw['depths']['-1']['epsilons']['0.001']['predictions']['width8_control']['trace'],
            'joint_loss_r2':raw['depths']['-1']['epsilons']['0.001']['predictions']['joint_loss_r2']['trace']}
    out={'status':'completed','rows':[]}; start=time.perf_counter()
    for depth in [-1,1]:
        mask=masks[depth]; center=torch.load(DEPTH/f'state_{depth}.pt',weights_only=True,map_location='cpu')['physical']['trace']
        def material(a):
            mf=mask.to(dtype=template['gamma'].dtype)
            return torch.where(mf>0.5,torch.full_like(mf,a),torch.ones_like(mf))
        target=generate(solver,force,{**template,'gamma':material(.502)},shape,rx,ry,device,.502)['traces_full']
        residual=target-center
        physj=raw['depths'][str(depth)]['epsilons']['0.001']['physical']['trace']
        dirs={'width8':raw['depths'][str(depth)]['epsilons']['0.001']['predictions']['width8_control']['trace'],
              'joint_loss_r2':raw['depths'][str(depth)]['epsilons']['0.001']['predictions']['joint_loss_r2']['trace'],
              'nominal_constant_secant':jnom,'physical_response_oracle':physj}
        for name,j in dirs.items():
            step=ls_step(j,residual); a=.5+step
            d=generate(solver,force,{**template,'gamma':material(a)},shape,rx,ry,device,a)
            out['rows'].append({'depth':depth,'method':name,'step':step,'candidate_a':a,'initial_cost':cost(center,target),'candidate_cost':cost(d['traces_full'],target),'true_material_error':abs(a-.502),'response_norm':float(j.double().norm()),'residual_norm':float(residual.double().norm())})
            print(depth,name,'step',step,'cost',out['rows'][-1]['initial_cost'],'->',out['rows'][-1]['candidate_cost'],flush=True)
    out['elapsed_seconds']=time.perf_counter()-start; (OUT/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

def verify():
    p=json.loads((OUT/'protocol.json').read_text());
    for f,h in p['inputs_sha256'].items(): assert sha(Path(f))==h
    s=json.loads((OUT/'summary.json').read_text()); assert s['status']=='completed' and len(s['rows'])==8
    (OUT/'verification.json').write_text(json.dumps({'status':'passed','rows':len(s['rows']),'inputs':'unchanged'},indent=2)); print('Verification passed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');a=ap.parse_args();verify() if a.verify else main()

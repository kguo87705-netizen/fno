"""Frozen audit of Adam preconditioned directions; no parameter updates."""
import argparse,json,time
from pathlib import Path
import torch
from run_paired_depth_coverage import policy,data_setup,BasicFNO,ROOT,state_hash,encode_time_features,supervised_losses,dump,sha
OUT=Path('.build/fno-adam-direction-audit-20261010')
PLAN=OUT/'preregistered_protocol.json'
KEYS=['depth-2','depth0','depth2','contrast03','contrast07']
def rv(x):
    return (torch.view_as_real(x).reshape(-1) if x.is_complex() else x.reshape(-1)).detach().cpu().float()
def direction(ck,model):
    st=ck['optimizer_state_dict']['state'];step=400.;b1=.9;b2=.999;eps=1e-8
    out=[]
    for i,p in enumerate(model.parameters()):
        q=st[i];m=q['exp_avg'];v=q['exp_avg_sq']
        if p.is_complex(): m=torch.view_as_real(m);v=torch.view_as_real(v)
        mh=m/(1-b1**step);vh=v/(1-b2**step);d=-mh/(torch.sqrt(vh)+eps);out.append(d.reshape(-1).detach().cpu().float())
    return torch.cat(out)
def stat(g,d):
    gn=float(g.norm());dn=float(d.norm());dot=float(torch.dot(g,d));return {'gradient_norm':gn,'adam_descent_norm':dn,'dot_gradient_descent':dot,'cos_gradient_descent':dot/(gn*dn) if min(gn,dn)>0 else None,'cos_negative_gradient_descent':-dot/(gn*dn) if min(gn,dn)>0 else None}
def run():
    policy();plan=json.loads(PLAN.read_text(encoding='utf-8'));assert not (OUT/'summary.json').exists()
    for p,h in plan['frozen_sha256'].items():assert sha(Path(p))==h
    OUT.mkdir(exist_ok=True)
    device,nom,bg,scale,sr,samples=data_setup();ids=nom['receiver_physical_indices'].to(device);start=time.perf_counter();rows={}
    for name in ['loss_original','loss_field_only']:
        ck=torch.load(Path(plan['checkpoints'][name]),weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();assert state_hash(model)==ck['state_hash'];d=direction(ck,model);params=tuple(model.parameters());rows[name]=[]
        for key in KEYS:
            s={'gamma':samples[key]['gamma'],'source_xy_m':nom['source_xy_m'][0],'x_m':nom['x_m'],'y_m':nom['y_m'],'time_seconds':nom['time_seconds_snapshots']};x=encode_time_features(s,mode='zero')[None].to(device);target=(samples[key]['physical']['field'].to(device)/scale-bg['wavefield_snapshots'][0].to(device)/scale)[None,None];pred=model(x);lf,lr=supervised_losses(pred,target,ids,scale,sr);gf=torch.cat([rv(g) for g in torch.autograd.grad(lf,params,retain_graph=True)]);gr=torch.cat([rv(g) for g in torch.autograd.grad(lr,params)]);gt=gf+gr
            rows[name].append({'state':key,'field_receiver':{'field_vs_adam':stat(gf,d),'receiver_vs_adam':stat(gr,d),'total_vs_adam':stat(gt,d)},'field_loss':float(lf.detach()),'receiver_loss':float(lr.detach()),'real_coordinates':int(gf.numel())});del x,target,pred,lf,lr,gf,gr,gt
        assert state_hash(model)==ck['state_hash'];del model,ck,d;torch.cuda.empty_cache()
    raw={'status':'completed','models':rows,'elapsed_seconds':time.perf_counter()-start,'calls':{'forward':10,'backward':20,'optimizer':0,'physical':0},'scope':'Adam step400 moments reconstructed exactly in real/imag view; current objective gradients; no update or physical derivative.'};torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',raw);print('Completed')
def verify():
    policy();p=json.loads(PLAN.read_text(encoding='utf-8'))
    for f,h in p['frozen_sha256'].items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert raw==json.loads((OUT/'summary.json').read_text(encoding='utf-8'));assert raw['calls']=={'forward':10,'backward':20,'optimizer':0,'physical':0}
    assert all(abs(r['field_receiver']['field_vs_adam']['cos_gradient_descent'])<=1+1e-5 for m in raw['models'].values() for r in m)
    dump(OUT/'verification.json',{'status':'passed','saved_summary':'Exact raw recomputation','optimizer_updates':0,'physical_calls':0,'protocol_sha256':sha(PLAN),'limits':'No finite-difference or physical derivative validation;one final state per arm'})
    print('Verification passed')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');x=a.parse_args();verify() if x.verify else run()

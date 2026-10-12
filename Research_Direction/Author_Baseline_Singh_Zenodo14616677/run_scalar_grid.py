"""Fixed41-point scalar recovery. Selection uses observations only."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_contrast_derivative_audit import ROOT,BASE,WIDE,cost,dump,f32
from run_fno_local_contrast import samples,physical_setup
from discrete_reference import DiscreteReference
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_reduced_material_scan import sha
OUT=Path('.build/fno-scalar-grid-recovery-20261009')
PREV=Path('.build/fno-contrast-precision-control-20261009')
PLAN=PREV/'next_scalar_recovery_protocol.json'

def report(raw):
    curves={};selected=[];grid=raw['grid'];fd=raw['predictions']['physical']
    for m,preds in raw['predictions'].items():
        curves[m]={}
        for key,target in raw['targets'].items():
            values=[float(cost(p,target)) for p in preds];curves[m][key]=values
            index=min(range(41),key=lambda i:(values[i],grid[i]))
            physical_values=[float(cost(p,target)) for p in fd]
            fi=min(range(41),key=lambda i:(physical_values[i],grid[i]))
            a=grid[index];truth=float(key);error=preds[index]-target;pe=fd[index]-target
            selected.append({'method':m,'target':truth,'index':index,'selected_a':a,
                'absolute_contrast_error':abs(a-truth),'material_half_mse':.5*(a-truth)**2*raw['support_fraction'],
                'selected_proxy_cost':values[index],'physical_cost':physical_values[index],
                'physical_cost_over_initial_a05':physical_values[index]/physical_values[20],
                'full_receiver_error':float(error.norm()/target.norm()),
                'scatter_receiver_error':float(error.norm()/(target-raw['background']).norm()),
                'physical_full_receiver_error':float(pe.norm()/target.norm()),
                'physical_scatter_receiver_error':float(pe.norm()/(target-raw['background']).norm()),
                'gap_to_physical_grid_a':abs(a-grid[fi])})
    return {'status':'completed','curves':curves,'selected':selected,'elapsed_seconds':raw['elapsed_seconds'],
        'method_seconds':raw['method_seconds'],'calls':{m:len(p) for m,p in raw['predictions'].items()},
        'constant_training_a05_cost':{k:float(cost(raw['constant'],t)) for k,t in raw['targets'].items()},
        'constant_selection':'All candidates tie; no meaningful material recovery',
        'offline_costs':{'width8_training_and_four_evaluations_seconds':532.9938059000124,'data_generation_seconds':None,
                         'note':'Historical single run; no total-cost speedup claim'},
        'scope':'Known support scalar grid development diagnostic, resolution0.01; not blind FWI or independent test'}

def run():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());prior=json.loads((Path('.build/fno-contrast-derivative-20261009')/'protocol.json').read_text())
    frozen={**prior['plan']['frozen_files_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    for n,h in prior['dependency_sha256'].items():assert sha(ROOT/n)==h
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
        'frozen_sha256':frozen,'dependencies':prior['dependency_sha256'],'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'matmul_tf32':False,'cudnn_tf32':False,'status':'fixed_before_run'})
    start=time.perf_counter();manifest,groups=samples();s=groups['train'][0];scale=manifest['normalization']['wavefield_rms']
    bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')
    old=torch.load(PREV/'raw.pt',weights_only=True,map_location='cpu')
    solver,force,template,shape,rx,ry,device=physical_setup();solver.double();force=force.double()
    ref=DiscreteReference(solver,rx,ry);support=(s['gamma']<.99).to(device);ids=s['receiver_indices'].to(device);ci=ids.cpu()
    background=bg['traces_full'][0:1].double();times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
    training=[(x['target'][0][ci[:,0],ci[:,1],:]-bg['wavefield_snapshots'][0][ci[:,0],ci[:,1],:]/scale).to(device) for x in groups['train']]
    grid=[f32(.30+i*.01) for i in range(41)]
    raw={'grid':grid,'targets':{str(f32(x['contrast'])):x['traces_full'][None].double() for x in groups['development_interpolation']},
        'background':background,'support_fraction':float(support.sum())/support.numel(),'predictions':{},'method_seconds':{},
        'constant':groups['train'][1]['traces_full'][None].double()}
    for method in ['width8','piecewise_linear','physical']:
        model=None
        if method=='width8':
            ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device)
            model.load_state_dict(ck['state_dict']);model.eval();del ck
        torch.cuda.synchronize();t=time.perf_counter();preds=[]
        with torch.no_grad():
            for a in grid:
                if time.perf_counter()-start>600:raise TimeoutError('10-minute cap')
                av=torch.tensor(a,device=device,dtype=torch.float64 if method=='physical' else torch.float32)
                if method=='physical':
                    gamma=F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.)
                    p=ref.traces(gamma,force).cpu()
                else:
                    if model is not None:
                        field=model(encode_time_features(s,gamma=torch.where(support,av,1.),mode='zero')[None])[0,0]
                        sparse=field[ids[:,0],ids[:,1],:]
                    else:
                        lo,hi=(0,1) if a<=.5 else (1,2);w=(av-groups['train'][lo]['contrast'])/.2
                        sparse=(1-w)*training[lo]+w*training[hi]
                    p=cubic_time_reconstruction(sparse*scale,s['time_seconds'].to(device),times)[None].double().cpu()+background
                    if a in [f32(.4),f32(.6)]:assert torch.equal(p,old['arms']['ieee'][method][str(a)]['center'])
                preds.append(p)
        torch.cuda.synchronize();raw['method_seconds'][method]=time.perf_counter()-t;raw['predictions'][method]=preds
        del model;torch.cuda.empty_cache();print(method,'completed',raw['method_seconds'][method],flush=True)
    raw['elapsed_seconds']=time.perf_counter()-start;torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))

def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['script_sha256'] and sha(PLAN)==p['plan_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert report(raw)==json.loads((OUT/'summary.json').read_text())
    assert len(raw['grid'])==41 and all(len(x)==41 for x in raw['predictions'].values())
    assert all(torch.isfinite(x).all() for ps in raw['predictions'].values() for x in ps)
    dump(OUT/'verification.json',{'status':'passed','metrics_and_argmins':'exact recomputation from all123 saved predictions',
        'frozen_inputs_dependencies':'SHA passed','prior_IEEE_centers':'bit identical checked during run',
        'raw_sha256':sha(OUT/'raw.pt'),'limitation':'Same discrete physics; no independent solver verification'})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

"""Frozen local grid refinement; observations only determine selected index."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_contrast_derivative_audit import ROOT,WIDE,cost,dump,f32
from run_fno_local_contrast import samples,physical_setup
from discrete_reference import DiscreteReference
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_reduced_material_scan import sha
OLD=Path('.build/fno-scalar-grid-recovery-20261009')
OUT=Path('.build/fno-scalar-grid-refinement-20261009')
PLAN=OLD/'next_refinement_protocol.json'

def report(raw):
    rows=[];curves={}
    for key,grid in raw['windows'].items():
        target=raw['targets'][key];curves[key]={}
        fd=[raw['predictions']['physical'][str(a)] for a in grid]
        pc=[float(cost(p,target)) for p in fd];fi=min(range(len(grid)),key=lambda i:(pc[i],grid[i]))
        for m,ps in raw['predictions'].items():
            preds=[ps[str(a)] for a in grid];values=[float(cost(p,target)) for p in preds];curves[key][m]=values
            i=min(range(len(grid)),key=lambda i:(values[i],grid[i]));a=grid[i];err=preds[i]-target;pe=fd[i]-target
            coarse=next(x for x in raw['coarse']['selected'] if x['method']==m and str(x['target'])==key)
            rows.append({'method':m,'target':float(key),'selected_a':a,'boundary_minimum':i in [0,len(grid)-1],
                'absolute_contrast_error':abs(a-float(key)),'material_half_mse':.5*(a-float(key))**2*raw['support_fraction'],
                'coarse_selected_a':coarse['selected_a'],'shift_from_coarse':a-coarse['selected_a'],
                'selected_proxy_cost':values[i],'physical_cost':pc[i],
                'physical_cost_over_initial_a05':pc[i]/float(cost(raw['initial_physical'],target)),
                'full_receiver_error':float(err.norm()/target.norm()),'scatter_receiver_error':float(err.norm()/(target-raw['background']).norm()),
                'physical_full_receiver_error':float(pe.norm()/target.norm()),'physical_scatter_receiver_error':float(pe.norm()/(target-raw['background']).norm()),
                'gap_to_physical_grid_a':abs(a-grid[fi])})
    return {'status':'completed','selected':rows,'curves':curves,'windows':raw['windows'],
        'method_seconds':raw['method_seconds'],'elapsed_seconds':raw['elapsed_seconds'],'calls':111,
        'constant_reference_cost':{k:float(cost(raw['constant'],t)) for k,t in raw['targets'].items()},
        'scope':'Development known support scalar recovery, resolution0.002, no independent test or speedup claim'}

def run():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());previous=json.loads((OLD/'protocol.json').read_text())
    frozen={**previous['frozen_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    for n,h in previous['dependencies'].items():assert sha(ROOT/n)==h
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
        'frozen_sha256':frozen,'dependencies':previous['dependencies'],'status':'fixed_before_run',
        'execution':'IEEE float32 proxies, same promoted float64 stencil, four CPU threads',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    start=time.perf_counter();old=torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu')
    windows={k:[f32(.30+j*.002) for j in range(v['coarse_low_index']*5,v['coarse_high_index']*5+1)] for k,v in plan['windows'].items()}
    grid=sorted(set(a for g in windows.values() for a in g));assert len(grid)==37
    raw={k:old[k] for k in ['targets','background','support_fraction','constant']}
    raw.update(windows=windows,initial_physical=old['predictions']['physical'][20],coarse=json.loads((OLD/'summary.json').read_text()),predictions={},method_seconds={})
    manifest,groups=samples();s=groups['train'][0];scale=manifest['normalization']['wavefield_rms']
    bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')
    solver,force,template,shape,rx,ry,device=physical_setup();solver.double();force=force.double();ref=DiscreteReference(solver,rx,ry)
    support=(s['gamma']<.99).to(device);ids=s['receiver_indices'].to(device);ci=ids.cpu()
    training=[(x['target'][0][ci[:,0],ci[:,1],:]-bg['wavefield_snapshots'][0][ci[:,0],ci[:,1],:]/scale).to(device) for x in groups['train']]
    times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
    parity=[]
    for method in ['width8','piecewise_linear','physical']:
        model=None
        if method=='width8':
            ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();del ck
        torch.cuda.synchronize();t=time.perf_counter();preds={}
        with torch.no_grad():
            for a in grid:
                if time.perf_counter()-start>600:raise TimeoutError('10-minute cap')
                av=torch.tensor(a,device=device,dtype=torch.float64 if method=='physical' else torch.float32)
                if method=='physical':p=ref.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu()
                else:
                    if model is not None:
                        field=model(encode_time_features(s,gamma=torch.where(support,av,1.),mode='zero')[None])[0,0];sparse=field[ids[:,0],ids[:,1],:]
                    else:
                        lo,hi=(0,1) if a<=.5 else (1,2);w=(av-groups['train'][lo]['contrast'])/.2;sparse=(1-w)*training[lo]+w*training[hi]
                    p=cubic_time_reconstruction(sparse*scale,s['time_seconds'].to(device),times)[None].double().cpu()+raw['background']
                if a in old['grid']:
                    assert torch.equal(p,old['predictions'][method][old['grid'].index(a)]),(method,a)
                    parity.append([method,a])
                preds[str(a)]=p
        torch.cuda.synchronize();raw['method_seconds'][method]=time.perf_counter()-t;raw['predictions'][method]=preds
        del model;torch.cuda.empty_cache();print(method,'completed',raw['method_seconds'][method],flush=True)
    raw['overlap_parity']=parity;raw['elapsed_seconds']=time.perf_counter()-start
    torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))

def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['script_sha256'] and sha(PLAN)==p['plan_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert report(raw)==json.loads((OUT/'summary.json').read_text())
    assert all(len(ps)==37 and all(torch.isfinite(v).all() for v in ps.values()) for ps in raw['predictions'].values())
    dump(OUT/'verification.json',{'status':'passed','all111_predictions':'finite, metrics and argmins recomputed exactly',
        'overlap_predictions_bit_identical':len(raw['overlap_parity']),'inputs_and_dependencies':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt')})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

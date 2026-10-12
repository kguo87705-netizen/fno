"""Frozen prospective contrast validation; full common grid, no fitting."""
import argparse,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_contrast_derivative_audit import ROOT,BASE,WIDE,cost,dump,f32
from run_fno_local_contrast import physical_setup
from discrete_reference import DiscreteReference
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_reduced_material_scan import sha
OLD=Path('.build/fno-training-affine-bias-control-20261009')
OUT=Path('.build/fno-frozen-new-contrast-validation-20261009')
PLAN=OLD/'next_frozen_validation_protocol.json'

def report(raw):
    rows=[];curves={};grid=raw['grid'];bg=raw['background'];fd=raw['predictions']['physical'];c=raw['constant_bias'];aff=raw['affine']
    providers={**raw['predictions'],
       'constant_width8':[p-c for p in raw['predictions']['width8']],
       'affine_width8':[p-aff['b0']-(a-aff['mean_a'])*aff['b1'] for a,p in zip(grid,raw['predictions']['width8'])]}
    for key,obs in raw['observations'].items():
        curves[key]={};physical_values=[float(cost(p,obs)) for p in fd]
        pi=min(range(len(grid)),key=lambda i:(physical_values[i],grid[i]));ti=grid.index(float(key));initial=grid.index(f32(.5))
        for method,ps in providers.items():
            values=[float(cost(p,obs)) for p in ps];curves[key][method]=values
            i=min(range(len(grid)),key=lambda i:(values[i],grid[i]));a=grid[i];e=ps[i]-obs;pe=fd[i]-obs;te=ps[ti]-obs
            rows.append({'method':method,'target':float(key),'selected_a':a,'boundary_minimum':i in [0,len(grid)-1],
                'absolute_contrast_error':abs(a-float(key)),'physical_material_half_mse':.5*(a-float(key))**2*raw['support_fraction'],
                'selected_proxy_cost':values[i],'true_physical_cost':physical_values[i],
                'physical_cost_over_initial_a05':physical_values[i]/physical_values[initial],
                'selected_full_receiver_error':float(e.norm()/obs.norm()),'selected_scatter_receiver_error':float(e.norm()/(obs-bg).norm()),
                'true_input_full_receiver_error':float(te.norm()/obs.norm()),'true_input_scatter_receiver_error':float(te.norm()/(obs-bg).norm()),
                'physical_full_receiver_error':float(pe.norm()/obs.norm()),'physical_scatter_receiver_error':float(pe.norm()/(obs-bg).norm()),
                'gap_to_physical_grid_a':abs(a-grid[pi])})
    return {'status':'completed','rows':rows,'curves':curves,'grid':grid,'elapsed_seconds':raw['elapsed_seconds'],
        'method_seconds':raw['method_seconds'],'calls':{'observations_float32':4,'width8':201,'piecewise_linear':201,'physical_float64':201,'total':607},
        'offline_costs':{'width8_prior_training_and_four_evaluations_seconds':532.9938059000124,'prior_training_data_generation_seconds':None},
        'scope':'Frozen prospective validation on4 new contrasts in known training geometry; no independent final test, blind recovery or total-cost claim'}

def run():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());oldp=json.loads(Path('.build/fno-scalar-grid-refinement-20261009/protocol.json').read_text())
    frozen={**oldp['frozen_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    for n,h in oldp['dependencies'].items():assert sha(ROOT/n)==h
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'status':'fixed_before_labels','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
       'dependencies':oldp['dependencies'],'frozen_sha256':frozen,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
       'execution':'IEEE float32 proxies/observations; original coefficients promoted float64 FD; four CPU threads',
       'count_correction_before_run':'201*3+4=607, not605 in prior text; unchanged calls and10-minute cap'})
    start=time.perf_counter()
    def budget():
        if time.perf_counter()-start>600:raise TimeoutError('10-minute cap')
    constant=torch.load('.build/fno-training-bias-control-20261009/calibration.pt',weights_only=True,map_location='cpu')['bias']
    affine=torch.load(OLD/'affine.pt',weights_only=True,map_location='cpu')
    manifest=json.loads((BASE/'data_manifest.json').read_text());scale=manifest['normalization']['wavefield_rms']
    train=[torch.load(BASE/f'contrast_{i:02d}.pt',weights_only=True,map_location='cpu') for i in [3,5,7]]
    d=train[0];s={'gamma':d['gamma'],'source_xy_m':d['source_xy_m'][0],'x_m':d['x_m'],'y_m':d['y_m'],'time_seconds':d['time_seconds_snapshots']}
    bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')
    solver,force,template,shape,rx,ry,device=physical_setup();support=(s['gamma']<.99).to(device)
    reference=DiscreteReference(solver,rx,ry);observations={};t=time.perf_counter()
    with torch.no_grad():
        for a in plan['targets']:
            budget();a=f32(a);av=torch.tensor(a,device=device,dtype=torch.float32)
            gamma=F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.)
            observations[str(a)]=reference.traces(gamma,force).cpu().double()
    torch.save(observations,OUT/'observations.pt');dump(OUT/'observation_commit.json',{'sha256':sha(OUT/'observations.pt'),'generated_once_after_method_freeze':True})
    print('Four observations generated and frozen',flush=True)
    ids=d['receiver_physical_indices'].to(device);ci=ids.cpu();times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
    training=[(x['wavefield_snapshots'][0][ci[:,0],ci[:,1],:]/scale-bg['wavefield_snapshots'][0][ci[:,0],ci[:,1],:]/scale).to(device) for x in train]
    grid=[f32(.3+i*.002) for i in range(201)];background=bg['traces_full'][0:1].double()
    raw={'observations':observations,'grid':grid,'constant_bias':constant,'affine':affine,'background':background,
         'support_fraction':float(support.sum())/support.numel(),'predictions':{},'method_seconds':{'observations':time.perf_counter()-t}}
    # Baseline interpolation arithmetic matches the previous normalized-field protocol.
    prior=torch.load('.build/fno-scalar-grid-refinement-20261009/raw.pt',weights_only=True,map_location='cpu');parity=0
    for method in ['width8','piecewise_linear','physical']:
        model=None
        if method=='width8':
            ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();del ck
        if method=='physical':solver.double();force=force.double()
        torch.cuda.synchronize();t=time.perf_counter();preds=[]
        with torch.no_grad():
            for index,a in enumerate(grid):
                budget();av=torch.tensor(a,device=device,dtype=torch.float64 if method=='physical' else torch.float32)
                if method=='physical':
                    p=reference.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu()
                else:
                    if model is not None:
                        field=model(encode_time_features(s,gamma=torch.where(support,av,1.),mode='zero')[None])[0,0];sparse=field[ids[:,0],ids[:,1],:]
                    else:
                        lo,hi=(0,1) if a<=.5 else (1,2);w=(av-[f32(.3),f32(.5),f32(.7)][lo])/.2
                        # Previous code used literal training contrasts .3,.5,.7; preserve them.
                        w=(av-[.3,.5,.7][lo])/.2;sparse=(1-w)*training[lo]+w*training[hi]
                    p=cubic_time_reconstruction(sparse*scale,s['time_seconds'].to(device),times)[None].double().cpu()+background
                if str(a) in prior['predictions'][method]:
                    assert torch.equal(p,prior['predictions'][method][str(a)]),(method,a);parity+=1
                preds.append(p)
                if (index+1)%50==0:print(method,index+1,'/201',flush=True)
        torch.cuda.synchronize();raw['method_seconds'][method]=time.perf_counter()-t;raw['predictions'][method]=preds
        torch.save({'grid':grid,'predictions':preds},OUT/f'{method}_grid.pt')
        del model;torch.cuda.empty_cache();print(method,'completed',raw['method_seconds'][method],flush=True)
    raw['overlap_parity']=parity;raw['elapsed_seconds']=time.perf_counter()-start
    torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))

def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h
    assert sha(OUT/'observations.pt')==json.loads((OUT/'observation_commit.json').read_text())['sha256']
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert report(raw)==json.loads((OUT/'summary.json').read_text())
    assert len(raw['observations'])==4 and len(raw['grid'])==201
    assert all(len(v)==201 and all(torch.isfinite(x).all() for x in v) for v in raw['predictions'].values())
    solver,force,template,shape,rx,ry,device=physical_setup();ref=DiscreteReference(solver,rx,ry);support=(template['gamma']<.99).to(device)
    with torch.no_grad():
        for key,obs in raw['observations'].items():
            av=torch.tensor(float(key),device=device,dtype=torch.float32)
            again=ref.traces(F.pad(torch.where(support,av,1.)[None,None],(1,1,1,1),value=1.),force).cpu().double()
            assert torch.equal(again,obs)
    dump(OUT/'verification.json',{'status':'passed','curves_argmins_scores':'exact recomputation from saved predictions',
        'observations':'four float32 labels regenerated bit identical','prior_overlap_predictions':raw['overlap_parity'],
        'frozen_files_dependencies':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),
        'limitation':'Same stencil, no independent physics or new Jacobian validation'})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

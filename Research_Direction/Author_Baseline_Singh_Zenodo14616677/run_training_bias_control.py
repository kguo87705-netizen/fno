"""Training-only constant receiver calibration, frozen model, no optimization."""
import argparse,json,time
from pathlib import Path
import torch
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_contrast_derivative_audit import ROOT,BASE,WIDE,cost,dump,f32
from run_reduced_material_scan import sha
OLD=Path('.build/fno-scalar-grid-refinement-20261009')
OUT=Path('.build/fno-training-bias-control-20261009')
PLAN=OLD/'next_training_bias_protocol.json'

def report(raw):
    old=raw['reference'];bias=raw['calibration']['bias'];rows=[];curves={};derivative_delta=[]
    for key,grid in old['windows'].items():
        target=old['targets'][key];curves[key]={}
        original=old['predictions']['width8'];physical=old['predictions']['physical']
        corrected={str(a):original[str(a)]-bias for a in grid}
        methods={'width8':original,'corrected_width8':corrected,
                 'piecewise_linear':old['predictions']['piecewise_linear'],'physical':physical}
        for method,preds in methods.items():
            values=[float(cost(preds[str(a)],target)) for a in grid];curves[key][method]=values
            i=min(range(len(grid)),key=lambda i:(values[i],grid[i]));a=grid[i];truth=float(key)
            pred=preds[str(a)];fp=physical[str(a)];error=pred-target;pe=fp-target
            origin=preds[key];oe=origin-target
            rows.append({'method':method,'target':truth,'selected_a':a,'boundary_minimum':i in [0,len(grid)-1],
                'absolute_contrast_error':abs(a-truth),'material_half_mse':.5*(a-truth)**2*old['support_fraction'],
                'selected_proxy_cost':values[i],'physical_cost':float(cost(fp,target)),
                'physical_cost_over_initial_a05':float(cost(fp,target)/cost(old['initial_physical'],target)),
                'selected_full_receiver_error':float(error.norm()/target.norm()),
                'selected_scatter_receiver_error':float(error.norm()/(target-old['background']).norm()),
                'physical_full_receiver_error':float(pe.norm()/target.norm()),
                'physical_scatter_receiver_error':float(pe.norm()/(target-old['background']).norm()),
                'true_input_full_receiver_error':float(oe.norm()/target.norm()),
                'true_input_scatter_receiver_error':float(oe.norm()/(target-old['background']).norm())})
        for lo,hi in zip(grid[:-1],grid[1:]):
            j0=(original[str(hi)]-original[str(lo)])/(hi-lo)
            j1=(corrected[str(hi)]-corrected[str(lo)])/(hi-lo)
            derivative_delta.append(float((j1-j0).abs().max()))
    return {'status':'completed','rows':rows,'curves':curves,'bias_norm':float(bias.norm()),'bias_max':float(bias.abs().max()),
        'max_adjacent_derivative_absolute_change':max(derivative_delta),'elapsed_seconds':raw['elapsed_seconds'],
        'new_network_calls':3,'new_physical_calls':0,'scope':'Receiver-only constant correction; same development targets, known geometry, no blind recovery or cost advantage'}

def run():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=json.loads(PLAN.read_text());prior=json.loads((OLD/'protocol.json').read_text())
    frozen={**prior['frozen_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    for n,h in prior['dependencies'].items():assert sha(ROOT/n)==h,n
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'status':'fixed_before_calibration','plan':plan,'plan_sha256':sha(PLAN),
        'script_sha256':sha(Path(__file__)),'dependencies':prior['dependencies'],'frozen_sha256':frozen,
        'execution':'IEEE float32 network; float64 receiver residual and cached objective arithmetic; four CPU threads',
        'data_order':'Save calibration.pt and SHA before torch.load development-containing prior raw.pt',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
    start=time.perf_counter();device='cuda';manifest=json.loads((BASE/'data_manifest.json').read_text());scale=manifest['normalization']['wavefield_rms']
    bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')
    ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();del ck
    times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9;records=[]
    with torch.no_grad():
        for index in [3,5,7]:
            if time.perf_counter()-start>300:raise TimeoutError('5-minute cap')
            d=torch.load(BASE/f'contrast_{index:02d}.pt',weights_only=True,map_location='cpu')
            s={'gamma':d['gamma'],'source_xy_m':d['source_xy_m'][0],'x_m':d['x_m'],'y_m':d['y_m'],'time_seconds':d['time_seconds_snapshots']}
            ids=d['receiver_physical_indices'].to(device)
            field=model(encode_time_features(s,gamma=s['gamma'].to(device),mode='zero')[None])[0,0]
            pred=cubic_time_reconstruction(field[ids[:,0],ids[:,1],:]*scale,s['time_seconds'].to(device),times)[None].double().cpu()+bg['traces_full'][0:1].double()
            truth=d['traces_full'].double()
            records.append({'a':f32(index/10),'prediction':pred,'label':truth,'residual':pred-truth})
    calibration={'records':records,'bias':torch.stack([v['residual'] for v in records]).mean(0),
                 'training_only':True,'coefficient':1.0,'weights':[1/3]*3}
    torch.save(calibration,OUT/'calibration.pt')
    dump(OUT/'calibration_commit.json',{'sha256':sha(OUT/'calibration.pt'),'saved_before_development_load':True,
        'training_files':[f'contrast_{i:02d}.pt' for i in [3,5,7]]})
    del model,field;torch.cuda.empty_cache()
    print('Training-only calibration saved before development load',flush=True)
    old=torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu')
    raw={'reference':old,'calibration':calibration,'elapsed_seconds':time.perf_counter()-start}
    torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))

def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['script_sha256'] and sha(PLAN)==p['plan_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h,n
    for n,h in p['dependencies'].items():assert sha(ROOT/n)==h
    c=torch.load(OUT/'calibration.pt',weights_only=True,map_location='cpu')
    assert sha(OUT/'calibration.pt')==json.loads((OUT/'calibration_commit.json').read_text())['sha256']
    assert torch.equal(c['bias'],torch.stack([r['prediction']-r['label'] for r in c['records']]).mean(0))
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');r=report(raw)
    assert r==json.loads((OUT/'summary.json').read_text())
    prior=json.loads((OLD/'summary.json').read_text());zero_matches=True
    for key,g in raw['reference']['windows'].items():
        for a in g:
            v=raw['reference']['predictions']['width8'][str(a)]
            assert torch.equal(v-torch.zeros_like(c['bias']),v)
        for row in prior['selected']:
            if str(row['target'])!=key:continue
            current=next(x for x in r['rows'] if str(x['target'])==key and x['method']==row['method'])
            assert current['selected_a']==row['selected_a'] and current['physical_cost']==row['physical_cost']
    assert r['max_adjacent_derivative_absolute_change']<1e-10
    dump(OUT/'verification.json',{'status':'passed','calibration':'training-only construction order and hash committed',
        'mean_residual':'exactly recomputed','all_metrics_and_argmins':'exactly recomputed',
        'uncorrected_and_other_baselines':'selected a and physical costs match frozen prior',
        'zero_correction':'bit identical','derivative_invariant_max_absolute_change':r['max_adjacent_derivative_absolute_change'],
        'raw_sha256':sha(OUT/'raw.pt'),'inputs_dependencies':'SHA unchanged'})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

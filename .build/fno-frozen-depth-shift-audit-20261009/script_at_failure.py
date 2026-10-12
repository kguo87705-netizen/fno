"""Frozen three-state depth audit: no training, fitting or inversion."""
import argparse,json,time
from pathlib import Path
import torch
from run_contrast_derivative_audit import ROOT,BASE,WIDE,dump,metrics
from run_fno_local_contrast import physical_setup,generate
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_reduced_material_scan import sha
PLAN=Path('.build/fno-physical-fixed-budget-control-20261009/next_depth_shift_protocol.json')
OUT=Path('.build/fno-frozen-depth-shift-audit-20261009')
BG=Path('.build/operator-pilot-v3-20261008/homogeneous.pt')

def policy():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def shift_mask(mask,k):
    out=torch.zeros_like(mask)
    if k<0:out[:,:k]=mask[:,-k:]
    elif k>0:out[:,k:]=mask[:,:-k]
    else:out.copy_(mask)
    assert out.sum()==mask.sum()==449
    assert not out[0].any() and not out[-1].any() and not out[:,0].any() and not out[:,-1].any()
    return out

def setup():
    solver,force,template,shape,rx,ry,device=physical_setup()
    nominal=torch.load(BASE/'contrast_05.pt',weights_only=True,map_location='cpu')
    bg=torch.load(BG,weights_only=True,map_location='cpu')
    scale=json.loads((BASE/'data_manifest.json').read_text())['normalization']['wavefield_rms']
    ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu')
    model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False)
    return solver,force,template,shape,rx,ry,device,nominal,bg,scale,model

def predict(model,d,device,scale,bg):
    s={'gamma':d['gamma'],'source_xy_m':d['source_xy_m'][0],'x_m':d['x_m'],'y_m':d['y_m'],'time_seconds':d['time_seconds_snapshots']}
    ids=d['receiver_physical_indices'].to(device)
    with torch.no_grad():
        scatter=model(encode_time_features(s,mode='zero')[None])[0,0]*scale
        trace=cubic_time_reconstruction(scatter[ids[:,0],ids[:,1],:],s['time_seconds'].to(device),torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9)[None].double().cpu()+bg['traces_full'][0:1].double()
        field=scatter.cpu().double()+bg['wavefield_snapshots'][0].double()
    return {'field':field,'trace':trace}

def summarize(raw):
    rows=[];bg=raw['background'];nom=raw['nominal'];states=raw['states']
    for key,s in states.items():
        ref=s['physical'];preds={'width8':s['prediction'],'nominal_lookup':nom,'zero_scatter':bg}
        for method,p in preds.items():
            rec={'shift_rows':int(key),'method':method}
            for space in ['field','trace']:
                diff=p[space].double()-ref[space].double()
                rec['full_'+space+'_relative_l2']=float(diff.norm()/ref[space].double().norm())
                rec['scatter_'+space+'_relative_l2']=float(diff.norm()/(ref[space].double()-bg[space].double()).norm())
            rows.append(rec)
    phys=states['2']['physical']['trace'].double()-states['-2']['physical']['trace'].double()
    pred=states['2']['prediction']['trace']-states['-2']['prediction']['trace']
    response=metrics(pred,phys)
    ratio=float(phys.norm()/(nom['trace'].double()-bg['trace'].double()).norm())
    passed=ratio>1e-6 and response['relative_l2']<=.2 and response['cosine']>=.9
    for k in [-2,2]:
        n=next(x for x in rows if x['shift_rows']==k and x['method']=='width8')
        b=next(x for x in rows if x['shift_rows']==k and x['method']=='nominal_lookup')
        passed=passed and n['scatter_field_relative_l2']<=.2 and n['scatter_trace_relative_l2']<=.2 and n['scatter_trace_relative_l2']<b['scatter_trace_relative_l2']
    return {'status':'completed','rows':rows,'finite_geometry_receiver_response':response,'physical_response_over_nominal_scatter_norm':ratio,
        'decision':'inconclusive' if ratio<=1e-6 else ('screen_passed' if passed else 'screen_failed'),
        'elapsed_seconds':raw['elapsed_seconds'],'calls':{'physical':3,'network':3},'checks':raw['checks'],
        'scope':'Three supplied geometries, a=.5, source0; finite discrete depth response, not local Jacobian or blind material recovery'}

def run():
    policy();plan=json.loads(PLAN.read_text())
    for p,h in plan['frozen_evidence_sha256'].items():assert sha(Path(p))==h
    files=[WIDE/'final.pt',BASE/'contrast_05.pt',BASE/'data_manifest.json',BG,ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5',Path('.build/operator-pilot-v3-20261008/synthetic_workpiece_000.pt'),Path('.build/fno-frozen-new-contrast-validation-20261009/width8_grid.pt')]
    deps=['basic_fno.py','run_fno_local_contrast.py','time_feature_input.py','operator_data.py','time_reconstruction.py','FiniteDifferencePyTorchConv.py','Utilities.py','run_contrast_derivative_audit.py','run_reduced_material_scan.py']
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'status':'frozen_before_labels','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
        'files_sha256':{str(p):sha(p) for p in files},'dependencies':{str(ROOT/n):sha(ROOT/n) for n in deps},
        'execution':'IEEE float32 physical and neural forwards; metric accumulation float64; no fitting; four CPU threads',
        'verification_budget':'Separate replay may repeat3 physical and3 network predictions for bit parity, no method changes',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    start=time.perf_counter();solver,force,template,shape,rx,ry,device,nom,bg,scale,model=setup()
    raw={'states':{},'background':{'field':bg['wavefield_snapshots'][0],'trace':bg['traces_full'][0:1]},'nominal':{'field':nom['wavefield_snapshots'][0],'trace':nom['traces_full']},'checks':{}}
    dy=float(nom['y_m'][1]-nom['y_m'][0]);mask=nom['gamma']<.99
    with torch.no_grad():
        for k in [-2,0,2]:
            if time.perf_counter()-start>300:raise TimeoutError('5-minute cap')
            moved=shift_mask(mask,k);t={**template,'gamma':torch.where(moved,.5,1.)}
            d=generate(solver,force,t,shape,rx,ry,device,.5)
            assert torch.equal(d['gamma'],t['gamma'])
            for name in ['source_xy_m','receiver_xy_m','receiver_physical_indices','time_seconds_full','time_seconds_snapshots','x_m','y_m']:
                assert torch.equal(d[name],nom[name]),name
            p=predict(model,d,device,scale,bg)
            assert all(torch.isfinite(v).all() for v in p.values())
            state={'gamma':d['gamma'],'shift_m':k*dy,'physical':{'field':d['wavefield_snapshots'][0],'trace':d['traces_full']},'prediction':p}
            raw['states'][str(k)]=state;torch.save(state,OUT/f'state_{k}.pt')
            if k==0:
                assert torch.equal(state['physical']['field'],raw['nominal']['field']) and torch.equal(state['physical']['trace'],raw['nominal']['trace'])
                old=torch.load('.build/fno-frozen-new-contrast-validation-20261009/width8_grid.pt',weights_only=True,map_location='cpu')
                i=old['grid'].index(.5);assert torch.equal(p['trace'],old['predictions'][i]);del old
            print('shift',k,'completed',time.perf_counter()-start,flush=True)
    raw['checks']={'support_cells':449,'no_wrap_and_interior':True,'coordinates_and_source':'exact equality','nominal_physics':'bit identical training field/trace','nominal_network_trace':'bit identical frozen IEEE grid prediction','dy_m':dy,'shift_m':[-2*dy,0,2*dy]}
    raw['elapsed_seconds']=time.perf_counter()-start;torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summarize(raw))

def verify():
    policy();p=json.loads((OUT/'protocol.json').read_text())
    assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in {**p['files_sha256'],**p['dependencies'],**p['plan']['frozen_evidence_sha256']}.items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert summarize(raw)==json.loads((OUT/'summary.json').read_text())
    solver,force,template,shape,rx,ry,device,nom,bg,scale,model=setup()
    with torch.no_grad():
        for k in [-2,0,2]:
            state=raw['states'][str(k)];assert torch.equal(state['gamma'],torch.where(shift_mask(nom['gamma']<.99,k),.5,1.))
            d=generate(solver,force,{**template,'gamma':state['gamma']},shape,rx,ry,device,.5)
            assert torch.equal(d['wavefield_snapshots'][0],state['physical']['field']) and torch.equal(d['traces_full'],state['physical']['trace'])
            pred=predict(model,d,device,scale,bg)
            assert all(torch.equal(pred[n],state['prediction'][n]) for n in pred)
    dump(OUT/'verification.json',{'status':'passed','metrics':'exact saved recomputation','physical_and_neural_predictions':'All3 field/trace pairs regenerated bit identical','inputs':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'scope':'same discrete physics, not independent solver validation'})
    print('Verification passed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');args=a.parse_args()
    try:verify() if args.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

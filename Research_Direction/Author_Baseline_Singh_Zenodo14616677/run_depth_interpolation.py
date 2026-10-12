"""Training-only depth references versus frozen FNO at two new depths."""
import argparse,json,time
from pathlib import Path
import torch
from run_frozen_depth_shift import OUT as OLD,setup,policy,shift_mask,predict
from run_fno_local_contrast import generate
from run_contrast_derivative_audit import dump,metrics
from run_reduced_material_scan import sha
from time_reconstruction import cubic_time_reconstruction
OUT=Path('.build/fno-depth-interpolation-control-20261009')
PLAN=OLD/'next_depth_interpolation_protocol.json'

def references(old,nom,device):
    result={};bg=old['background'];ids=nom['receiver_physical_indices'].to(device)
    for k in [-1,1]:
        lo,hi=(-2,0) if k==-1 else (0,2)
        pair=[old['states'][str(v)]['physical'] for v in [lo,hi]]
        linear={n:sum(.5*(p[n].double()-bg[n].double()) for p in pair)+bg[n].double() for n in ['field','trace']}
        scatter=(linear['field']-bg['field'].double()).float().to(device)
        trace=cubic_time_reconstruction(scatter[ids[:,0],ids[:,1],:],nom['time_seconds_snapshots'].to(device),torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9)[None].double().cpu()+bg['trace'].double()
        result[str(k)]={'linear_direct':linear,'linear_snapshot':{'field':linear['field'],'trace':trace},
            'nearest':pair[0],'zero_scatter':bg}
    return result

def summarize(raw):
    rows=[];responses={};bg=raw['background'];states=raw['states']
    methods=['width8','linear_direct','linear_snapshot','nearest','zero_scatter']
    for key,s in states.items():
        for m,p in s['predictions'].items():
            rec={'shift_rows':int(key),'method':m}
            for space in ['field','trace']:
                e=p[space].double()-s['physical'][space].double()
                rec['full_'+space+'_relative_l2']=float(e.norm()/s['physical'][space].double().norm())
                rec['scatter_'+space+'_relative_l2']=float(e.norm()/(s['physical'][space].double()-bg[space].double()).norm())
            rows.append(rec)
    f=states['1']['physical']['trace'].double()-states['-1']['physical']['trace'].double()
    signal=float(f.norm()/(raw['nominal']['trace'].double()-bg['trace'].double()).norm())
    passes={}
    for m in methods:
        p=states['1']['predictions'][m]['trace'].double()-states['-1']['predictions'][m]['trace'].double()
        responses[m]=metrics(p,f) if p.norm()>0 else {'relative_l2':1.,'norm_ratio':0.,'cosine':None}
        q=responses[m];rr=[x for x in rows if x['method']==m]
        passes[m]=signal>1e-6 and q['relative_l2']<=.2 and q['cosine'] is not None and q['cosine']>=.9 and all(x['scatter_field_relative_l2']<=.2 and x['scatter_trace_relative_l2']<=.2 for x in rr)
    beats=all(next(x for x in rows if x['shift_rows']==k and x['method']=='linear_direct')['scatter_trace_relative_l2']<next(x for x in rows if x['shift_rows']==k and x['method']=='width8')['scatter_trace_relative_l2'] for k in [-1,1])
    decision='inconclusive' if signal<=1e-6 else ('linear_passes_and_beats_network_both' if passes['linear_direct'] and beats else ('both_fail' if not passes['linear_direct'] and not passes['width8'] else ('network_passes_linear_fails' if passes['width8'] and not passes['linear_direct'] else 'mixed')))
    gaps={k:float((s['predictions']['linear_direct']['trace']-s['predictions']['linear_snapshot']['trace']).norm()/(s['physical']['trace'].double()-bg['trace'].double()).norm()) for k,s in states.items()}
    return {'status':'completed','rows':rows,'finite_geometry_receiver_responses':responses,'physical_response_over_nominal_scatter_norm':signal,
        'screen_passes':passes,'linear_beats_width8_both_receiver':beats,'decision':decision,'direct_vs_snapshot_gap_over_true_scatter_norm':gaps,
        'calls':{'physical':2,'network':2,'training':0},'elapsed_seconds':raw['elapsed_seconds'],'checks':raw['checks'],
        'scope':'Known depth and contrast supplied forward inputs; references have three physical depth labels unavailable during original FNO training; no architecture causal comparison or material recovery'}

def run():
    policy();plan=json.loads(PLAN.read_text());oldp=json.loads((OLD/'protocol.json').read_text())
    frozen={**oldp['files_sha256'],**oldp['dependencies'],**plan['frozen_sha256']}
    frozen[str(Path(__file__).with_name('run_frozen_depth_shift.py'))]=sha(Path(__file__).with_name('run_frozen_depth_shift.py'))
    for p,h in frozen.items():assert sha(Path(p))==h
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'status':'frozen_before_new_labels','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'frozen_sha256':frozen,'verification':'Separate two physical and two neural replays allowed','execution':'IEEE float32 forwards; float64 reference interpolation and metrics; sparse interpolated scatter cast float32 for existing reconstruction'})
    start=time.perf_counter();solver,force,template,shape,rx,ry,device,nom,bg,scale,model=setup()
    old=torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu');refs=references(old,nom,device)
    torch.save(refs,OUT/'references.pt');dump(OUT/'reference_commit.json',{'status':'saved_before_new_labels','sha256':sha(OUT/'references.pt'),'weights':{'-1':{'-2':.5,'0':.5},'1':{'0':.5,'2':.5}},'nearest':{'-1':-2,'1':0}})
    raw={'background':old['background'],'nominal':old['nominal'],'states':{},'checks':{}}
    del old
    with torch.no_grad():
        for k in [-1,1]:
            if time.perf_counter()-start>180:raise TimeoutError('3-minute cap')
            gamma=torch.where(shift_mask(nom['gamma']<.99,k),.5,1.)
            d=generate(solver,force,{**template,'gamma':gamma},shape,rx,ry,device,.5)
            assert torch.equal(d['gamma'],gamma)
            for name in ['source_xy_m','receiver_xy_m','receiver_physical_indices','time_seconds_full','time_seconds_snapshots','x_m','y_m']:assert torch.equal(d[name],nom[name]),name
            p=predict(model,d,device,scale,bg)
            assert all(torch.isfinite(v).all() for v in p.values())
            raw['states'][str(k)]={'gamma':gamma,'shift_m':k*float(nom['y_m'][1]-nom['y_m'][0]),'physical':{'field':d['wavefield_snapshots'][0],'trace':d['traces_full']},'predictions':{'width8':p,**refs[str(k)]}}
            torch.save(raw['states'][str(k)],OUT/f'state_{k}.pt');print('shift',k,'complete',flush=True)
    raw['checks']={'449_cells_no_wrap_interior':True,'coordinates':'exact unchanged','references':'committed before new labels','nearest_tie_rule':'lower signed depth'}
    raw['elapsed_seconds']=time.perf_counter()-start;torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summarize(raw))

def verify():
    policy();p=json.loads((OUT/'protocol.json').read_text());assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in p['frozen_sha256'].items():assert sha(Path(f))==h
    assert sha(OUT/'references.pt')==json.loads((OUT/'reference_commit.json').read_text())['sha256']
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');report=summarize(raw);assert report==json.loads((OUT/'summary.json').read_text())
    solver,force,template,shape,rx,ry,device,nom,bg,scale,model=setup()
    refs=references(torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu'),nom,device)
    with torch.no_grad():
        for k in [-1,1]:
            s=raw['states'][str(k)];assert torch.equal(s['gamma'],torch.where(shift_mask(nom['gamma']<.99,k),.5,1.))
            d=generate(solver,force,{**template,'gamma':s['gamma']},shape,rx,ry,device,.5)
            assert torch.equal(d['wavefield_snapshots'][0],s['physical']['field']) and torch.equal(d['traces_full'],s['physical']['trace'])
            again={'width8':predict(model,d,device,scale,bg),**refs[str(k)]}
            for m,values in again.items():
                for space,v in values.items():assert torch.equal(v,s['predictions'][m][space]),(k,m,space)
    assert all(x['scatter_field_relative_l2']==x['scatter_trace_relative_l2']==1. for x in report['rows'] if x['method']=='zero_scatter')
    dump(OUT/'verification.json',{'status':'passed','metrics':'Exact saved recomputation','new_physical_and_network':'Both field/trace predictions regenerated bit identical','all_references':'Exact reconstruction from training labels only','files':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'limits':'Same discrete solver; development data, no final test'})
    print('Verification passed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');args=a.parse_args()
    try:verify() if args.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise

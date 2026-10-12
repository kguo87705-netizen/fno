"""Preregistered one-geometry FNO contrast interpolation, fixed1200 updates."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import FiniteDifferencePyTorchConv as FD
import Utilities
from basic_fno import BasicFNO
from fixed_source_inputs import FixedSourceInputs
from time_feature_input import encode_time_features
from train_multiworkpiece_control import state_hash
from run_receiver_loss_control import supervised_losses
from run_four_materials import evaluate
from run_reduced_material_scan import sha, check_sources

ROOT=Path(__file__).resolve().parent
OLD=Path('.build/operator-pilot-v3-20261008')
OUT=Path('.build/fno-local-contrast-20261009')
PLAN=Path('.build/author-scalar-material-control-20261008/next_fno_local_contrast_protocol.json')


def physical_setup():
    cfg=pd.read_csv(ROOT/'settings2D.csv').iloc[0]
    nx,ny,n,ns=(int(cfg[k]) for k in ['Nx','Ny','N','numberOfSources'])
    dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solver=FD.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device)
    sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),ns)
    rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx)
    force=torch.tensor(pd.read_hdf(ROOT/'data_casestudy/source.h5').values).reshape(ns,1,nx+3,ny+3,n)[0:1].clone().to(device)
    template=torch.load(OLD/'synthetic_workpiece_000.pt',weights_only=True,map_location='cpu')
    return solver,force,template,(nx,ny,n),rx,ry,device


def generate(solver,force,template,shape,rx,ry,device,a):
    nx,ny,n=shape
    material=torch.where(template['gamma']<.99,torch.full_like(template['gamma'],a),torch.ones_like(template['gamma']))
    gamma=torch.ones(1,1,nx+3,ny+3,device=device)
    gamma[0,0,1:-1,1:-1]=material.to(device)
    zeros=torch.zeros_like(gamma)
    with torch.no_grad():
        wave=solver.forwardNSteps(zeros,zeros.clone(),gamma,force,nx,ny,n,1,device)
        traces=wave[:,rx,ry,1:].cpu().clone()
        snapshots=wave[:,1:-1,1:-1,:].index_select(-1,(template['snapshot_trace_indices']+1).to(device)).cpu()
    assert torch.isfinite(traces).all() and torch.isfinite(snapshots).all()
    ids=template['receiver_physical_indices']
    assert torch.equal(snapshots[:,ids[:,0],ids[:,1],:],traces.index_select(-1,template['snapshot_trace_indices']))
    data={k:template[k] for k in ['snapshot_trace_indices','time_seconds_full','time_seconds_snapshots','x_m','y_m','receiver_physical_indices','receiver_xy_m']}
    data.update(gamma=material,traces_full=traces,wavefield_snapshots=snapshots,source_xy_m=template['source_xy_m'][0:1])
    return data


def samples():
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    scale=manifest['normalization']['wavefield_rms']
    result={'train':[],'development_interpolation':[]}
    for entry in manifest['records']:
        d=torch.load(OUT/entry['file'],weights_only=True,map_location='cpu')
        s={'gamma':d['gamma'],'target':d['wavefield_snapshots']/scale,'source_xy_m':d['source_xy_m'][0],
           'x_m':d['x_m'],'y_m':d['y_m'],'time_seconds':d['time_seconds_snapshots'],
           'receiver_indices':d['receiver_physical_indices'],'traces_full':d['traces_full'][0],
           'snapshot_trace_indices':d['snapshot_trace_indices'],'group_id':entry['group_id'],
           'split':entry['split'],'shot':0,'wavefield_scale':scale,'contrast':entry['contrast']}
        result[entry['split']].append(s)
    return manifest,result


def pool(items):
    energy={k:sum(x['energy_sums'][k] for x in items) for k in items[0]['energy_sums']}
    regions={}
    for name in items[0]['regions']:
        e={k:sum(x['regions'][name]['energy_sums'][k] for x in items) for k in ['error','reference']}
        regions[name]={'scatter_field_relative_l2':float(np.sqrt(e['error']/e['reference'])),'energy_sums':e}
    return {'split':items[0]['split'],'groups':len(items),'shots':len(items),
            **{k:float(np.mean([x[k] for x in items])) for k in ['mean_field_mse','mean_receiver_mse','mean_total_objective']},
            'full_field_relative_l2':float(np.sqrt(energy['field_error']/energy['field_true'])),
            'full_trace_relative_l2':float(np.sqrt(energy['trace_error']/energy['trace_true'])),
            'scatter_field_relative_l2':float(np.sqrt(energy['field_error']/energy['scatter_field_true'])),
            'scatter_trace_relative_l2':float(np.sqrt(energy['trace_error']/energy['scatter_trace_true'])),
            'regions':regions,'energy_sums':energy,'records':[x['records'][0] for x in items]}


def refs(train,held,bg,scale,sr,device):
    targets=[s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/scale for s in train]
    mean=torch.stack(targets).mean(0)
    result={}
    coefficients=[]
    for name in ['zero','training_mean','contrast_nearest','piecewise_linear']:
        result[name]={}
        for split,group in [('train',train),('development_interpolation',held)]:
            metrics=[]
            for s in group:
                a=s['contrast']
                if name=='zero':prediction=torch.zeros_like(mean)
                elif name=='training_mean':prediction=mean
                elif name=='contrast_nearest':
                    j=min(range(3),key=lambda i:(round(abs(train[i]['contrast']-a),12),train[i]['contrast']))
                    prediction=targets[j]
                else:
                    if a<=.5:lo,hi=0,1
                    else:lo,hi=1,2
                    w=(a-train[lo]['contrast'])/(train[hi]['contrast']-train[lo]['contrast'])
                    prediction=(1-w)*targets[lo]+w*targets[hi]
                    coefficients.append({'split':split,'contrast':a,'low':train[lo]['contrast'],'high':train[hi]['contrast'],'high_weight':w})
                m,_=evaluate(None,[s],bg,scale,sr,device,fixed=prediction)
                metrics.append(m)
            result[name][split]=pool(metrics)
    return result,coefficients,targets


def run():
    torch.set_num_threads(4)
    check_sources()
    assert not OUT.exists(),'Use a new directory; never overwrite prior results.'
    plan=json.loads(PLAN.read_text());assert plan['status']=='planned_not_run'
    assert plan['training_contrasts']==[.3,.5,.7] and plan['development_interpolation_contrasts']==[.4,.6]
    OUT.mkdir()
    protocol={'status':'fixed_before_generation_and_training','plan':plan,'plan_sha256':sha(PLAN),
              'script_sha256':sha(Path(__file__)),'data_manifest_sha256':sha(OLD/'manifest.json'),
              'template_sha256':sha(OLD/'synthetic_workpiece_000.pt'),'homogeneous_sha256':sha(OLD/'homogeneous.pt'),
              'source_sha256':sha(ROOT/'data_casestudy/source.h5'),
              'dependency_hashes':{name:sha(ROOT/name) for name in ['basic_fno.py','fixed_source_inputs.py','time_feature_input.py','operator_data.py','run_four_materials.py','run_receiver_loss_control.py','time_reconstruction.py','FiniteDifferencePyTorchConv.py','Utilities.py']},
              'updates':1200,'evaluation_updates':[0,400,800,1200],'sample_order':[i%3 for i in range(1200)],
              'scope':'Fixed training geometry contrast interpolation only; no unseen geometry, old validation/test or independent final test'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    solver,force,template,shape,rx,ry,device=physical_setup()
    records=[]
    for split,contrasts in [('train',[.3,.5,.7]),('development_interpolation',[.4,.6])]:
        for a in contrasts:
            name=f'contrast_{round(a*10):02d}';data=generate(solver,force,template,shape,rx,ry,device,a)
            torch.save(data,OUT/(name+'.pt'))
            records.append({'group_id':name,'split':split,'contrast':a,'file':name+'.pt','file_sha256':sha(OUT/(name+'.pt'))})
            print('generated '+name+' '+split,flush=True)
            del data
    with torch.no_grad():
        ones=torch.ones_like(template['gamma']);temp={**template,'gamma':ones}
        homogeneous=generate(solver,force,temp,shape,rx,ry,device,1.)
    bg=torch.load(OLD/'homogeneous.pt',weights_only=True,map_location='cpu')
    assert torch.equal(homogeneous['wavefield_snapshots'],bg['wavefield_snapshots'][0:1])
    assert torch.equal(homogeneous['traces_full'],bg['traces_full'][0:1])
    normalization=json.loads((OLD/'manifest.json').read_text())['normalization']
    manifest={'status':'generated','records':records,'normalization':normalization,'geometry_support_cells':int((template['gamma']<.99).sum()),'homogeneous_parity':'bit_identical_source0','scope':protocol['scope']}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    del solver,force,template,homogeneous
    manifest,groups=samples();train,held=groups['train'],groups['development_interpolation']
    scale,sr=normalization['wavefield_rms'],normalization['trace_rms']
    references,coefficients,targets=refs(train,held,bg,scale,sr,device)
    (OUT/'references.json').write_text(json.dumps({'metrics':references,'linear_coefficients':coefficients},indent=2),encoding='utf-8')
    inputs=FixedSourceInputs(train,device)
    for i in range(3):assert torch.equal(inputs[i],encode_time_features(train[i],mode='zero')[None].to(device))
    torch.manual_seed(2);model=BasicFNO(input_channels=8).to(device)
    initial_hash=state_hash(model)
    old=json.loads(Path('.build/basic-fno-twelve-materials-20261008/comparison.json').read_text())
    assert initial_hash==old['protocol']['initial_state_hash']
    (OUT/'initial_model.json').write_text(json.dumps({'hash':initial_hash,'config':model.config,'old_initial_parity':True},indent=2),encoding='utf-8')
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];start=time.perf_counter();exposures=[0,0,0]
    for update in range(1201):
        if update in protocol['evaluation_updates']:
            model.eval()
            training,_=evaluate(model,train,bg,scale,sr,device,inputs=inputs)
            development,example=evaluate(model,held,bg,scale,sr,device,save_example=update==1200)
            history.append({'update':update,'training':training,'development_interpolation':development})
            (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps({'update':update,'train_scatter_field':training['scatter_field_relative_l2'],'train_scatter_trace':training['scatter_trace_relative_l2'],'interpolation_scatter_field':development['scatter_field_relative_l2'],'interpolation_scatter_trace':development['scatter_trace_relative_l2']}),flush=True)
        if update==1200:break
        i=update%3;model.train();opt.zero_grad(set_to_none=True)
        pred=model(inputs[i]);lf,lr=supervised_losses(pred,targets[i][None,None],train[i]['receiver_indices'].to(device),scale,sr)
        loss=lf+lr;assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();exposures[i]+=1
        del pred,lf,lr,loss
        if (update+1)%100==0:print(f'Update {update+1}/1200, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    assert exposures==[400,400,400]
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':model.config,
                'updates':1200,'scale':scale,'exposures':exposures},OUT/'final.pt')
    torch.save(example,OUT/'interpolation_example.pt')
    report={'status':'completed','history':history,'final_training':training,'final_interpolation':development,
            'references':references,'exposures':exposures,'elapsed_seconds':time.perf_counter()-start,'scope':protocol['scope']}
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def verify():
    torch.set_num_threads(4);check_sources()
    p=json.loads((OUT/'protocol.json').read_text())
    for path,key in [(PLAN,'plan_sha256'),(Path(__file__),'script_sha256'),(OLD/'manifest.json','data_manifest_sha256'),(OLD/'synthetic_workpiece_000.pt','template_sha256'),(OLD/'homogeneous.pt','homogeneous_sha256'),(ROOT/'data_casestudy/source.h5','source_sha256')]:assert sha(path)==p[key]
    for name,h in p['dependency_hashes'].items():assert sha(ROOT/name)==h
    m,groups=samples();solver,force,template,shape,rx,ry,device=physical_setup()
    for entry in m['records']:
        assert sha(OUT/entry['file'])==entry['file_sha256']
        saved=torch.load(OUT/entry['file'],weights_only=True,map_location='cpu')
        d=generate(solver,force,template,shape,rx,ry,device,entry['contrast'])
        for k in d:assert torch.equal(d[k],saved[k]),(entry['contrast'],k)
        print('physical data verified',entry['contrast'],flush=True)
    del solver,force,template,d,saved
    train,held=groups['train'],groups['development_interpolation'];scale=m['normalization']['wavefield_rms'];sr=m['normalization']['trace_rms']
    bg=torch.load(OLD/'homogeneous.pt',weights_only=True,map_location='cpu');r=json.loads((OUT/'comparison.json').read_text())
    refs_again,coefficients,_=refs(train,held,bg,scale,sr,device)
    assert refs_again==r['references']
    assert coefficients==json.loads((OUT/'references.json').read_text())['linear_coefficients']
    torch.manual_seed(2);model=BasicFNO(input_channels=8).to(device)
    assert state_hash(model)==json.loads((OUT/'initial_model.json').read_text())['hash']
    checkpoint=torch.load(OUT/'final.pt',weights_only=True,map_location='cpu')
    assert checkpoint['updates']==1200 and checkpoint['exposures']==r['exposures']==[400,400,400]
    for st in checkpoint['optimizer_state_dict']['state'].values():
        assert int(st['step'])==1200 and all(torch.isfinite(v).all() for v in st.values() if isinstance(v,torch.Tensor))
    model.load_state_dict(checkpoint['state_dict']);model.eval();inputs=FixedSourceInputs(train,device)
    training,_=evaluate(model,train,bg,scale,sr,device,inputs=inputs)
    development,example=evaluate(model,held,bg,scale,sr,device,save_example=True)
    assert training==r['final_training'] and development==r['final_interpolation']
    saved=torch.load(OUT/'interpolation_example.pt',weights_only=True,map_location='cpu')
    for k in example:
        if isinstance(example[k],torch.Tensor):assert torch.equal(example[k],saved[k])
        else:assert example[k]==saved[k]
    report={'status':'passed','five_physical_targets':'bit_identical_recomputed','references':'exact_recomputed',
            'final_metrics':'exact_reload','example_predictions':'bit_identical','optimizer_steps':1200,
            'exposures':[400,400,400],'source_and_dependency_hashes':'passed','scope':p['scope']}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    verify() if args.verify else run()

"""Immutable-model field/receiver and geometry/material parameter-gradient audit."""
import argparse,json,time
from pathlib import Path
import torch
from run_depth_coverage_storage_retry import OUT as PREV
from run_paired_depth_coverage import policy,data_setup,ROOT,WIDE,BasicFNO,state_hash,encode_time_features,supervised_losses,dump,sha
OUT=Path('.build/fno-frozen-loss-gradient-audit-20261009')
PLAN=PREV/'next_experiment_protocol.json'
KEYS=['depth-2','depth0','depth2','contrast03','contrast07']
REGIONS={'opposite_boundary':slice(0,2),'interior':slice(2,126),'array_side':slice(126,128)}

def real_vector(grads):
    return torch.cat([(torch.view_as_real(g.detach()).reshape(-1) if g.is_complex() else g.detach().reshape(-1)).cpu().double() for g in grads])

def stats(gram,u,v):
    u,v=torch.tensor(u,dtype=torch.float64),torch.tensor(v,dtype=torch.float64)
    a=float(u@gram@u);b=float(v@gram@v);dot=float(u@gram@v)
    assert a>=-1e-12 and b>=-1e-12
    an,bn=max(a,0.)**.5,max(b,0.)**.5
    return {'first_norm':an,'second_norm':bn,'dot':dot,'cosine':dot/(an*bn) if min(an,bn)>1e-12 else None,'second_over_first':bn/an if an>1e-12 else None,
            'negative_first_direction_derivative_of_second':-dot}

def summarize(raw):
    models={}
    for name,data in raw['models'].items():
        gram=data['gram'];rows=[]
        for i,key in enumerate(KEYS):
            u=[0.]*10;v=u.copy();u[2*i]=1;v[2*i+1]=1;total=[a+b for a,b in zip(u,v)]
            rows.append({'state':key,**data['rows'][i],'field_receiver':stats(gram,u,v),'total_field':stats(gram,total,u)})
        geo=[1/3]*6+[0.]*4;mat=[0.]*6+[.5]*4
        fieldgeo=[1/3 if i in [0,2,4] else 0. for i in range(10)]
        fieldindividualnorms=[float(gram[i,i])**.5 for i in [0,2,4]]
        fieldmean=float(torch.tensor(fieldgeo,dtype=torch.float64)@gram@torch.tensor(fieldgeo,dtype=torch.float64))**.5
        models[name]={'rows':rows,'geometry_material':stats(gram,geo,mat),'geometry_field_mean_norm':fieldmean,
            'geometry_field_mean_over_mean_individual_norm':fieldmean/(sum(fieldindividualnorms)/3),'hash_before':data['hash_before'],'hash_after':data['hash_after']}
    return {'status':'completed','models':models,'elapsed_seconds':raw['elapsed_seconds'],'calls':{'forward':10,'backward':20,'optimizer':0,'physical':0},
        'scope':'Real Euclidean network parameter gradients on five stored training/anchor states; not material Jacobian or actual Adam step.'}

def run():
    policy();plan=json.loads(PLAN.read_text(encoding='utf-8'))
    frozen=plan['frozen_sha256']
    for p,h in frozen.items():assert sha(Path(p))==h
    assert not OUT.exists();OUT.mkdir();deps=['run_paired_depth_coverage.py','run_frozen_depth_shift.py','train_multiworkpiece_control.py','operator_data.py']
    dump(OUT/'protocol.json',{'status':'frozen_before_measurement','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'frozen_sha256':frozen,
        'dependencies':{str(ROOT/n):sha(ROOT/n) for n in deps},'accumulation':'CPU float64 real-coordinate Gram matrix; complete complex real/imag components','regions_tolerance':'partition within1e-10 relative and1e-12 absolute float64 energy',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    start=time.perf_counter();device,nom,bg,scale,sr,samples=data_setup();ids=nom['receiver_physical_indices'].to(device)
    raw={'models':{}}
    for name,path in [('original',WIDE/'final.pt'),('coverage',PREV/'depth_coverage_update300.pt')]:
        ck=torch.load(path,weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();del ck
        before=state_hash(model);vectors=[];rows=[];params=tuple(model.parameters())
        for key in KEYS:
            if time.perf_counter()-start>120:raise TimeoutError('2-minute audit cap')
            s={'gamma':samples[key]['gamma'],'source_xy_m':nom['source_xy_m'][0],'x_m':nom['x_m'],'y_m':nom['y_m'],'time_seconds':nom['time_seconds_snapshots']}
            x=encode_time_features(s,mode='zero')[None].to(device)
            target=(samples[key]['physical']['field'].to(device)/scale-bg['wavefield_snapshots'][0].to(device)/scale)[None,None]
            pred=model(x);lf,lr=supervised_losses(pred,target,ids,scale,sr)
            assert torch.isfinite(lf) and torch.isfinite(lr)
            gf=real_vector(torch.autograd.grad(lf,params,retain_graph=True));gr=real_vector(torch.autograd.grad(lr,params));assert torch.isfinite(gf).all() and torch.isfinite(gr).all()
            vectors.extend([gf,gr]);e=(pred.detach()[0,0].double()-target[0,0].double()).cpu();t=target[0,0].detach().double().cpu()
            total=float(e.square().sum());regions={}
            for region,ys in REGIONS.items():
                error=float(e[:,ys,:].square().sum());ref=float(t[:,ys,:].square().sum());regions[region]={'error_energy':error,'scatter_reference_energy':ref,'relative_l2':(error/ref)**.5,'fraction_total_error':error/total}
            assert abs(sum(q['error_energy'] for q in regions.values())-total)<=1e-10*total+1e-12
            rows.append({'field_loss':float(lf.detach()),'receiver_loss':float(lr.detach()),'regions':regions,'total_error_energy':total,'real_parameter_coordinates':gf.numel()})
            del x,target,pred,lf,lr,e,t,gf,gr
            print(name,key,'measured',flush=True)
        gram=torch.zeros(10,10,dtype=torch.float64)
        for i in range(10):
            for j in range(i,10):gram[i,j]=gram[j,i]=torch.dot(vectors[i],vectors[j])
        assert state_hash(model)==before
        raw['models'][name]={'gram':gram,'rows':rows,'hash_before':before,'hash_after':state_hash(model)}
        torch.save(raw['models'][name],OUT/f'{name}_sums.pt');del vectors,model,params;torch.cuda.empty_cache()
    assert time.perf_counter()-start<=120
    raw['elapsed_seconds']=time.perf_counter()-start;torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summarize(raw))

def verify():
    policy();p=json.loads((OUT/'protocol.json').read_text());assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in {**p['frozen_sha256'],**p['dependencies']}.items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert summarize(raw)==json.loads((OUT/'summary.json').read_text())
    for data in raw['models'].values():
        g=data['gram'];assert torch.isfinite(g).all() and torch.equal(g,g.T)
        assert torch.linalg.eigvalsh(g).min()>=-1e-10*max(float(g.diag().max()),1.)
        assert data['hash_before']==data['hash_after']
        for row in data['rows']:assert abs(sum(q['error_energy'] for q in row['regions'].values())-row['total_error_energy'])<=1e-10*row['total_error_energy']+1e-12
    dump(OUT/'verification.json',{'status':'passed','gradients':'Finite complete real/imag coordinates;Gram symmetric positive semidefinite within tolerance','summaries':'Exact recomputation from saved sufficient sums','parameters':'Before/after hash unchanged','regions':'Energy partition checked','dependencies':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'limits':'No independent gradient FD test or physical material derivative validation'})
    print('Verification passed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');args=a.parse_args()
    try:verify() if args.verify else run()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'class':type(e).__name__,'error':str(e)})
        raise

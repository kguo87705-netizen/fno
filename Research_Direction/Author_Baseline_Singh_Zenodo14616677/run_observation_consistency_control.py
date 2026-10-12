"""Fixed 35-step case1 observation/initialization control; no original-file edits."""
import hashlib,json,time,urllib.request
from pathlib import Path
import torch,pandas as pd,numpy as np
import FiniteDifferencePyTorchConv as FD
import Utilities
from discrete_reference import DiscreteReference,time_cost,variable_mask

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root=Path(__file__).resolve().parent;out=Path('.build/author-observation-control-20261008');out.mkdir(exist_ok=True);assert not (out/'protocol.json').exists();torch.set_num_threads(4);torch.manual_seed(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sources=out/'downloaded_generation_sources';sources.mkdir(exist_ok=True)
    expected={'GenerateMeasurementDataTest.py':'681d26903cb78a52e591a8163342af4f','GenerateMeasurementDataTrain.py':'9ca35d656f0c9dd95e121e347bef04ce'};source_manifest={}
    for name,md5 in expected.items():
        url='https://zenodo.org/records/14616677/files/'+name+'?download=1';data=urllib.request.urlopen(url,timeout=30).read();assert hashlib.md5(data).hexdigest()==md5;(sources/name).write_bytes(data);source_manifest[name]={'url':url,'md5':md5,'sha256':sha(sources/name)}
    (sources/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2),encoding='utf-8')
    cfg=pd.read_csv(root/'settings2D.csv').iloc[0];nx,ny,n,ns=(int(cfg[k]) for k in ['Nx','Ny','N','numberOfSources']);dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt']);solver=FD.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device);sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),ns);rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx);ref=DiscreteReference(solver,rx,ry)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values).reshape(ns,1,nx+3,ny+3,n).to(device);released=torch.tensor(pd.read_hdf(root/'data_casestudy/measurement1.h5').values).reshape(ns,len(rx),n).to(device);truth=torch.ones(1,1,nx+3,ny+3,device=device);truth[0,0,1:-1,1:-1]=torch.tensor(pd.read_hdf(root/'data_casestudy/material1.h5').values,device=device);mask=variable_mask(truth,rx,ry,sx,sy)
    protocol={'status':'fixed_before_training','case':1,'arms':[[o,i] for o in ['released','self_consistent'] for i in ['uniform','near_truth']],'updates_per_arm':35,'evaluation':'every saved state; fixed final35 main result, no truth-based stopping','initialization':'uniform gamma1 or .95*truth+.05; near_truth diagnostic only','optimizer':'Adam lr.02, scaled cost1e12, gradient clip1e-5, mask and bounds same as existing reference','observation':'released measurement1 or frozen ref.traces(truth) from identical float32 stencil','scope':'Case1 development observation-consistency/initialization comparison. No independent-physics, final-test or quantitative reconstruction success guarantee.','script_sha256':sha(Path(__file__)),'input_hashes':{p:sha(root/p) for p in ['settings2D.csv','data_casestudy/source.h5','data_casestudy/measurement1.h5','data_casestudy/material1.h5']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with torch.no_grad():self_obs=ref.traces(truth,force,False)
    torch.save({'truth':truth.cpu(),'released':released.cpu(),'self_consistent':self_obs.cpu()},out/'frozen_observations.pt');results={}
    for obs_name,target in [('released',released),('self_consistent',self_obs)]:
        for init in ['uniform','near_truth']:
            key=obs_name+'_'+init;initial=torch.ones_like(truth) if init=='uniform' else .95*truth+.05;gamma=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([gamma],lr=.02);history=[];records=[];start=time.perf_counter()
            for update in range(36):
                opt.zero_grad(set_to_none=True)
                if update==35:
                    with torch.no_grad():pred=ref.traces(gamma,force,False);loss=time_cost(pred,target,dx,dy,dt)
                else:pred=ref.traces(gamma,force);loss=time_cost(pred,target,dx,dy,dt)
                assert torch.isfinite(loss);records.append({'update':update,'cost':float(loss.detach()),'waveform_relative_l2':float((pred.detach()-target).double().norm()/target.double().norm()),'material_half_mse':float(.5*(gamma.detach()-truth).square().mean()),'physical_material_half_mse':float(.5*(gamma.detach()[...,1:-1,1:-1]-truth[...,1:-1,1:-1]).square().mean())});history.append(gamma.detach().cpu().clone())
                if update<35:
                    (loss*1e12).backward();gamma.grad*=mask;assert torch.isfinite(gamma.grad).all();torch.nn.utils.clip_grad_norm_([gamma],1e-5);opt.step()
                    with torch.no_grad():gamma.clamp_(float(cfg['gamma0']),1)
                if update%5==0:print(key+' '+json.dumps(records[-1]),flush=True)
            arm=out/key;arm.mkdir();torch.save({'history':torch.cat(history),'final_optimizer':opt.state_dict(),'prediction_final':pred.detach().cpu()},arm/'states.pt');result={'records':records,'final_cost_over_initial':records[-1]['cost']/records[0]['cost'],'final_material_mse_over_initial':records[-1]['material_half_mse']/records[0]['material_half_mse'],'elapsed_seconds':time.perf_counter()-start};(arm/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');results[key]=result
    report={'status':'completed','protocol':protocol,'generation_sources':source_manifest,'results':results};(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('all four arms complete',flush=True)
if __name__=='__main__':main()

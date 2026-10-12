"""Case1 observation consistency and unconstrained local response Gram audit."""
import hashlib,json,time
from pathlib import Path
import torch,pandas as pd
import FiniteDifferencePyTorchConv as FD
import Utilities
from discrete_reference import DiscreteReference,time_cost,variable_mask

def main():
    torch.set_num_threads(4); torch.set_default_dtype(torch.float64); root=Path(__file__).resolve().parent; out=Path('.build/author-fwi-response-gram-v2-20261008'); out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists()
    cfg=pd.read_csv(root/'settings2D.csv').iloc[0]; nx,ny,n,ns=(int(cfg[k]) for k in ['Nx','Ny','N','numberOfSources']); dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt']); device='cuda' if torch.cuda.is_available() else 'cpu'
    solver=FD.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),torch.device(device)).double(); sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),ns); rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx); ref=DiscreteReference(solver,rx,ry)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values,dtype=torch.float64).reshape(ns,1,nx+3,ny+3,n).to(device); obs=torch.tensor(pd.read_hdf(root/'data_casestudy/measurement1.h5').values,dtype=torch.float64).reshape(ns,len(rx),n).to(device)
    truth=torch.ones(1,1,nx+3,ny+3,device=device); truth[0,0,1:-1,1:-1]=torch.tensor(pd.read_hdf(root/'data_casestudy/material1.h5').values,device=device); mask=variable_mask(truth,rx,ry,sx,sy)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protocol={'scope':'Case1 development only; observation consistency, local unconstrained positive-density derivative responses, separate feasible projected trials. Not a full identifiability proof.','directions':['shallow','deep','left','right','defect_contrast','uniform'],'direction_definition':'gamma times masked peak-normalized bump; parameter is relative gamma perturbation, defect_contrast uses mask*(gamma<.5)','derivative_peak_amplitudes':[.0001,.00005],'projected_trial_peak_amplitude':.005,'boundary':'ghost and source/receiver material masked','precision':'float64 FD using promoted original float32 coefficients','no_validation_test':True,'script_sha256':sha(Path(__file__))}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with torch.no_grad():
        predicted=ref.traces(truth,force,False); bg=ref.traces(torch.ones_like(truth),force,False)
    err=predicted-obs; consistency={'relative_l2':float(err.double().norm()/obs.double().norm()),'truth_cost':float(time_cost(predicted,obs,dx,dy,dt)),'uniform_cost':float(time_cost(bg,obs,dx,dy,dt)),'trace_cosine':float(torch.nn.functional.cosine_similarity(predicted.double().flatten(),obs.double().flatten(),dim=0)),'amplitude_ls_pred_to_obs':float((predicted.double()*obs.double()).sum()/predicted.double().square().sum())}
    print('consistency '+json.dumps(consistency),flush=True)
    shifts=[]
    for lag in range(-8,9):
        a=predicted if lag==0 else predicted[...,lag:] if lag>0 else predicted[...,:lag]
        b=obs if lag==0 else obs[...,:-lag] if lag>0 else obs[...,-lag:]
        scale=float((a*b).sum()/a.square().sum()); shifts.append({'lag_samples':lag,'scale':scale,'raw_relative_l2':float((a-b).norm()/b.norm()),'scaled_relative_l2':float((a*scale-b).norm()/b.norm())})
    consistency['time_shift_amplitude_diagnostic']=shifts
    x,y=torch.meshgrid(torch.linspace(-dx,float(cfg['Lx'])+dx,nx+3,device=device),torch.linspace(-dy,float(cfg['Ly'])+dy,ny+3,device=device),indexing='ij')
    # y=50mm is array side; all bumps have identical 8mm physical width.
    dirs={name:torch.exp(-((x-cx)**2+(y-cy)**2)/(2*.008**2))[None,None]*mask for name,cx,cy in [('shallow',.05,.04),('deep',.05,.01),('left',.025,.025),('right',.075,.025)]}
    dirs['defect_contrast']=(truth<.5).to(truth)*mask; dirs['uniform']=mask.clone()
    dirs={k:truth*(v/v.abs().max()) for k,v in dirs.items()}; names=list(dirs); responses=[]; records={}; eps=.0001
    with torch.no_grad():
        for name,d in dirs.items():
            checks=[]; curves=[]
            for e in [eps,eps/2]:
                plus=ref.traces(truth+e*d,force,False); minus=ref.traces(truth-e*d,force,False); jv=(plus-minus).double()/(2*e);curves.append(jv)
                assert (truth-e*d).min()>0 and torch.isfinite(jv).all()
                checks.append({'peak_epsilon':e,'response_norm':float(jv.norm()),'response_relative_to_observation':float(jv.norm()/obs.double().norm()),'plus_min_gamma':float((truth+e*d).min()),'plus_max_gamma':float((truth+e*d).max())})
            convergence=float((curves[0]-curves[1]).norm()/curves[1].norm())
            # Differentiation may exceed gamma<=1; feasible trial is separately clipped.
            candidate=(truth+.005*d).clamp(float(cfg['gamma0']),1); actual=candidate-truth; trial=ref.traces(candidate,force,False)
            records[name]={'derivative_checks':checks,'two_epsilon_relative_difference':convergence,'projected_trial':{'requested_peak_step':.005,'actual_peak_step':float(actual.abs().max()),'actual_norm':float(actual.norm()),'cost_over_truth':float(time_cost(trial,obs,dx,dy,dt)/time_cost(predicted,obs,dx,dy,dt)),'trace_change_relative_to_observation':float((trial-predicted).double().norm()/obs.double().norm())}}
            responses.append(curves[0].flatten());print(name+' '+json.dumps(records[name]),flush=True)
    matrix=torch.stack(responses,dim=1); gram=matrix.T@matrix; norms=gram.diag().sqrt(); corr=gram/(norms[:,None]*norms[None,:]); singular=torch.linalg.svdvals(matrix); scaled_singular=torch.linalg.svdvals(matrix/norms[None,:])
    assert torch.isfinite(matrix).all() and torch.allclose(gram,gram.T) and torch.allclose(corr.diag(),torch.ones_like(norms))
    report={'status':'completed','protocol':protocol,'observation_consistency':consistency,'directions':records,'direction_names':names,'response_gram':gram.tolist(),'normalized_response_correlation':corr.tolist(),'singular_values':singular.tolist(),'unit_response_singular_values':scaled_singular.tolist(),'limitations':'Finite differences use gamma-relative directions and may exceed optimization upper bound, but gamma stays positive. Six directions do not prove full Jacobian rank. A small cost first derivative does not measure response norm.','input_hashes':{p:sha(root/p) for p in ['settings2D.csv','data_casestudy/source.h5','data_casestudy/measurement1.h5','data_casestudy/material1.h5']}}
    torch.save({'truth_prediction':predicted.cpu(),'observation':obs.cpu(),'responses':matrix.cpu(),'gram':gram.cpu(),'correlation':corr.cpu(),'directions':{k:v.cpu() for k,v in dirs.items()}},out/'responses.pt');(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('completed',flush=True)
if __name__=='__main__':main()

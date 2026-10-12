"""Independently recompute Gram and released-solver parity; audit truth gradients."""
import json,hashlib
from pathlib import Path
import torch,pandas as pd
import FiniteDifferencePyTorchConv as FD
import Utilities
from discrete_reference import DiscreteReference,time_cost,variable_mask

def main():
    torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
    root=Path(__file__).resolve().parent;out=Path('.build/author-fwi-response-gram-v2-20261008');r=json.loads((out/'report.json').read_text());saved=torch.load(out/'responses.pt',weights_only=True,map_location='cpu');a=saved['responses'];g=a.T@a; norms=g.diag().sqrt();corr=g/(norms[:,None]*norms[None,:]);assert torch.allclose(g,saved['gram'],atol=1e-10,rtol=1e-10) and torch.allclose(corr,saved['correlation']);assert torch.isfinite(a).all();assert all(v['two_epsilon_relative_difference']<1e-4 for v in r['directions'].values())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for p,h in r['input_hashes'].items():assert sha(root/p)==h
    for p,v in json.loads((root/'source_manifest.json').read_text())['files'].items():assert sha(root/p)==v['sha256']
    cfg=pd.read_csv(root/'settings2D.csv').iloc[0];nx,ny,n,ns=(int(cfg[k]) for k in ['Nx','Ny','N','numberOfSources']);dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');solver=FD.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device).double();sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),ns);rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx);ref=DiscreteReference(solver,rx,ry)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values,dtype=torch.float64).reshape(ns,1,nx+3,ny+3,n).to(device);truth=torch.ones(1,1,nx+3,ny+3,device=device);truth[0,0,1:-1,1:-1]=torch.tensor(pd.read_hdf(root/'data_casestudy/material1.h5').values,device=device);mask=variable_mask(truth,rx,ry,sx,sy)
    with torch.no_grad():
        z=truth.new_zeros(ns,1,nx+3,ny+3);wave=solver.forwardNSteps(z.clone(),z.clone(),truth,force,nx,ny,n,ns,device);author=wave[:,rx,ry,1:].clone();del wave
    reference=saved['truth_prediction'].to(device);parity=float((author-reference).norm()/reference.norm());assert parity<1e-10
    print('author parity '+str(parity),flush=True)
    obs=saved['observation'].to(device);results={}
    for name,target in [('released_observation',obs),('same_solver_truth_observation',reference)]:
        var=truth.detach().clone().requires_grad_(True);pred=ref.traces(var,force);loss=time_cost(pred,target,dx,dy,dt);grad=torch.autograd.grad(loss*1e12,var)[0]*mask;assert torch.isfinite(grad).all()
        opt=torch.optim.Adam([var],lr=.02);var.grad=grad.clone();torch.nn.utils.clip_grad_norm_([var],1e-5);opt.step()
        with torch.no_grad():
            var.clamp_(float(cfg['gamma0']),1);final=ref.traces(var,force,False);cost=float(time_cost(final,target,dx,dy,dt));mse=float(.5*(var-truth).square().mean())
        results[name]={'initial_cost':float(loss.detach()),'scaled_masked_gradient_norm':float(grad.norm()),'after_one_step_cost':cost,'after_one_step_material_half_mse':mse,'after_step_gamma_max_change':float((var-truth).abs().max())};print(name+' '+json.dumps(results[name]),flush=True)
    assert results['same_solver_truth_observation']['initial_cost']==0 and results['same_solver_truth_observation']['scaled_masked_gradient_norm']==0 and results['same_solver_truth_observation']['after_step_gamma_max_change']<1e-12
    report={'status':'passed','response_gram_recomputed':'passed','double_two_epsilon_response_stability':'all below1e-4','seven_author_files':'unchanged','input_hashes':'passed','author_forward_vs_reference_relative_l2':parity,'truth_gradient_one_step_control':results,'scope':'Case1 development diagnostic, no validation/test or reconstruction success claim.'};(out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()

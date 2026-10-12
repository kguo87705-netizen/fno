"""Reload all four arm endpoints with the frozen observations and check provenance."""
import hashlib,json
from pathlib import Path
import torch,pandas as pd
import FiniteDifferencePyTorchConv as FD
import Utilities
from discrete_reference import DiscreteReference,time_cost

def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parent;out=Path('.build/author-observation-control-20261008');r=json.loads((out/'summary.json').read_text());assert r['status']=='completed'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,d in json.loads((root/'source_manifest.json').read_text())['files'].items():assert sha(root/name)==d['sha256']
    for name,h in r['protocol']['input_hashes'].items():assert sha(root/name)==h
    for name,d in r['generation_sources'].items():assert sha(out/'downloaded_generation_sources'/name)==d['sha256']
    assert sha(root/'run_observation_consistency_control.py')==r['protocol']['script_sha256']
    cfg=pd.read_csv(root/'settings2D.csv').iloc[0];nx,ny,n,ns=(int(cfg[k]) for k in ['Nx','Ny','N','numberOfSources']);dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');solver=FD.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device);sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),ns);rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx);ref=DiscreteReference(solver,rx,ry)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values).reshape(ns,1,nx+3,ny+3,n).to(device);frozen=torch.load(out/'frozen_observations.pt',weights_only=True,map_location='cpu');truth=frozen['truth'].to(device);checks={}
    with torch.no_grad():assert torch.equal(ref.traces(truth,force,False).cpu(),frozen['self_consistent'])
    for name,arm in r['results'].items():
        data=torch.load(out/name/'states.pt',weights_only=True,map_location='cpu');history=data['history'];assert history.shape==(36,1,nx+3,ny+3) and torch.isfinite(history).all()
        expected=torch.ones_like(truth) if name.endswith('_uniform') else .95*truth+.05;assert torch.equal(history[0:1],expected.cpu())
        for st in data['final_optimizer']['state'].values():assert int(st['step'])==35 and all(torch.isfinite(v).all() for v in st.values() if isinstance(v,torch.Tensor))
        target=frozen['self_consistent' if name.startswith('self_consistent') else 'released'].to(device);maxdiff=0
        for index in [0,35]:
            field=history[index:index+1].to(device)
            with torch.no_grad():pred=ref.traces(field,force,False)
            if index==35:assert torch.equal(pred.cpu(),data['prediction_final'])
            vals={'cost':float(time_cost(pred,target,dx,dy,dt)),'waveform_relative_l2':float((pred-target).double().norm()/target.double().norm()),'material_half_mse':float(.5*(field-truth).square().mean()),'physical_material_half_mse':float(.5*(field[...,1:-1,1:-1]-truth[...,1:-1,1:-1]).square().mean())}
            for k,v in vals.items():assert abs(v-arm['records'][index][k])<1e-8;maxdiff=max(maxdiff,abs(v-arm['records'][index][k]))
        checks[name]={'endpoint_reload':'bit_identical','max_metric_difference':maxdiff,'finite_adam_steps':35}
    report={'status':'passed','seven_author_files':'unchanged','inputs_and_downloaded_generation_sources':'SHA verified','same_solver_truth_observation':'bit_identical','arms':checks,'scope':'Case1 fixed35 development; no validation/test, independent-physics or general reconstruction claim.'};(out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

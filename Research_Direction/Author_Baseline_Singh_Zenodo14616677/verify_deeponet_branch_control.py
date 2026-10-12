"""Frozen DeepONet final reload, sampled order and finite optimizer audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from deeponet_time_features import TimeFeatureDeepONet as BasicDeepONet
from operator_data import OperatorPilotDataset
from deeponet_time_features import grid_predict as predict7
from run_four_materials import evaluate,REGIONS
from audit_four_material_response import energies,metrics
from time_reconstruction import cubic_time_reconstruction
from train_multiworkpiece_control import state_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    r=json.loads((a.run/'comparison.json').read_text());protocol=r['protocol'];grid_predict=lambda model,sample,gamma,device:predict7(model,sample,gamma,device,protocol['time_feature_mode']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert r['status']=='completed' and protocol['updates']==4000 and protocol['selection'].endswith('no validation/test access')
    assert sha(a.data/'manifest.json')==protocol['data_manifest_sha256'] and sha(a.data/'homogeneous.pt')==protocol['homogeneous_sha256']
    for name,key in [('basic_deeponet.py','model_sha256'),('run_deeponet_branch_control.py','script_sha256'),('deeponet_time_features.py','feature_module_sha256')]:assert sha(Path(__file__).with_name(name))==protocol[key]
    originals=json.loads(Path(__file__).with_name('source_manifest.json').read_text())
    for name,v in originals['files'].items():assert sha(Path(__file__).parent/name)==v['sha256']
    ds=OperatorPilotDataset(a.data,'train');s=ds[0];assert protocol['training_groups']==[s['group_id']] and s['shot']==0
    generator=torch.Generator().manual_seed(protocol['sampling_seed']);h=hashlib.sha256()
    for _ in range(4000):h.update(torch.randint(256*128*500,(32768,),generator=generator).numpy().tobytes())
    assert h.hexdigest()==r['sampled_indices_sha256']
    torch.manual_seed(2);model=BasicDeepONet();assert state_hash(model)==protocol['initial_state_hash'];assert state_hash(model.branch)==protocol['initial_branch_hash'];initial_branch={k:v.clone() for k,v in model.branch.state_dict().items()};model.to(device)
    c=torch.load(a.run/'final.pt',weights_only=True,map_location=device);assert c['updates']==4000 and c['scale']==ds.scale
    if protocol['arm']=='frozen_initial_branch':
        for p in model.branch.parameters():p.requires_grad_(False)
    assert [name for name,p in model.named_parameters() if p.requires_grad]==protocol['trainable_parameters']
    assert len(c['optimizer_state_dict']['param_groups'][0]['params'])==len(protocol['trainable_parameters'])
    assert len(c['optimizer_state_dict']['state'])==len(protocol['trainable_parameters'])
    for st in c['optimizer_state_dict']['state'].values():
        assert int(st['step'])==4000;assert all(torch.isfinite(v).all() for v in st.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(v).all() for v in c['state_dict'].values());model.load_state_dict(c['state_dict']);assert state_hash(model.branch)==r['final_branch_hash']
    if protocol['arm']=='frozen_initial_branch':
        assert state_hash(model.branch)==protocol['initial_branch_hash']
        assert all(torch.equal(initial_branch[k],p.cpu()) for k,p in model.branch.state_dict().items())
    model.eval();bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    actual=grid_predict(model,s,s['gamma'],device);example=torch.load(a.run/'training_example.pt',weights_only=True,map_location='cpu');assert torch.equal(actual.cpu(),example['prediction_residual_field'])
    actual_metrics,_=evaluate(None,[s],bg,ds.scale,ds.manifest['normalization']['trace_rms'],device,fixed=actual)
    maxdiff=max(abs(actual_metrics[k]-r['final_training'][k]) for k,v in actual_metrics.items() if isinstance(v,(float,np.floating)));assert maxdiff<1e-7
    homogeneous=grid_predict(model,s,torch.ones_like(s['gamma']),device);response=actual-homogeneous;true=s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale;ids=s['receiver_indices'].to(device)
    trace=cubic_time_reconstruction(response[ids[:,0],ids[:,1],:]*ds.scale,s['time_seconds'].to(device),torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9)
    values={'field':energies(response,true),'trace':energies(trace,s['traces_full'].to(device)-bg['traces_full'][0].to(device)),**{name:energies(response[:,ys,:],true[:,ys,:]) for name,ys in REGIONS.items()}}
    for name,v in values.items():
        z=metrics(v);assert abs(z['relative_l2']-r['material_response_training_only'][name]['relative_l2'])<1e-7;assert np.isclose(v['error'],v['predicted']+v['reference']-2*v['dot'],rtol=1e-7)
    result={'status':'passed','updates':4000,'sampled_indices_sequence_sha256':'recomputed_exact','same_initialization_and_source_data_hashes':'passed','seven_author_files':'unchanged','full_grid_prediction_reload':'bit_identical','reload_max_metric_difference':maxdiff,'training_only_material_response_reload_and_energy_identity':'passed','Adam_steps_and_finite_states':'passed','branch_weight_integrity_and_optimizer_parameter_selection':'passed','scope':'Training000/source0 only. No validation/test, physical gradient or architecture superiority claim.'}
    (a.run/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()

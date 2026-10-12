"""Behavioral and structural verification for the 131-input control."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from deeponet_multifrequency import MultiFrequencyDeepONet, grid_predict
from operator_data import OperatorPilotDataset
from run_four_materials import evaluate, REGIONS
from audit_four_material_response import energies, metrics
from time_reconstruction import cubic_time_reconstruction
from train_multiworkpiece_control import state_hash

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--run',type=Path,required=True); a=p.parse_args(); torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'cpu'
    r=json.loads((a.run/'comparison.json').read_text()); q=r['protocol']; sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    assert r['status']=='completed' and q['updates']==4000 and q['arm'] in ['zero126','fourier126']
    assert q['branch']=='frozen_initial_seed2' and q['trainable_parameters']==['bias','trunk.0.weight','trunk.0.bias','trunk.2.weight','trunk.2.bias','trunk.4.weight','trunk.4.bias']
    assert sha(a.data/'manifest.json')==q['data_manifest_sha256'] and sha(a.data/'homogeneous.pt')==q['homogeneous_sha256']
    assert sha(Path(__file__).with_name('deeponet_multifrequency.py'))==q['feature_module_sha256']
    ds=OperatorPilotDataset(a.data,'train'); s=ds[0]; assert q['training_groups']==[s['group_id']] and s['shot']==0
    generator=torch.Generator().manual_seed(19); h=hashlib.sha256()
    for _ in range(4000): h.update(torch.randint(256*128*500,(32768,),generator=generator).numpy().tobytes())
    assert h.hexdigest()==r['sampled_indices_sha256']
    torch.manual_seed(2); model=MultiFrequencyDeepONet(); assert state_hash(model)==q['initial_state_hash']; initial={k:v.clone() for k,v in model.branch.state_dict().items()}; initial_branch=state_hash(model.branch)
    for z in model.branch.parameters(): z.requires_grad_(False)
    assert initial_branch==q['initial_branch_hash']; assert [n for n,z in model.named_parameters() if z.requires_grad]==q['trainable_parameters']
    c=torch.load(a.run/'final.pt',weights_only=True,map_location=device); assert c['updates']==4000; assert len(c['optimizer_state_dict']['param_groups'][0]['params'])==7; assert len(c['optimizer_state_dict']['state'])==7
    for st in c['optimizer_state_dict']['state'].values(): assert int(st['step'])==4000 and all(torch.isfinite(v).all() for v in st.values() if isinstance(v,torch.Tensor))
    assert all(torch.isfinite(v).all() for v in c['state_dict'].values()); model.to(device); model.load_state_dict(c['state_dict']); assert state_hash(model.branch)==q['initial_branch_hash']; assert all(torch.equal(initial[k],v.cpu()) for k,v in model.branch.state_dict().items())
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu'); actual=grid_predict(model,s,s['gamma'],device,q['arm']); example=torch.load(a.run/'training_example.pt',weights_only=True,map_location='cpu'); assert torch.equal(actual.cpu(),example['prediction_residual_field'])
    actual_metrics,_=evaluate(None,[s],bg,ds.scale,ds.manifest['normalization']['trace_rms'],device,fixed=actual); diffs=[abs(actual_metrics[k]-r['final_training'][k]) for k in actual_metrics if isinstance(actual_metrics[k],(float,np.floating))]; assert max(diffs)<1e-7
    report={'status':'passed','updates':4000,'arm':q['arm'],'same_initialization_and_sampling':'passed','frozen_branch_state':'bit_identical_to_initial','optimizer_parameter_selection':'passed','finite_adam_state':'passed','full_grid_reload':'bit_identical','reload_max_metric_difference':max(diffs),'scope':'Training000/source0 only. No validation/test, physical-gradient/FWI or generalization claim.'}; (a.run/'artifact_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report),flush=True)
if __name__=='__main__': main()

"""Reload fixed final model, frozen training mean and grouped energy checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from train_multiworkpiece_control import state_hash
from run_four_materials import load_samples,evaluate
from run_receiver_loss_control import supervised_losses


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'comparison.json').read_text());protocol=r['protocol'];sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert r['status']=='completed' and protocol['sample_order']==[i%4 for i in range(1600)]
    assert all(protocol['sample_order'].count(i)==400 for i in range(4))
    assert sha(a.data/'manifest.json')==protocol['data_manifest_sha256'] and sha(a.data/'homogeneous.pt')==protocol['homogeneous_sha256']
    assert sha(Path(__file__).parent/'run_four_materials.py')==protocol['script_sha256']
    original=json.loads((Path(__file__).parent/'source_manifest.json').read_text())
    for name,v in original['files'].items():assert sha(Path(__file__).parent/name)==v['sha256']
    torch.manual_seed(2);initial=BasicFNO(input_channels=8);assert state_hash(initial)==protocol['initial_state_hash'];del initial
    device='cuda' if torch.cuda.is_available() else 'cpu';ds,train,val=load_samples(a.data);assert not ({s['group_id'] for s in train}&{s['group_id'] for s in val})
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu');saved_mean=torch.load(a.run/'training_mean.pt',weights_only=True,map_location=device)
    assert saved_mean['training_groups']==protocol['training_groups'] and saved_mean['scale']==ds.scale
    expected=torch.stack([s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale for s in train]).mean(0)
    assert torch.equal(expected,saved_mean['normalized_residual']);del expected
    lf,lr=supervised_losses(torch.zeros_like(saved_mean['normalized_residual'])[None,None],saved_mean['normalized_residual'][None,None],train[0]['receiver_indices'].to(device),ds.scale,ds.manifest['normalization']['trace_rms'])
    assert abs((r['references']['zero']['train']['mean_total_objective']-r['references']['training_mean']['train']['mean_total_objective'])-float(lf+lr))<2e-7
    c=torch.load(a.run/'final.pt',weights_only=True,map_location=device);assert c['updates']==1600 and c['scale']==ds.scale and c['config']['input_channels']==8
    assert all(torch.isfinite(v).all() for v in c['state_dict'].values())
    for state in c['optimizer_state_dict']['state'].values():
        assert int(state['step'])==1600
        assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
    model=BasicFNO(input_channels=8).to(device);model.load_state_dict(c['state_dict']);model.eval();differences={};sr=ds.manifest['normalization']['trace_rms']
    for split,samples in [('train',train),('validation',val)]:
        actual,_=evaluate(model,samples,bg,ds.scale,sr,device);saved=r[f'final_{split if split=="validation" else "training"}']
        maxdiff=max(abs(actual[k]-saved[k]) for k in actual if isinstance(actual[k],(float,np.floating)));assert maxdiff<1e-7;differences[split]=maxdiff
        fixed,_=evaluate(None,samples,bg,ds.scale,sr,device,fixed=saved_mean['normalized_residual'])
        assert abs(fixed['mean_total_objective']-r['references']['training_mean'][split]['mean_total_objective'])<1e-7
        assert np.isclose(sum(v['energy_sums']['error'] for v in actual['regions'].values()),actual['energy_sums']['field_error'],rtol=1e-7)
    mr=json.loads((a.run/'material_response.json').read_text())
    for name,z in mr['summary'].items():
        e=z['energy_sums'];assert np.isclose(e['error'],e['predicted']+e['reference']-2*e['dot'],rtol=1e-7)
    result={'status':'passed','updates':1600,'exposures_per_training_workpiece':400,'same_eight_channel_initialization':'passed',
        'frozen_mean_recomputed_from_four_training_workpieces_only':'passed','quadratic_training_mean_identity':'passed','group_disjointness':'passed',
        'final_reload_max_metric_differences':differences,'fixed_mean_references_recomputed':'passed','region_and_material_response_energy_identities':'passed',
        'Adam_steps_and_finite_states':'passed','seven_original_files_and_data_hashes':'unchanged',
        'scope':'Four fitted training materials and six development validation materials, source0 only, one seed; no test or physical-gradient/FWI evaluation.'}
    (a.run/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

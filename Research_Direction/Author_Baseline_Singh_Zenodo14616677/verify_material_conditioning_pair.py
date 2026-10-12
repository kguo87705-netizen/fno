"""Check frozen pair models and the material-independent quadratic oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from run_material_conditioning_pair import load_pair,pair_metrics
from run_receiver_loss_control import supervised_losses


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'comparison.json').read_text());protocol=r['protocol'];sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert r['status']=='completed' and protocol['initial_state_hashes'][0]==protocol['initial_state_hashes'][1]
    assert protocol['sample_order']==[i%2 for i in range(800)] and all(protocol['sample_order'].count(i)==400 for i in [0,1])
    assert sha(a.data/'manifest.json')==protocol['data_manifest_sha256'] and sha(a.data/'homogeneous.pt')==protocol['homogeneous_sha256']
    assert sha(Path(__file__).parent/'run_material_conditioning_pair.py')==protocol['script_sha256']
    original=json.loads((Path(__file__).parent/'source_manifest.json').read_text())
    for name,info in original['files'].items():assert sha(Path(__file__).parent/name)==info['sha256']
    device='cuda' if torch.cuda.is_available() else 'cpu';ds,samples,bg,targets,inputs=load_pair(a.data,device);maximums=[]
    mean=(targets[0]+targets[1])/2
    with torch.no_grad():
        oracle,_=pair_metrics([mean,mean],samples,targets,bg,ds.scale,ds.manifest['normalization']['trace_rms'])
        assert abs(oracle['mean_total_objective']-r['mean_oracle']['mean_total_objective'])<1e-8
        for arm in [0,1]:
            c=torch.load(a.run/f'final_arm{arm}.pt',weights_only=True,map_location=device);assert c['updates']==800 and c['scale']==ds.scale and c['arm']==protocol['arms'][arm]
            assert all(torch.isfinite(x).all() for x in c['state_dict'].values())
            for state in c['optimizer_state_dict']['state'].values():
                for value in state.values():
                    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
                assert int(state['step'])==800
            model=BasicFNO(input_channels=8).to(device);assert model.config==c['config'];model.load_state_dict(c['state_dict']);model.eval()
            predictions=[model(x)[0,0] for x in inputs[arm]];actual,_=pair_metrics(predictions,samples,targets,bg,ds.scale,ds.manifest['normalization']['trace_rms'])
            diff=max(abs(actual[k]-r['final'][arm][k]) for k in actual if isinstance(actual[k],(float,np.floating)));assert diff<1e-7;maximums.append(diff)
            for key in ['pair_field_response','pair_trace_response']:
                z=actual[key];e=z['energy_sums'];assert np.isclose(z['relative_l2']**2,(e['predicted']+e['reference']-2*e['dot'])/e['reference'],atol=1e-9)
            if arm==1:
                assert torch.equal(predictions[0],predictions[1]) and actual['pair_trace_response']['norm_ratio']==0.
                lf,lr=supervised_losses(predictions[0][None,None],mean[None,None],samples[0]['receiver_indices'].to(device),ds.scale,ds.manifest['normalization']['trace_rms'])
                excess=actual['mean_total_objective']-oracle['mean_total_objective'];assert excess>=-1e-7
                assert abs(excess-float(lf+lr))<2e-7
            del model,c,predictions
    result={'status':'passed','updates_per_arm':800,'exposures_per_workpiece':400,'paired_initial_hash_equal':True,
        'final_model_reload_max_metric_difference':maximums,'constant_input_predictions_identical':True,
        'quadratic_mean_oracle_excess_loss_identity':'passed','pair_response_energy_identity':'passed',
        'Adam_state_steps_and_finiteness':'passed','seven_original_files_manifest_and_homogeneous_hashes':'unchanged',
        'scope':'Two fitted training workpieces, same shot; no validation/test, local physical-gradient or FWI evidence.'}
    (a.run/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

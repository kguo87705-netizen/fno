"""Behavioral and frozen-checkpoint verification for the paired trial."""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input
from time_feature_input import encode_time_features
from train_multiworkpiece_control import state_hash
from train_scatter_multiworkpiece import evaluate_dataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--control',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'comparison.json').read_text());m=json.loads((a.run/'material_response.json').read_text())
    old=json.loads((a.control/'comparison.json').read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert r['status']=='completed' and r['initial_state_hashes'][0]==r['initial_state_hashes'][1]
    assert r['sample_orders']==old['sample_orders'] and len(r['sample_orders'])==20
    assert all(sorted(x)==list(range(48)) for x in r['sample_orders'])
    assert sha(a.data/'manifest.json')==r['protocol']['data_manifest_sha256']
    assert sha(a.data/'homogeneous.pt')==r['protocol']['homogeneous_sha256']
    originals=json.loads((Path(__file__).parent/'source_manifest.json').read_text())
    for name,info in originals['files'].items():assert sha(Path(__file__).parent/name)==info['sha256']
    torch.manual_seed(2);legacy=BasicFNO();assert state_hash(legacy)==old['initial_state_hashes'][1];del legacy
    tiny={'gamma':torch.ones(3,2,requires_grad=True),'x_m':torch.linspace(0,.1,3),'y_m':torch.linspace(0,.05,2),
        'source_xy_m':torch.tensor([.04,.05]),'time_seconds':torch.arange(1,501,dtype=torch.float64)*120e-9}
    z=encode_time_features(tiny,mode='zero');f=encode_time_features(tiny,mode='sincos')
    assert torch.equal(z[:6],encode_fno_input(tiny)) and torch.equal(z[:6],f[:6]) and not z[6:].count_nonzero()
    assert torch.allclose(f[6].square()+f[7].square(),torch.ones_like(f[6]),atol=2e-7)
    assert int(torch.fft.rfft(f[6,0,0]).abs().argmax())==30
    f[0].sum().backward();assert torch.equal(tiny['gamma'].grad,torch.full((3,2),500.))
    # Known linear time signal checks interpolation, physical scale and energy aggregation.
    t=tiny['time_seconds'];full=torch.arange(1,2001,dtype=torch.float64)*30e-9
    target=(t/60e-6).float()[None,None,None,:].expand(1,3,2,500).clone()
    mock={'shot':0,'target':target,'receiver_indices':torch.tensor([[0,1],[2,1]]),'time_seconds':t,
        'traces_full':(full/60e-6).float()[None,:].expand(2,2000).clone(),'group_id':'fixture'}
    class Dataset:
        scale=1.;manifest={'dt_seconds':30e-9};entries=[({'split':'validation'},0)]
        def __len__(self):return 4
        def __getitem__(self,index):return mock
    class Model:
        def __init__(self,exact):self.exact=exact
        def __call__(self,x):return target[None] if self.exact else torch.zeros_like(target[None])
    fixturebg={'wavefield_snapshots':torch.zeros(1,3,2,500),'traces_full':torch.zeros(1,2,2000)}
    encoder=lambda sample:torch.zeros(8,3,2,500)
    zero=evaluate_dataset(Model(False),Dataset(),fixturebg,'cpu',encoder);exact=evaluate_dataset(Model(True),Dataset(),fixturebg,'cpu',encoder)
    assert zero['scatter_field_relative_l2']==zero['scatter_trace_relative_l2']==1.
    assert exact['scatter_field_relative_l2']==0. and exact['scatter_trace_relative_l2']<2e-7
    ds=OperatorPilotDataset(a.data,'validation');bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu');device='cuda' if torch.cuda.is_available() else 'cpu'
    differences=[]
    for arm in range(2):
        c=torch.load(a.run/f'final_weight{arm}.pt',weights_only=True,map_location=device)
        assert c['epoch']==20 and c['updates']==960 and c['scale']==ds.scale and c['receiver_loss_weight']==1.
        assert c['config']['input_channels']==8 and c['time_feature_mode']==['zero','sincos'][arm]
        assert all(torch.isfinite(x).all() for x in c['state_dict'].values())
        cfg=c['config'];model=BasicFNO(width=cfg['width'],modes=tuple(cfg['modes']),layers=cfg['layers'],input_channels=8).to(device);model.load_state_dict(c['state_dict']);model.eval()
        actual=evaluate_dataset(model,ds,bg,device,partial(encode_time_features,mode=c['time_feature_mode']))
        maxdiff=max(abs(actual[k]-r['final_validation'][arm][k]) for k in actual if k.endswith('relative_l2'));assert maxdiff<1e-7;differences.append(maxdiff)
        mr=m['models'][arm];norm=mr['response_norm_ratio'];cos=mr['response_cosine'];error=mr['secant_response_relative_l2']
        assert np.isclose(error**2,1+norm**2-2*norm*cos,atol=1e-7)
        del model,c
    result={'status':'passed','paired_initial_weights_equal':True,'legacy_six_channel_initialization_preserved':True,
        'orders_match_twenty_permutations':True,'updates_per_arm':960,'validation_shots_recomputed_per_arm':24,
        'checkpoint_reload_max_metric_difference':differences,'known_zero_and_exact_linear_residual_checks':'passed',
        'input_frequency_unit_circle_and_material_autograd_checks':'passed','material_response_energy_identity':'passed',
        'data_manifest_homogeneous_and_seven_original_source_hashes':'unchanged',
        'scope':'No test reads, physical gradient, real data or inverse evaluation.'}
    (a.run/'artifact_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

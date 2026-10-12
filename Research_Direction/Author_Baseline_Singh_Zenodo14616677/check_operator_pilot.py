"""Check grouping, physical sample mapping, normalization and sparse-time QA."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from operator_data import OperatorPilotDataset, encode_fno_input, sample_receivers
from time_reconstruction import cubic_time_reconstruction


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--time-audit',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.folder/'manifest.json').read_text())
    records=manifest['records']
    assert len({r['group_id'] for r in records})==len(records)
    assert len({r['gamma_sha256'] for r in records})==len(records)
    summaries=[]
    trace_square_sum=wave_square_sum=0.;trace_count=wave_count=0
    for r in records:
        data=torch.load(args.folder/r['file'],weights_only=True)
        assert hashlib.sha256(data['gamma'].numpy().tobytes()).hexdigest()==r['gamma_sha256']
        field,traces,indices=data['wavefield_snapshots'],data['traces_full'],data['snapshot_trace_indices']
        assert tuple(field.shape)==(4,256,128,indices.numel()) and tuple(traces.shape)==(4,24,2000)
        assert torch.isfinite(field).all() and torch.isfinite(traces).all()
        rx,ry=data['receiver_physical_indices'].T
        assert torch.equal(field[:,rx,ry,:],traces[...,indices])
        restored=cubic_time_reconstruction(traces[...,indices],data['time_seconds_snapshots'],data['time_seconds_full'])
        error=float((restored-traces).norm()/traces.norm())
        stride_errors={}
        for stride in (2,4,8):
            selected_indices=torch.arange(stride-1,2000,stride)
            decoded=cubic_time_reconstruction(traces[...,selected_indices],data['time_seconds_full'][selected_indices],data['time_seconds_full'])
            stride_errors[str(stride)]=float((decoded-traces).norm()/traces.norm())
        summaries.append({'group_id':r['group_id'],'split':r['split'],'linear_error':r['linear_temporal_interpolation_relative_error'],'cubic_error':error,'cubic_stride_errors':stride_errors})
        if r['split']=='train':
            trace_square_sum+=float(traces.double().square().sum());trace_count+=traces.numel()
            wave_square_sum+=float(field.double().square().sum());wave_count+=field.numel()
    assert np.isclose(np.sqrt(trace_square_sum/trace_count),manifest['normalization']['trace_rms'])
    assert np.isclose(np.sqrt(wave_square_sum/wave_count),manifest['normalization']['wavefield_rms'])
    lengths={}
    for split in ('train','validation','test'):
        dataset=OperatorPilotDataset(args.folder,split)
        lengths[split]=len(dataset)
        sample=dataset[0]
        gamma=sample['gamma'].clone().requires_grad_(True)
        features=encode_fno_input(sample,gamma)
        assert features.shape==(6,256,128,sample['time_seconds'].numel())
        assert torch.isfinite(features).all()
        features[0].mean().backward()
        assert torch.allclose(gamma.grad,torch.full_like(gamma,1/gamma.numel()))
        selected=sample_receivers(sample['target'],sample)[0]
        assert torch.allclose(selected,sample['traces_full'][...,sample['snapshot_trace_indices']],rtol=1e-6,atol=1e-7)
        # Temporal decoder must retain an unbroken derivative to its sparse input.
        sparse=selected.clone().requires_grad_(True)
        full_times=torch.arange(1,2001,dtype=torch.float64)*manifest['dt_seconds']
        decoded=cubic_time_reconstruction(sparse,sample['time_seconds'],full_times)
        decoded.square().mean().backward()
        assert torch.isfinite(sparse.grad).all() and sparse.grad.norm()>0
        del features
    # Independent polynomial check: the interpolator reproduces cubics with p(0)=0.
    knots=torch.linspace(.1,1,10,dtype=torch.float64);query=torch.linspace(.01,1,100,dtype=torch.float64)
    expected=query**3-2*query**2+query
    polynomial=cubic_time_reconstruction(knots**3-2*knots**2+knots,knots,query)
    assert torch.allclose(polynomial,expected,atol=1e-14,rtol=1e-12)
    nt=manifest['records'][0]['wavefield_shape'][-1]
    report={'status':'pilot_interface_checks_passed','groups':len(records),'shot_samples_by_split':lengths,'input_shape':[6,256,128,nt],'output_shape':[1,256,128,nt],'records':summaries,'normalization_train_only_verified':True,'input_material_derivative_preserved':True,'temporal_decoder_derivative_preserved':True,'limitation':'Cubic waveform reconstruction checked on pilot traces; full FWI objective and material-gradient distortion still require separate validation. No neural network trained.'}
    if args.time_audit:
        audit=json.loads(args.time_audit.read_text())
        assert audit['stride']==manifest['snapshot_stride'] and audit['screening_passed']
        report['time_sampling_audit']=str(args.time_audit)
        report['time_sampling_screening_passed']=True
        report['limitation']='Sparse-time reference separately screened on four author cases at two states. Unseen-geometry FWI and neural-network approximation not validated; no network trained.'
    (args.folder/'interface_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

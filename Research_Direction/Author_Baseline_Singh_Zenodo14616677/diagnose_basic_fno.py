"""Fixed 400-update training-only memorization and train/validation spectral audit."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset, encode_fno_input
from time_reconstruction import cubic_time_reconstruction


def spectral_audit(field, modes=(16,8,64)):
    # Full-FFT rectangular signal audit: x[-mx,mx-1], y[-my,my-1], t[-mt+1,mt-1].
    # This descriptive mask is not the exact Hermitian output of the learned rFFT layer.
    spectrum=torch.fft.fftn(field,dim=(-3,-2,-1),norm='ortho')
    mx,my,mt=modes;nx,ny,nt=field.shape
    keepx=torch.zeros(nx,dtype=torch.bool,device=field.device);keepx[:mx]=True;keepx[-mx:]=True
    keepy=torch.zeros(ny,dtype=torch.bool,device=field.device);keepy[:my]=True;keepy[-my:]=True
    keept=torch.zeros(nt,dtype=torch.bool,device=field.device);keept[:mt]=True;keept[-(mt-1):]=True
    energy=spectrum.abs().square();total=energy.sum()
    temporal=energy.sum((0,1))
    fraction=float(energy[keepx][:,keepy][:,:,keept].sum()/total)
    timefraction=float(temporal[keept].sum()/total)
    return {'joint_retained_energy_fraction':fraction,'temporal_retained_energy_fraction':timefraction,
            'ideal_linear_joint_truncation_relative_l2':float(np.sqrt(max(0,1-fraction)))},temporal.cpu().numpy()


def evaluate(model,inputs,target,sample):
    with torch.no_grad():
        prediction=model(inputs)[0,0]
        indices=sample['receiver_indices'].to(prediction.device)
        sparse=prediction[indices[:,0],indices[:,1],:]*sample['wavefield_scale']
        times=torch.arange(1,2001,device=prediction.device,dtype=prediction.dtype)*3e-8
        full=cubic_time_reconstruction(sparse,sample['time_seconds'].to(prediction.device),times)
        reference=sample['traces_full'].to(prediction.device)
        amplitude=float(full.norm()/reference.norm())
        correlation=float((full*reference).sum()/(full.norm()*reference.norm()).clamp_min(1e-30))
        metrics={'field_relative_l2':float((prediction-target[0,0]).norm()/target.norm()),
                 'trace_relative_l2':float((full-reference).norm()/reference.norm()),
                 'trace_norm_ratio':amplitude,'trace_zero_lag_correlation':correlation,
                 'normalized_field_mse':float((prediction-target[0,0]).square().mean())}
    return metrics,full.cpu(),prediction.cpu()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'protocol.json').exists(),'Use new output directory'
    protocol={'updates':400,'evaluation_updates':[0,50,100,200,400],'selection':'final fixed update; no selection',
              'seed':2,'learning_rate':.001,'training_sample':'first train workpiece, shot 0',
              'data_access':['train','validation'],'no_test_access':True,
              'purpose':'memorization diagnosis, not held-out generalization or material-gradient validation'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=OperatorPilotDataset(args.data,'train');sample=train[0]
    assert sample['split']=='train'
    inputs=encode_fno_input(sample).unsqueeze(0).to(device);target=sample['target'].unsqueeze(0).to(device)
    model=BasicFNO().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];start=time.perf_counter()
    for update in range(401):
        if update in protocol['evaluation_updates']:
            model.eval();metrics,traces,field=evaluate(model,inputs,target,sample)
            history.append({'update':update,**metrics});print(json.dumps(history[-1]),flush=True)
            (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if update==400:break
        model.train();optimizer.zero_grad(set_to_none=True);prediction=model(inputs)
        loss=(prediction-target).square().mean();assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();del prediction,loss
        if (update+1)%25==0:print(f'Update {update+1}/400, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    torch.save({'state_dict':model.state_dict(),'config':model.config,'update':400,'group_id':sample['group_id'],
                'shot':sample['shot'],'scale':train.scale,'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()},args.out/'final.pt')
    torch.save({'prediction_field':field,'prediction_traces':traces,'reference_traces':sample['traces_full']},args.out/'example.pt')
    del model,optimizer,inputs,target
    # Aggregate reference spectral energies across both allowed splits.
    records=[];spectra={};totals={}
    with torch.no_grad():
        for split in ['train','validation']:
            ds=OperatorPilotDataset(args.data,split);energy_sum=np.zeros(500);weighted=0.;total=0.
            for index in range(len(ds)):
                item=ds[index];truth=item['target'][0].to(device)
                result,energy=spectral_audit(truth)
                norm=float(truth.square().sum());weighted+=result['joint_retained_energy_fraction']*norm;total+=norm;energy_sum+=energy
                records.append({'split':split,'group_id':item['group_id'],'shot':item['shot'],**result});del truth
            spectra[split]=energy_sum;totals[split]={'joint_retained_energy_fraction':weighted/total,
                'temporal_retained_energy_fraction':float((energy_sum[:64].sum()+energy_sum[-63:].sum())/energy_sum.sum()),'shots':len(ds)}
            print(f'Spectral audit complete: {split}',flush=True)
    report={'scope':protocol,'group_id':sample['group_id'],'shot':sample['shot'],'history':history,
            'spectral_summary':totals,'spectral_records':records,'elapsed_seconds':time.perf_counter()-start,
            'caution':'Linear target FFT truncation is a signal diagnostic, not a strict FNO error bound: local branches and nonlinearities can create additional frequencies.'}
    (args.out/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax[0,0].plot([r['update'] for r in history],[r['field_relative_l2'] for r in history],label='Field')
    ax[0,0].plot([r['update'] for r in history],[r['trace_relative_l2'] for r in history],label='Receiver')
    ax[0,0].set(xlabel='Update',ylabel='Relative L2',title='One training shot: fixed 400 updates');ax[0,0].legend()
    t=np.arange(1,2001)*.03
    ax[0,1].plot(t,sample['traces_full'][12].numpy(),label='Reference');ax[0,1].plot(t,traces[12].numpy(),label='FNO final',alpha=.8)
    ax[0,1].set(xlabel='Time (us)',ylabel='Scalar u',title='Training receiver 12');ax[0,1].legend()
    freq=np.fft.fftfreq(500,d=120e-9)/1e6;positive=freq>=0
    for split in spectra:ax[1,0].semilogy(freq[positive],spectra[split][positive]/spectra[split].sum(),label=split)
    ax[1,0].axvline(63/(500*120e-9)/1e6,color='k',linestyle='--',label='Last retained positive mode')
    ax[1,0].set(xlabel='Temporal frequency (MHz)',ylabel='Fraction per frequency bin',title='Reference field temporal spectrum');ax[1,0].legend()
    ax[1,1].imshow((field[:,:,250]-sample['target'][0,:,:,250]).T.numpy(),origin='lower',extent=(0,100,0,50),cmap='RdBu_r')
    ax[1,1].set(xlabel='x (mm)',ylabel='y (mm)',title='Normalized field residual at 30.12 us')
    fig.savefig(args.out/'single_sample_diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps({'final':history[-1],'spectral_summary':totals}),flush=True)


if __name__=='__main__':main()

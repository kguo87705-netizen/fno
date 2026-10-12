"""Training-only per-receiver temporal projections; not a learned operator."""
import json,hashlib
from pathlib import Path
import torch
from operator_data import OperatorPilotDataset
from time_reconstruction import cubic_time_reconstruction

def main():
    torch.set_num_threads(4);data=Path('.build/operator-pilot-v3-20261008');out=Path('.build/receiver-temporal-basis-20261008');out.mkdir(exist_ok=True)
    s=OperatorPilotDataset(data,'train')[0];assert s['group_id']=='synthetic_workpiece_000' and s['shot']==0
    bg=torch.load(data/'homogeneous.pt',weights_only=True,map_location='cpu');ids=s['receiver_indices'];target=((s['target'][0]-bg['wavefield_snapshots'][0]/s['wavefield_scale'])[ids[:,0],ids[:,1],:]*s['wavefield_scale']).double().T
    t=s['time_seconds'].double();tau=t/t[-1];phase=2*torch.pi*500000*t;constant=torch.ones_like(t)
    carrier=torch.stack([constant,tau,phase.sin(),phase.cos()],-1)
    centers=torch.linspace(0,1,16,dtype=torch.float64);windows=(1-(tau[:,None]-centers[None]).abs()*15).clamp_min(0)
    local=torch.cat([torch.stack([constant,tau],-1),windows*phase.sin()[:,None],windows*phase.cos()[:,None]],-1)
    k=torch.arange(1,64,dtype=torch.float64);angle=2*torch.pi*tau[:,None]*k[None];fourier=torch.cat([constant[:,None],angle.sin(),angle.cos()],-1)
    fulltimes=torch.arange(1,2001,dtype=torch.float64)*30e-9;fulltarget=(s['traces_full']-bg['traces_full'][0]).double();results={};predictions={}
    for name,basis in [('single_carrier',carrier),('localized_carrier16',local),('fourier_modes0_to63',fourier)]:
        solution=torch.linalg.lstsq(basis,target,driver='gelsd');fit=basis@solution.solution;residual=fit-target;normal=float((basis.T@residual).norm()/(basis.norm()*target.norm()));assert normal<1e-10
        pred=cubic_time_reconstruction(fit.T,t,fulltimes);error=pred-fulltarget
        results[name]={'basis_columns':basis.shape[1],'rank':int(solution.rank),'sparse500_relative_l2':float(residual.norm()/target.norm()),'full2000_relative_l2':float(error.norm()/fulltarget.norm()),'least_squares_normal_equation_relative_residual':normal,'records':[{'receiver':i,'sparse500_relative_l2':float(residual[:,i].norm()/target[:,i].norm()),'full2000_relative_l2':float(error[i].norm()/fulltarget[i].norm())} for i in range(24)]};predictions[name]=pred
    floor=cubic_time_reconstruction(target.T,t,fulltimes);floor_error=float((floor-fulltarget).norm()/fulltarget.norm())
    spectrum=torch.fft.rfft(target,dim=0,norm='ortho');truncated=spectrum.clone();truncated[64:]=0;fftfit=torch.fft.irfft(truncated,n=500,dim=0,norm='ortho');fft_parity=float((fftfit-(fourier@torch.linalg.lstsq(fourier,target,driver='gelsd').solution)).abs().max());assert fft_parity<1e-11
    report={'scope':'Only training000/source0, 24 receivers. Each receiver independently fits its own coefficients using its entire target waveform. This is an in-sample linear-basis oracle, not prediction, generalization, a medium-to-waveform operator, or a lower bound on nonlinear DeepONet capacity.','data_manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),'bases':results,'exact_sparse_target_cubic_full_trace_relative_l2':floor_error,'fourier_projection_vs_truncated_fft_max_absolute_discrepancy':fft_parity}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');torch.save({'reference_full':fulltarget,'predictions_full':predictions,'time_seconds_full':fulltimes,'target_sparse':target.T},out/'waveforms.pt');print(json.dumps({**report,'bases':{k:{x:y for x,y in v.items() if x!='records'} for k,v in results.items()}}),flush=True)

if __name__=='__main__':main()

"""One frozen-checkpoint test evaluation, simple baseline and one gradient audit."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities
from basic_fno import BasicFNO
from discrete_reference import DiscreteReference,time_cost,variable_mask
from operator_data import OperatorPilotDataset,encode_fno_input,sample_receivers
from time_reconstruction import cubic_time_reconstruction


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.run/'best.pt',weights_only=True,map_location=device)
    model=BasicFNO(width=checkpoint['config']['width'],modes=tuple(checkpoint['config']['modes']),layers=checkpoint['config']['layers']).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    dataset=OperatorPilotDataset(args.data,'test');assert checkpoint['scale']==dataset.scale
    background=torch.load(args.data/'homogeneous.pt',weights_only=True)
    records=[];aggregates={k:0. for k in ('field_error','field_background_error','field_true','trace_error','trace_background_error','trace_true','scatter_true')}
    plot_data=None
    with torch.no_grad():
        for i in range(len(dataset)):
            sample=dataset[i];shot=sample['shot'];truth=sample['target']*dataset.scale
            prediction=model(encode_fno_input(sample).unsqueeze(0).to(device))[0].cpu()*dataset.scale
            sparse=sample_receivers(prediction/dataset.scale,sample)[0]
            full_times=torch.arange(1,2001,dtype=torch.float64)*dataset.manifest['dt_seconds']
            traces=cubic_time_reconstruction(sparse,sample['time_seconds'],full_times)
            observation=sample['traces_full'];bg_field=background['wavefield_snapshots'][shot:shot+1];bg_trace=background['traces_full'][shot]
            values={'field_error':float((prediction-truth).double().square().sum()),'field_background_error':float((bg_field-truth).double().square().sum()),'field_true':float(truth.double().square().sum()),'trace_error':float((traces-observation).double().square().sum()),'trace_background_error':float((bg_trace-observation).double().square().sum()),'trace_true':float(observation.double().square().sum()),'scatter_true':float((observation-bg_trace).double().square().sum())}
            assert all(np.isfinite(v) for v in values.values())
            for key in aggregates:aggregates[key]+=values[key]
            records.append({'group_id':sample['group_id'],'shot':shot,'field_relative_l2':float(np.sqrt(values['field_error']/values['field_true'])),'trace_relative_l2':float(np.sqrt(values['trace_error']/values['trace_true'])),'homogeneous_trace_relative_l2':float(np.sqrt(values['trace_background_error']/values['trace_true'])),'scatter_normalized_trace_error':float(np.sqrt(values['trace_error']/values['scatter_true'])),'trace_zero_lag_correlation':float((traces*observation).sum()/traces.norm()/observation.norm())})
            if i==0:plot_data=(truth,prediction,bg_field,observation,traces,bg_trace,sample)
    report={'scope':'Frozen validation-selected ordinary FNO, one test evaluation; synthetic scalar model only.','selected_epoch':checkpoint['epoch'],'test_groups':len(dataset)//4,'test_shots':len(dataset),'field_relative_l2':float(np.sqrt(aggregates['field_error']/aggregates['field_true'])),'homogeneous_field_relative_l2':float(np.sqrt(aggregates['field_background_error']/aggregates['field_true'])),'trace_relative_l2':float(np.sqrt(aggregates['trace_error']/aggregates['trace_true'])),'homogeneous_trace_relative_l2':float(np.sqrt(aggregates['trace_background_error']/aggregates['trace_true'])),'scatter_normalized_trace_error':float(np.sqrt(aggregates['trace_error']/aggregates['scatter_true'])),'zero_prediction_relative_l2':1.0,'records':records}
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    # One held-out workpiece, one shot, homogeneous inversion initial model.
    root=Path(__file__).resolve().parent;cfg=pd.read_csv(root/'settings2D.csv').iloc[0]
    nx,ny,steps,shots=(int(cfg[k]) for k in ('Nx','Ny','N','numberOfSources'))
    dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt'])
    sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),shots);rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx)
    sample=dataset[0];observation=sample['traces_full'].unsqueeze(0).to(device)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values).reshape(shots,1,nx+3,ny+3,steps)[0:1].to(device)
    solver=FiniteDifference.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device)
    reference=DiscreteReference(solver,rx,ry)
    gamma=torch.ones((1,1,nx+3,ny+3),device=device,requires_grad=True)
    mask=variable_mask(gamma,rx,ry,sx,sy)
    physical_cost=time_cost(reference.traces(gamma,force),observation,dx,dy,dt)
    physical_gradient=torch.autograd.grad(physical_cost*1e12,gamma)[0]/1e12*mask
    variable=gamma.detach()[0,0,1:-1,1:-1].clone().requires_grad_(True)
    indices=torch.tensor(sample['receiver_indices'].numpy(),device=device)
    full_times=torch.arange(1,steps+1,dtype=torch.float64,device=device)*dt
    def objective(material):
        field=model(encode_fno_input(sample,material).unsqueeze(0))[0]
        sparse=field[0,indices[:,0],indices[:,1],:]*dataset.scale
        traces=cubic_time_reconstruction(sparse,sample['time_seconds'].to(device),full_times)
        return time_cost(traces.unsqueeze(0),observation,dx,dy,dt)
    neural_cost=objective(variable)
    neural_gradient=torch.autograd.grad(neural_cost*1e12,variable)[0]/1e12*mask[0,0,1:-1,1:-1]
    pg=physical_gradient[0,0,1:-1,1:-1]
    cosine=float((pg*neural_gradient).sum()/pg.norm()/neural_gradient.norm())
    direction=neural_gradient/neural_gradient.norm();projected=float((neural_gradient*direction).sum())
    numerical=[]
    with torch.no_grad():
        for epsilon in (.1,.05):
            fd=float((objective(variable+epsilon*direction)-objective(variable-epsilon*direction))/(2*epsilon))
            numerical.append({'epsilon':epsilon,'finite_difference':fd,'ad_projection':projected,'relative_difference':abs(fd/projected-1)})
    report['one_shot_gradient_audit']={'group_id':sample['group_id'],'shot':0,'state':'homogeneous inversion initial material','reference_cost':float(physical_cost.detach()),'neural_cost':float(neural_cost.detach()),'cosine_with_physical_gradient':cosine,'angle_degrees':float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),'relative_gradient_l2':float((pg-neural_gradient).norm()/pg.norm()),'gradient_norm_ratio':float(neural_gradient.norm()/pg.norm()),'network_internal_fd_checks':numerical,'limitation':'One held-out workpiece and one shot; not a full FWI experiment. Internal AD consistency does not establish physical-gradient accuracy.'}
    assert torch.isfinite(neural_gradient).all()
    torch.save({'physical_gradient':pg.cpu(),'neural_gradient':neural_gradient.cpu()},args.run/'gradient_fields.pt')
    (args.run/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    truth,pred,bg,observation,traces,bgtrace,sample=plot_data
    training=json.loads((args.run/'history.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    frame=250
    vmax=max(float(truth[...,frame].abs().max()),float(pred[...,frame].abs().max()))
    for ax,field,title in zip(axes[0],(truth,pred,bg),('Reference field','Ordinary FNO','Homogeneous field')):
        ax.imshow(field[0,...,frame].T,origin='lower',extent=(0,100,0,50),vmin=-vmax,vmax=vmax,cmap='RdBu_r');ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)')
    axes[1,0].plot([r['epoch'] for r in training],[r['training_normalized_mse'] for r in training],label='Train')
    axes[1,0].plot([r['epoch'] for r in training],[r['validation_normalized_mse'] for r in training],label='Validation');axes[1,0].set(title='Wavefield training loss',xlabel='Epoch',ylabel='Normalized MSE');axes[1,0].legend()
    t=np.arange(1,2001)*dt*1e6
    for trace,label in ((observation[12],'Reference'),(traces[12],'FNO'),(bgtrace[12],'Homogeneous')):axes[1,1].plot(t,trace.numpy(),label=label)
    axes[1,1].set(title='Held-out shot 0, receiver 12',xlabel='Time (us)',ylabel='Scalar u');axes[1,1].legend(fontsize=8)
    axes[1,2].bar(['FNO','Homogeneous'],[report['trace_relative_l2'],report['homogeneous_trace_relative_l2']]);axes[1,2].set(title='All 24 test shots',ylabel='Receiver relative L2')
    fig.savefig(args.run/'fno_first_training.png',dpi=160);plt.close(fig)
    print(json.dumps(report['one_shot_gradient_audit']),flush=True)


if __name__=='__main__':main()

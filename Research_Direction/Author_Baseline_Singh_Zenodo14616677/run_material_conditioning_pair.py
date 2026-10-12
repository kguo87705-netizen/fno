"""Training-only same-source two-workpiece true versus constant material control."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset
from time_feature_input import encode_time_features
from train_multiworkpiece_control import state_hash
from run_receiver_loss_control import supervised_losses
from time_reconstruction import cubic_time_reconstruction


def pair_metrics(predictions,samples,targets,bg,scale,trace_scale):
    records=[];traces=[];totals={k:0. for k in ['field_error','scatter_field_true','full_field_true','trace_error','scatter_trace_true','full_trace_true']}
    for prediction,s,target in zip(predictions,samples,targets):
        ids=s['receiver_indices'].to(prediction.device)
        lf,lr=supervised_losses(prediction[None,None],target[None,None],ids,scale,trace_scale)
        fulltimes=torch.arange(1,2001,device=prediction.device,dtype=torch.float64)*30e-9
        trace=cubic_time_reconstruction(prediction[ids[:,0],ids[:,1],:]*scale,s['time_seconds'].to(prediction.device),fulltimes)
        true_trace=s['traces_full'].to(prediction.device)-bg['traces_full'][0].to(prediction.device);traces.append(trace)
        vals={'field_error':float((prediction-target).double().square().sum()),'scatter_field_true':float(target.double().square().sum()),
            'full_field_true':float(s['target'][0].double().square().sum()),'trace_error':float((trace-true_trace).double().square().sum()),
            'scatter_trace_true':float(true_trace.double().square().sum()),'full_trace_true':float(s['traces_full'].double().square().sum())}
        for k,v in vals.items():totals[k]+=v
        records.append({'group_id':s['group_id'],'shot':0,'field_mse':float(lf),'receiver_mse':float(lr),'total_objective':float(lf+lr),**vals})
    actual=traces[0]-traces[1];reference=(samples[0]['traces_full']-samples[1]['traces_full']).to(actual.device)
    field_response=predictions[0]-predictions[1];field_reference=targets[0]-targets[1]
    def response_metrics(pred,true):
        pred=pred.double();true=true.double();p=float(pred.square().sum());t=float(true.square().sum());dot=float((pred*true).sum());err=float((pred-true).square().sum())
        return {'relative_l2':np.sqrt(err/t),'norm_ratio':np.sqrt(p/t),'cosine':dot/max(np.sqrt(p*t),1e-30),'energy_sums':{'error':err,'predicted':p,'reference':t,'dot':dot}}
    result={'mean_field_mse':np.mean([x['field_mse'] for x in records]),'mean_receiver_mse':np.mean([x['receiver_mse'] for x in records]),
        'mean_total_objective':np.mean([x['total_objective'] for x in records]),
        'scatter_field_relative_l2':np.sqrt(totals['field_error']/totals['scatter_field_true']),
        'scatter_trace_relative_l2':np.sqrt(totals['trace_error']/totals['scatter_trace_true']),
        'full_field_relative_l2':np.sqrt(totals['field_error']/totals['full_field_true']),
        'full_trace_relative_l2':np.sqrt(totals['trace_error']/totals['full_trace_true']),
        'pair_trace_response':response_metrics(actual,reference),'pair_field_response':response_metrics(field_response,field_reference),
        'energy_sums':totals,'records':records}
    assert all(np.isfinite(result[k]) for k in ['mean_total_objective','scatter_field_relative_l2','scatter_trace_relative_l2'])
    return result,torch.stack(traces).cpu()


def load_pair(data,device):
    ds=OperatorPilotDataset(data,'train');samples=[ds[0],ds[4]]
    assert [s['group_id'] for s in samples]==['synthetic_workpiece_000','synthetic_workpiece_001'] and all(s['shot']==0 and s['split']=='train' for s in samples)
    for key in ['x_m','y_m','time_seconds','source_xy_m','receiver_indices']:assert torch.equal(samples[0][key],samples[1][key])
    bg=torch.load(Path(data)/'homogeneous.pt',weights_only=True,map_location='cpu')
    targets=[s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale for s in samples]
    real=[encode_time_features(s,mode='zero')[None].to(device) for s in samples]
    constant=encode_time_features(samples[0],torch.ones_like(samples[0]['gamma']),mode='zero')[None].to(device)
    assert torch.equal(real[0][:,1:],real[1][:,1:]) and not torch.equal(real[0][:,0],real[1][:,0])
    return ds,samples,bg,targets,[real,[constant,constant]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    plan=json.loads(a.plan.read_text());assert plan['status']=='planned_not_run' and plan['updates_per_arm']==800
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(a.plan.parent/'next_pair_feasibility.json')==plan['feasibility_sha256']
    device='cuda' if torch.cuda.is_available() else 'cpu';ds,samples,bg,targets,inputs=load_pair(a.data,device)
    models=[];optimizers=[];hashes=[]
    for arm in range(2):
        torch.manual_seed(2);model=BasicFNO(input_channels=8).to(device);models.append(model);hashes.append(state_hash(model));optimizers.append(torch.optim.Adam(model.parameters(),lr=.001))
    assert hashes[0]==hashes[1]
    mean=(targets[0]+targets[1])/2
    with torch.no_grad():
        oracle,oracle_traces=pair_metrics([mean,mean],samples,targets,bg,ds.scale,ds.manifest['normalization']['trace_rms'])
        zero,_=pair_metrics([torch.zeros_like(mean)]*2,samples,targets,bg,ds.scale,ds.manifest['normalization']['trace_rms'])
    assert abs(oracle['mean_total_objective']-plan['training_mean_oracle']['training_mean_oracle']['mean_total_objective'])<1e-7
    protocol={**plan,'status':'fixed_budget_protocol','initial_state_hashes':hashes,'model_config':models[0].config,
        'normalization_scales':ds.manifest['normalization'],'data_manifest_sha256':sha(a.data/'manifest.json'),
        'homogeneous_sha256':sha(a.data/'homogeneous.pt'),'plan_sha256':sha(a.plan),'script_sha256':sha(Path(__file__)),
        'evaluation_updates':[0,100,200,400,800],'validation_unloaded':True,'test_unloaded':True,'sample_order':[i%2 for i in range(800)]}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    history=[];start=time.perf_counter();final_predictions=[];final_traces=[]
    for update in range(801):
        if update in protocol['evaluation_updates']:
            entries=[]
            for arm,model in enumerate(models):
                model.eval()
                with torch.no_grad():
                    predictions=[model(x)[0,0] for x in inputs[arm]]
                    metrics,traces=pair_metrics(predictions,samples,targets,bg,ds.scale,ds.manifest['normalization']['trace_rms'])
                metrics['objective_over_mean_oracle']=metrics['mean_total_objective']/oracle['mean_total_objective'];entries.append(metrics)
                if arm==1:assert torch.equal(predictions[0],predictions[1]) and metrics['pair_trace_response']['norm_ratio']==0.
                if update==800:final_predictions.append(torch.stack(predictions).cpu());final_traces.append(traces)
            history.append({'update':update,'arms':entries});(a.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps({'update':update,'arms':[{'objective':x['mean_total_objective'],'over_mean':x['objective_over_mean_oracle'],'scatter_trace_l2':x['scatter_trace_relative_l2'],'pair_response':x['pair_trace_response']['relative_l2'],'pair_response_norm':x['pair_trace_response']['norm_ratio']} for x in entries]}),flush=True)
        if update==800:break
        index=update%2
        for arm,(model,optimizer) in enumerate(zip(models,optimizers)):
            model.train();optimizer.zero_grad(set_to_none=True);pred=model(inputs[arm][index]);lf,lr=supervised_losses(pred,targets[index][None,None],samples[index]['receiver_indices'].to(device),ds.scale,ds.manifest['normalization']['trace_rms']);loss=lf+lr
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();del pred,lf,lr,loss
        if (update+1)%50==0:print(f'Paired update {update+1}/800, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    for arm,(model,optimizer) in enumerate(zip(models,optimizers)):
        torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'config':model.config,'updates':800,'scale':ds.scale,'arm':plan['arms'][arm],'receiver_loss_weight':1.,'target':'defect_minus_homogeneous','data_manifest_sha256':protocol['data_manifest_sha256']},a.out/f'final_arm{arm}.pt')
    torch.save({'prediction_residual_fields':final_predictions,'prediction_scatter_traces':final_traces,'reference_residual_fields':torch.stack([x.cpu() for x in targets]),'reference_scatter_traces':torch.stack([s['traces_full']-bg['traces_full'][0] for s in samples]),'mean_oracle_traces':oracle_traces,'group_ids':[s['group_id'] for s in samples]},a.out/'example.pt')
    report={'status':'completed','protocol':protocol,'mean_oracle':oracle,'zero_scatter':zero,'history':history,'final':history[-1]['arms'],'elapsed_seconds':time.perf_counter()-start}
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Paired material conditioning trial complete',flush=True)


if __name__=='__main__':main()

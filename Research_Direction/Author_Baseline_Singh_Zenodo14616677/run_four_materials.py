"""Fixed-source four-material FNO training with frozen training-mean baseline."""
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

REGIONS={'opposite_boundary':slice(0,2),'interior':slice(2,126),'array_side':slice(126,128)}


def load_samples(data):
    train=OperatorPilotDataset(data,'train');val=OperatorPilotDataset(data,'validation')
    fitted=[train[i] for i in [0,4,8,12]];held=[val[i] for i in range(0,len(val),4)]
    assert [s['group_id'] for s in fitted]==[f'synthetic_workpiece_{i:03d}' for i in range(4)] and len(held)==6
    for s in fitted+held:
        assert s['shot']==0
        for key in ['x_m','y_m','time_seconds','source_xy_m','receiver_indices']:assert torch.equal(fitted[0][key],s[key])
    return train,fitted,held


def evaluate(model,samples,bg,scale,trace_scale,device,fixed=None,inputs=None,save_example=False):
    totals={k:0. for k in ['field_error','field_true','scatter_field_true','trace_error','trace_true','scatter_trace_true']};records=[];regions={k:{'error':0.,'reference':0.} for k in REGIONS};losses=[];example=None
    with torch.no_grad():
        for i,s in enumerate(samples):
            x=None
            if fixed is None:
                x=inputs[i] if inputs is not None else encode_time_features(s,mode='zero')[None].to(device)
                pred=model(x)[0,0]
            else:pred=fixed
            truth=s['target'][0].to(device);target=truth-bg['wavefield_snapshots'][0].to(device)/scale;ids=s['receiver_indices'].to(device)
            lf,lr=supervised_losses(pred[None,None],target[None,None],ids,scale,trace_scale);losses.append((float(lf),float(lr)))
            times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
            traces=cubic_time_reconstruction(pred[ids[:,0],ids[:,1],:]*scale,s['time_seconds'].to(device),times)
            true_trace=s['traces_full'].to(device)-bg['traces_full'][0].to(device)
            vals={'field_error':float((pred-target).square().sum(dtype=torch.float64)),'field_true':float(truth.square().sum(dtype=torch.float64)),
                'scatter_field_true':float(target.square().sum(dtype=torch.float64)),'trace_error':float((traces-true_trace).square().sum(dtype=torch.float64)),
                'trace_true':float(s['traces_full'].square().sum(dtype=torch.float64)),'scatter_trace_true':float(true_trace.square().sum(dtype=torch.float64))}
            for k,v in vals.items():totals[k]+=v
            for name,ys in REGIONS.items():
                regions[name]['error']+=float((pred[:,ys,:]-target[:,ys,:]).square().sum(dtype=torch.float64));regions[name]['reference']+=float(target[:,ys,:].square().sum(dtype=torch.float64))
            records.append({'group_id':s['group_id'],'split':s['split'],'shot':0,'field_mse':float(lf),'receiver_mse':float(lr),**vals})
            if save_example and i==0:example={'group_id':s['group_id'],'prediction_residual_field':pred.cpu(),'reference_residual_field':target.cpu(),'prediction_scatter_traces':traces.cpu(),'reference_scatter_traces':true_trace.cpu()}
            del pred,truth,target,traces,true_trace,x
    result={'split':samples[0]['split'],'groups':len(samples),'shots':len(samples),'mean_field_mse':float(np.mean([x[0] for x in losses])),
        'mean_receiver_mse':float(np.mean([x[1] for x in losses])),'mean_total_objective':float(np.mean([sum(x) for x in losses])),
        'full_field_relative_l2':np.sqrt(totals['field_error']/totals['field_true']),'full_trace_relative_l2':np.sqrt(totals['trace_error']/totals['trace_true']),
        'scatter_field_relative_l2':np.sqrt(totals['field_error']/totals['scatter_field_true']),'scatter_trace_relative_l2':np.sqrt(totals['trace_error']/totals['scatter_trace_true']),
        'regions':{name:{'scatter_field_relative_l2':np.sqrt(v['error']/v['reference']),'energy_sums':v} for name,v in regions.items()},'energy_sums':totals,'records':records}
    assert all(np.isfinite(result[k]) for k in ['mean_total_objective','scatter_field_relative_l2','scatter_trace_relative_l2'])
    return result,example


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists();plan=json.loads(a.plan.read_text());assert plan['status']=='planned_not_run' and plan['updates']==1600
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();device='cuda' if torch.cuda.is_available() else 'cpu'
    ds,train,val=load_samples(a.data);bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu');sr=ds.manifest['normalization']['trace_rms']
    targets=[s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale for s in train]
    mean=torch.stack(targets).mean(0);inputs=[encode_time_features(s,mode='zero')[None].to(device) for s in train]
    torch.manual_seed(2);model=BasicFNO(input_channels=8).to(device);initial_hash=state_hash(model)
    old=json.loads((a.plan.parent/'comparison.json').read_text());assert initial_hash==old['protocol']['initial_state_hashes'][0]
    assert sha(a.data/'manifest.json')==old['protocol']['data_manifest_sha256']
    protocol={**plan,'status':'fixed_budget_protocol','initial_state_hash':initial_hash,'normalization_scales':ds.manifest['normalization'],
        'data_manifest_sha256':sha(a.data/'manifest.json'),'homogeneous_sha256':sha(a.data/'homogeneous.pt'),'plan_sha256':sha(a.plan),
        'script_sha256':sha(Path(__file__)),'model_config':model.config,'sample_order':[i%4 for i in range(1600)],'evaluation_updates':[0,400,800,1600],
        'training_groups':[s['group_id'] for s in train],'validation_groups':[s['group_id'] for s in val],'test_unloaded':True}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');torch.save({'normalized_residual':mean.cpu(),'training_groups':protocol['training_groups'],'scale':ds.scale,'data_manifest_sha256':protocol['data_manifest_sha256']},a.out/'training_mean.pt')
    references={}
    for name,fixed in [('zero',torch.zeros_like(mean)),('training_mean',mean)]:
        references[name]={}
        for split,samples in [('train',train),('validation',val)]:references[name][split],_=evaluate(None,samples,bg,ds.scale,sr,device,fixed=fixed)
    (a.out/'references.json').write_text(json.dumps(references,indent=2),encoding='utf-8')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[];start=time.perf_counter()
    for update in range(1601):
        if update in protocol['evaluation_updates']:
            model.eval();training,_=evaluate(model,train,bg,ds.scale,sr,device,inputs=inputs);validation,example=evaluate(model,val,bg,ds.scale,sr,device,save_example=update==1600)
            history.append({'update':update,'training':training,'validation':validation});(a.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps({'update':update,'train_objective':training['mean_total_objective'],'train_mean_oracle':references['training_mean']['train']['mean_total_objective'],'train_scatter_trace':training['scatter_trace_relative_l2'],'validation_scatter_trace':validation['scatter_trace_relative_l2'],'validation_scatter_field':validation['scatter_field_relative_l2']}),flush=True)
        if update==1600:break
        index=update%4;model.train();optimizer.zero_grad(set_to_none=True);pred=model(inputs[index]);lf,lr=supervised_losses(pred,targets[index][None,None],train[index]['receiver_indices'].to(device),ds.scale,sr);loss=lf+lr
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();del pred,lf,lr,loss
        if (update+1)%100==0:print(f'Update {update+1}/1600, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'config':model.config,'updates':1600,'scale':ds.scale,'time_feature_mode':'zero','target':'defect_minus_homogeneous','receiver_loss_weight':1.,'data_manifest_sha256':protocol['data_manifest_sha256']},a.out/'final.pt');torch.save(example,a.out/'validation_example.pt')
    report={'status':'completed','protocol':protocol,'references':references,'history':history,'final_training':training,'final_validation':validation,'elapsed_seconds':time.perf_counter()-start}
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Four-material fixed-source trial complete',flush=True)


if __name__=='__main__':main()

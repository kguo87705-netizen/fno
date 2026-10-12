"""Pre-registered paired 20-epoch ordinary FNO training; train/validation only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input
from run_receiver_loss_control import supervised_losses
from time_reconstruction import cubic_time_reconstruction


def state_hash(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def evaluate_pair(models,dataset,background,device,save_example=False):
    sums=[{k:0. for k in ['field_error','field_true','field_background_error','trace_error','trace_true','scatter_true','trace_background_error']} for _ in models]
    records=[[] for _ in models];examples=[]
    for index in range(len(dataset)):
        sample=dataset[index];inputs=encode_fno_input(sample)[None].to(device);truth=sample['target'][0].to(device)
        ids=sample['receiver_indices'].to(device);observed=sample['traces_full'].to(device)
        bg=background['traces_full'][sample['shot']].to(device)
        bgfield=background['wavefield_snapshots'][sample['shot']].to(device)/dataset.scale
        times=torch.arange(1,2001,device=device,dtype=torch.float64)*dataset.manifest['dt_seconds']
        for arm,model in enumerate(models):
            with torch.no_grad():
                field=model(inputs)[0,0]
                sparse=field[ids[:,0],ids[:,1],:]*dataset.scale
                traces=cubic_time_reconstruction(sparse,sample['time_seconds'].to(device),times)
                values={'field_error':float((field-truth).square().sum(dtype=torch.float64)),
                    'field_true':float(truth.square().sum(dtype=torch.float64)),
                    'field_background_error':float((bgfield-truth).square().sum(dtype=torch.float64)),
                    'trace_error':float((traces-observed).square().sum(dtype=torch.float64)),
                    'trace_true':float(observed.square().sum(dtype=torch.float64)),
                    'scatter_true':float((observed-bg).square().sum(dtype=torch.float64)),
                    'trace_background_error':float((observed-bg).square().sum(dtype=torch.float64))}
                assert all(np.isfinite(v) for v in values.values())
                for k,v in values.items():sums[arm][k]+=v
                records[arm].append({'group_id':sample['group_id'],'shot':sample['shot'],**values,
                    'field_relative_l2':np.sqrt(values['field_error']/values['field_true']),
                    'trace_relative_l2':np.sqrt(values['trace_error']/values['trace_true'])})
                if index==0 and save_example:
                    examples.append({'group_id':sample['group_id'],'shot':sample['shot'],'prediction_field':field.cpu(),
                        'truth_field':truth.cpu(),'prediction_traces':traces.cpu(),'reference_traces':sample['traces_full'],
                        'background_traces':background['traces_full'][sample['shot']]})
                del field,sparse,traces
        del inputs,truth,ids,observed,bg,bgfield
    reports=[]
    for total,recs in zip(sums,records):
        reports.append({'split':dataset.entries[0][0]['split'],'groups':len(dataset)//4,'shots':len(dataset),
            'field_relative_l2':np.sqrt(total['field_error']/total['field_true']),
            'trace_relative_l2':np.sqrt(total['trace_error']/total['trace_true']),
            'homogeneous_field_relative_l2':np.sqrt(total['field_background_error']/total['field_true']),
            'homogeneous_trace_relative_l2':np.sqrt(total['trace_background_error']/total['trace_true']),
            'scatter_normalized_trace_error':np.sqrt(total['trace_error']/total['scatter_true']),
            'energy_sums':total,'records':recs})
    return reports,examples


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists(),'Use new output directory'
    train=OperatorPilotDataset(args.data,'train');validation=OperatorPilotDataset(args.data,'validation')
    assert len(train)==48 and len(validation)==24
    assert {r['group_id'] for r,_ in train.entries}.isdisjoint({r['group_id'] for r,_ in validation.entries})
    manifest_hash=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    protocol={'epochs':20,'updates_per_arm':960,'seed':2,'sample_order_seed':2,'learning_rate':.001,'batch_size':1,
        'receiver_loss_weights':[0.,1.],'primary_comparison':'fixed final epoch 20; no validation checkpoint selection',
        'train_groups':12,'validation_groups':6,'dataset_access':['train','validation'],'test_not_loaded':True,
        'normalization':train.manifest['normalization'],'receiver_training_frames':500,'trace_evaluation_steps':2000,
        'data_manifest_sha256':manifest_hash,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Synthetic scalar material model, fixed source and array, one seed. No gradients or FWI evaluation.'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');models=[]
    for _ in range(2):torch.manual_seed(2);models.append(BasicFNO().to(device))
    initial_hashes=[state_hash(m) for m in models];assert initial_hashes[0]==initial_hashes[1]
    optimizers=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
    background=torch.load(args.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    rng=np.random.default_rng(2);history=[[],[]];orders=[];start=time.perf_counter()
    for epoch in range(1,21):
        order=[int(4*g+s) for g in rng.permutation(12) for s in rng.permutation(4)];orders.append(order)
        assert sorted(order)==list(range(48))
        losses=[[],[]]
        for model in models:model.train()
        for step,index in enumerate(order,1):
            sample=train[index];inputs=encode_fno_input(sample)[None].to(device);target=sample['target'][None].to(device);ids=sample['receiver_indices'].to(device)
            for arm,(model,optimizer,weight) in enumerate(zip(models,optimizers,[0.,1.])):
                optimizer.zero_grad(set_to_none=True);pred=model(inputs)
                lf,lr=supervised_losses(pred,target,ids,train.scale,train.manifest['normalization']['trace_rms'])
                loss=lf if weight==0 else lf+lr
                assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step();losses[arm].append((float(lf.detach()),float(lr.detach())))
                del pred,lf,lr,loss
            del inputs,target,ids
            if step==24:print(f'Epoch {epoch}/20, paired step {step}/48, elapsed {time.perf_counter()-start:.1f}s',flush=True)
        for model in models:model.eval()
        val,_=evaluate_pair(models,validation,background,device)
        for arm in range(2):
            row={'epoch':epoch,'updates':epoch*48,'training_field_mse_step_mean':float(np.mean([p[0] for p in losses[arm]])),
                'training_receiver_mse_step_mean':float(np.mean([p[1] for p in losses[arm]])),
                'validation':{k:v for k,v in val[arm].items() if k!='records'},'elapsed_seconds':time.perf_counter()-start}
            history[arm].append(row)
            (args.out/f'history_weight{arm}.json').write_text(json.dumps(history[arm],indent=2),encoding='utf-8')
            print(json.dumps({'weight':arm,'epoch':epoch,'val_field':val[arm]['field_relative_l2'],'val_trace':val[arm]['trace_relative_l2'],'val_scatter':val[arm]['scatter_normalized_trace_error']}),flush=True)
            torch.save({'state_dict':models[arm].state_dict(),'config':models[arm].config,'epoch':epoch,'updates':epoch*48,
                'scale':train.scale,'receiver_loss_weight':float(arm),'data_manifest_sha256':manifest_hash},args.out/f'latest_weight{arm}.pt')
    validation_reports,examples=evaluate_pair(models,validation,background,device,True)
    training_reports,_=evaluate_pair(models,train,background,device)
    for arm in range(2):
        torch.save(examples[arm],args.out/f'validation_example_weight{arm}.pt')
        torch.save({'state_dict':models[arm].state_dict(),'config':models[arm].config,'epoch':20,'updates':960,
            'scale':train.scale,'receiver_loss_weight':float(arm),'data_manifest_sha256':manifest_hash},args.out/f'final_weight{arm}.pt')
    report={'status':'completed','protocol':protocol,'initial_state_hashes':initial_hashes,'sample_orders':orders,
        'final_training':training_reports,'final_validation':validation_reports,'histories':history,'elapsed_seconds':time.perf_counter()-start,
        'caution':'Validation is used for research decisions, not a final independent test. One seed and 20 epochs do not establish convergence.'}
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Both fixed-budget arms complete',flush=True)


if __name__=='__main__':main()

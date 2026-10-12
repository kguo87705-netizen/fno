"""Fixed 20-epoch scattering-target arm, matched to saved direct-target control."""
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
from run_scatter_target_pilot import evaluate_scatter
from train_multiworkpiece_control import state_hash
from time_reconstruction import cubic_time_reconstruction


def evaluate_dataset(model,ds,bg,device,encoder=encode_fno_input):
    sums={k:0. for k in ['field_error','field_true','scatter_field_true','trace_error','trace_true','scatter_trace_true']};records=[]
    with torch.no_grad():
        for i in range(len(ds)):
            sample=ds[i];shot=sample['shot'];inputs=encoder(sample)[None].to(device)
            residual=model(inputs)[0,0];truth=sample['target'][0].to(device)
            bgfield=bg['wavefield_snapshots'][shot].to(device)/ds.scale;delta=truth-bgfield
            ids=sample['receiver_indices'].to(device);times=torch.arange(1,2001,device=device,dtype=torch.float64)*ds.manifest['dt_seconds']
            traces=cubic_time_reconstruction(residual[ids[:,0],ids[:,1],:]*ds.scale,sample['time_seconds'].to(device),times)
            observed=sample['traces_full'].to(device);delta_trace=observed-bg['traces_full'][shot].to(device)
            values={'field_error':float((residual-delta).square().sum(dtype=torch.float64)),
                'field_true':float(truth.square().sum(dtype=torch.float64)),
                'scatter_field_true':float(delta.square().sum(dtype=torch.float64)),
                'trace_error':float((traces-delta_trace).square().sum(dtype=torch.float64)),
                'trace_true':float(observed.square().sum(dtype=torch.float64)),
                'scatter_trace_true':float(delta_trace.square().sum(dtype=torch.float64))}
            assert all(np.isfinite(x) for x in values.values())
            for k,value in values.items():sums[k]+=value
            records.append({'group_id':sample['group_id'],'shot':shot,**values})
            del inputs,residual,truth,bgfield,delta,traces,observed,delta_trace
    return {'split':ds.entries[0][0]['split'],'groups':len(ds)//4,'shots':len(ds),
        'full_field_relative_l2':np.sqrt(sums['field_error']/sums['field_true']),
        'full_trace_relative_l2':np.sqrt(sums['trace_error']/sums['trace_true']),
        'scatter_field_relative_l2':np.sqrt(sums['field_error']/sums['scatter_field_true']),
        'scatter_trace_relative_l2':np.sqrt(sums['trace_error']/sums['scatter_trace_true']),
        'homogeneous_field_relative_l2':np.sqrt(sums['scatter_field_true']/sums['field_true']),
        'homogeneous_trace_relative_l2':np.sqrt(sums['scatter_trace_true']/sums['trace_true']),
        'energy_sums':sums,'records':records}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--control',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists(),'Use new output directory'
    old=json.loads((args.control/'comparison.json').read_text());manifest_hash=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    assert old['protocol']['data_manifest_sha256']==manifest_hash
    train=OperatorPilotDataset(args.data,'train');val=OperatorPilotDataset(args.data,'validation')
    assert len(train)==48 and len(val)==24
    protocol={'epochs':20,'updates':960,'seed':2,'learning_rate':.001,'receiver_loss_weight':1.,
        'target':'defect_minus_homogeneous','scales':train.manifest['normalization'],'normalization_changed':False,
        'primary_comparison':'final epoch20, no checkpoint selection','data_access':['train','validation'],'test_unloaded':True,
        'control':str(args.control),'data_manifest_sha256':manifest_hash,
        'control_checkpoint_sha256':hashlib.sha256((args.control/'final_weight1.pt').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'basis':'fixed homogeneous FD snapshots and full traces; material-independent',
        'scope':'One seed synthetic scalar model; no physical-gradient or inverse evaluation.'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=BasicFNO().to(device)
    initial_hash=state_hash(model);assert initial_hash==old['initial_state_hashes'][1]
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);bg=torch.load(args.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    history=[];start=time.perf_counter()
    for epoch,order in enumerate(old['sample_orders'],1):
        assert sorted(order)==list(range(48));model.train();losses=[]
        for step,index in enumerate(order,1):
            sample=train[index];inputs=encode_fno_input(sample)[None].to(device)
            target=(sample['target']-bg['wavefield_snapshots'][sample['shot']:sample['shot']+1]/train.scale)[None].to(device)
            ids=sample['receiver_indices'].to(device);optimizer.zero_grad(set_to_none=True);prediction=model(inputs)
            lf,lr=supervised_losses(prediction,target,ids,train.scale,train.manifest['normalization']['trace_rms']);loss=lf+lr
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();losses.append((float(lf.detach()),float(lr.detach())));del inputs,target,prediction,lf,lr,loss
            if step==24:print(f'Epoch {epoch}/20, step24/48, elapsed {time.perf_counter()-start:.1f}s',flush=True)
        model.eval();validation=evaluate_dataset(model,val,bg,device)
        history.append({'epoch':epoch,'updates':epoch*48,'training_field_mse_step_mean':float(np.mean([x[0] for x in losses])),
            'training_receiver_mse_step_mean':float(np.mean([x[1] for x in losses])),
            'validation':{k:v for k,v in validation.items() if k!='records'}})
        (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        checkpoint={'state_dict':model.state_dict(),'config':model.config,'epoch':epoch,'updates':epoch*48,'scale':train.scale,
            'receiver_loss_weight':1.,'target':'defect_minus_homogeneous','data_manifest_sha256':manifest_hash}
        torch.save(checkpoint,args.out/'latest_weight1.pt')
        print(json.dumps({'epoch':epoch,**{k:validation[k] for k in ['full_field_relative_l2','full_trace_relative_l2','scatter_field_relative_l2','scatter_trace_relative_l2']}}),flush=True)
    torch.save(checkpoint,args.out/'final_weight1.pt');training=evaluate_dataset(model,train,bg,device)
    sample=val[0];_,example=evaluate_scatter(model,encode_fno_input(sample)[None].to(device),sample['target'][None].to(device),sample,bg)
    torch.save(example,args.out/'validation_example.pt')
    report={'status':'completed','protocol':protocol,'initial_weights_match_control':True,'initial_hash':initial_hash,
        'sample_orders':old['sample_orders'],'history':history,'final_training':training,'final_validation':validation,
        'direct_control_training':old['final_training'][1],'direct_control_validation':old['final_validation'][1],
        'elapsed_seconds':time.perf_counter()-start}
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Multiworkpiece scatter-target training complete',flush=True)


if __name__=='__main__':main()

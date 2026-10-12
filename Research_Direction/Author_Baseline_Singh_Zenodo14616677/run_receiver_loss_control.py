"""Training-only single-shot control: field MSE plus fixed receiver MSE."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input
from diagnose_basic_fno import evaluate


def supervised_losses(prediction,target,indices,field_scale,trace_scale):
    field=(prediction-target).square().mean()
    difference=prediction[0,0,indices[:,0],indices[:,1],:]-target[0,0,indices[:,0],indices[:,1],:]
    receiver=(difference*(field_scale/trace_scale)).square().mean()
    return field,receiver


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'protocol.json').exists(),'Use new output directory'
    train=OperatorPilotDataset(args.data,'train');sample=train[0]
    control=torch.load(args.control/'final.pt',weights_only=True,map_location='cpu')
    manifest_hash=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    assert control['data_manifest_sha256']==manifest_hash and control['group_id']==sample['group_id'] and control['shot']==sample['shot']==0
    protocol={'updates':400,'evaluation_updates':[0,50,100,200,400],'seed':2,'learning_rate':.001,
        'receiver_loss_weight':1.0,'loss':'mean((field error / train wavefield RMS)^2) + mean((sparse receiver error / train trace RMS)^2)',
        'receiver_training_frames':500,'full_trace_evaluation_steps':2000,
        'receiver_target':'same saved sparse wavefield at 24 physical receiver indices',
        'selection':'final fixed update, no selection','sample_group':sample['group_id'],'shot':0,
        'data_access':['train'],'test_unloaded':True,'control_directory':str(args.control),
        'data_manifest_sha256':manifest_hash,'config':control['config'],
        'caution':'One fitted training shot only; not operator generalization, gradient accuracy or inverse reconstruction.'}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=BasicFNO().to(device);assert model.config==control['config']
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    inputs=encode_fno_input(sample).unsqueeze(0).to(device);target=sample['target'].unsqueeze(0).to(device)
    ids=sample['receiver_indices'].to(device);trace_scale=train.manifest['normalization']['trace_rms']
    history=[];start=time.perf_counter()
    old_history=json.loads((args.control/'history.json').read_text(encoding='utf-8'))
    for update in range(401):
        if update in protocol['evaluation_updates']:
            model.eval();metrics,traces,field=evaluate(model,inputs,target,sample)
            if update==0:
                assert all(abs(metrics[k]-old_history[0][k])<1e-6 for k in metrics),'Initial predictions differ from control'
            with torch.no_grad():
                lf,lr=supervised_losses(field[None,None].to(device),target,ids,train.scale,trace_scale)
            history.append({'update':update,**metrics,'receiver_normalized_mse':float(lr),'total_training_loss':float(lf+lr)})
            (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps(history[-1]),flush=True)
        if update==400:break
        model.train();optimizer.zero_grad(set_to_none=True);prediction=model(inputs)
        lf,lr=supervised_losses(prediction,target,ids,train.scale,trace_scale);loss=lf+lr
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();del prediction,loss,lf,lr
        if (update+1)%25==0:print(f'Update {update+1}/400, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    torch.save({'state_dict':model.state_dict(),'config':model.config,'update':400,'group_id':sample['group_id'],'shot':0,
        'scale':train.scale,'data_manifest_sha256':manifest_hash,'receiver_loss_weight':1.0},args.out/'final.pt')
    torch.save({'prediction_field':field,'prediction_traces':traces,'reference_traces':sample['traces_full']},args.out/'example.pt')
    report={'protocol':protocol,'initial_predictions_match_control':True,'history':history,'elapsed_seconds':time.perf_counter()-start,
        'control_final_metrics':old_history[-1],'field_error_ratio_to_control':history[-1]['field_relative_l2']/old_history[-1]['field_relative_l2'],
        'trace_error_ratio_to_control':history[-1]['trace_relative_l2']/old_history[-1]['trace_relative_l2']}
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

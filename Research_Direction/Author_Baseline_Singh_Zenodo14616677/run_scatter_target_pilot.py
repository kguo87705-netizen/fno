"""Fixed 400-update training-shot pilot: ordinary FNO predicts defect scattering."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input
from run_receiver_loss_control import supervised_losses
from diagnose_basic_fno import evaluate as evaluate_direct
from time_reconstruction import cubic_time_reconstruction


def evaluate_scatter(model,inputs,full_target,sample,background):
    with torch.no_grad():
        residual=model(inputs)[0,0]
        bgfield=background['wavefield_snapshots'][sample['shot']].to(residual.device)/sample['wavefield_scale']
        target=full_target[0,0]-bgfield
        indices=sample['receiver_indices'].to(residual.device)
        times=torch.arange(1,2001,device=residual.device,dtype=torch.float64)*30e-9
        sparse=residual[indices[:,0],indices[:,1],:]*sample['wavefield_scale']
        response=cubic_time_reconstruction(sparse,sample['time_seconds'].to(residual.device),times)
        bgtrace=background['traces_full'][sample['shot']].to(residual.device)
        reference=sample['traces_full'].to(residual.device);true_response=reference-bgtrace
        field=bgfield+residual;traces=bgtrace+response
        metrics={'scatter_field_relative_l2':float((residual-target).norm()/target.norm()),
            'scatter_trace_relative_l2':float((response-true_response).norm()/true_response.norm()),
            'full_field_relative_l2':float((field-full_target[0,0]).norm()/full_target.norm()),
            'full_trace_relative_l2':float((traces-reference).norm()/reference.norm()),
            'homogeneous_field_relative_l2':float(target.norm()/full_target.norm()),
            'homogeneous_trace_relative_l2':float(true_response.norm()/reference.norm()),
            'full_trace_norm_ratio':float(traces.norm()/reference.norm()),
            'full_trace_zero_lag_correlation':float((traces*reference).sum()/(traces.norm()*reference.norm())),
            'scatter_field_normalized_mse':float((residual-target).square().mean())}
    return metrics,{'prediction_residual_field':residual.cpu(),'reference_residual_field':target.cpu(),
        'prediction_field':field.cpu(),'prediction_traces':traces.cpu(),'prediction_scatter_traces':response.cpu(),
        'reference_traces':sample['traces_full'],'reference_scatter_traces':true_response.cpu(),
        'homogeneous_traces':bgtrace.cpu()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert plan['status']=='planned_not_run' and sha(args.data/'manifest.json')==plan['data_manifest_sha256']
    control=Path(plan['baseline_control']);assert sha(control)==plan['baseline_control_sha256']
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists(),'Use new output directory'
    ds=OperatorPilotDataset(args.data,'train');sample=ds[0]
    assert sample['group_id']==plan['group_id'] and sample['shot']==plan['shot']==0 and sample['split']=='train'
    assert ds.manifest['normalization']==plan['normalization']
    protocol={**plan,'status':'fixed_budget_protocol','evaluation_updates':[0,50,100,200,400],
        'plan_sha256':sha(args.plan),'script_sha256':sha(Path(__file__)),'homogeneous_sha256':sha(args.data/'homogeneous.pt'),
        'initial_weights':'same seed2 BasicFNO as direct target control; raw initial metrics checked',
        'data_access':['train'],'test_not_loaded':True,'normalization_changed':False}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    torch.manual_seed(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=BasicFNO().to(device)
    saved_control=torch.load(control,weights_only=True,map_location='cpu');assert model.config==saved_control['config']
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    inputs=encode_fno_input(sample)[None].to(device);full_target=sample['target'][None].to(device)
    background=torch.load(args.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    target=full_target-background['wavefield_snapshots'][0][None,None].to(device)/ds.scale
    indices=sample['receiver_indices'].to(device);trace_scale=ds.manifest['normalization']['trace_rms']
    old=json.loads((control.parent/'history.json').read_text(encoding='utf-8'))
    raw_initial,_,_=evaluate_direct(model,inputs,full_target,sample)
    assert all(abs(raw_initial[k]-old[0][k])<1e-6 for k in raw_initial)
    history=[];start=time.perf_counter()
    for update in range(401):
        if update in protocol['evaluation_updates']:
            model.eval();metrics,example=evaluate_scatter(model,inputs,full_target,sample,background)
            with torch.no_grad():
                lf,lr=supervised_losses(example['prediction_residual_field'][None,None].to(device),target,indices,ds.scale,trace_scale)
            history.append({'update':update,**metrics,'receiver_normalized_mse':float(lr),'total_training_loss':float(lf+lr)})
            (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps(history[-1]),flush=True)
        if update==400:break
        model.train();optimizer.zero_grad(set_to_none=True);prediction=model(inputs)
        lf,lr=supervised_losses(prediction,target,indices,ds.scale,trace_scale);loss=lf+lr
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();del prediction,lf,lr,loss
        if (update+1)%25==0:print(f'Update {update+1}/400, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    torch.save({'state_dict':model.state_dict(),'config':model.config,'update':400,'group_id':sample['group_id'],'shot':0,
        'scale':ds.scale,'receiver_loss_weight':1.,'target':'defect_minus_homogeneous','data_manifest_sha256':plan['data_manifest_sha256'],
        'homogeneous_sha256':protocol['homogeneous_sha256']},args.out/'final.pt')
    torch.save(example,args.out/'example.pt')
    report={'status':'completed','protocol':protocol,'initial_raw_predictions_match_control':True,'history':history,
        'direct_control_final':old[-1],'elapsed_seconds':time.perf_counter()-start,
        'scope':'One fitted training shot. No material-response, physical-gradient or held-out evaluation.'}
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Scattering-target pilot complete',flush=True)


if __name__=='__main__':main()

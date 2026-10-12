"""Fixed4800-step source0 trial over all twelve existing training materials."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset
from time_feature_input import encode_time_features
from train_multiworkpiece_control import state_hash
from run_receiver_loss_control import supervised_losses
from run_four_materials import evaluate
from fixed_source_inputs import FixedSourceInputs


def load_samples(data):
    train=OperatorPilotDataset(data,'train');val=OperatorPilotDataset(data,'validation')
    fitted=[train[i] for i in range(0,len(train),4)];held=[val[i] for i in range(0,len(val),4)]
    assert len(fitted)==12 and len(held)==6
    for s in fitted+held:
        assert s['shot']==0
        for key in ['x_m','y_m','time_seconds','source_xy_m','receiver_indices']:assert torch.equal(fitted[0][key],s[key])
    return train,fitted,held


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists();plan=json.loads(a.plan.read_text());assert plan['status']=='planned_not_run' and plan['updates']==4800
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();assert sha(a.data/'manifest.json')==plan['data_manifest_sha256']
    device='cuda' if torch.cuda.is_available() else 'cpu';ds,train,val=load_samples(a.data);assert [s['group_id'] for s in train]==plan['training_workpieces']
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu');sr=ds.manifest['normalization']['trace_rms']
    targets=[s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale for s in train]
    mean=torch.stack(targets).mean(0);inputs=FixedSourceInputs(train,device)
    for index in [0,11]:assert torch.equal(inputs[index],encode_time_features(train[index],mode='zero')[None].to(device))
    torch.manual_seed(2);model=BasicFNO(input_channels=8).to(device);initial_hash=state_hash(model)
    old=json.loads((a.plan.parent/'comparison.json').read_text());assert initial_hash==old['protocol']['initial_state_hash']
    protocol={**plan,'status':'fixed_budget_protocol','initial_state_hash':initial_hash,'normalization_scales':ds.manifest['normalization'],
        'homogeneous_sha256':sha(a.data/'homogeneous.pt'),'plan_sha256':sha(a.plan),'script_sha256':sha(Path(__file__)),
        'input_builder_sha256':sha(Path(__file__).parent/'fixed_source_inputs.py'),'streamed_input_parity_checks':'bit-identical to canonical encoder on train000 and019',
        'model_config':model.config,'sample_order':[i%12 for i in range(4800)],'training_groups':[s['group_id'] for s in train],
        'validation_groups':[s['group_id'] for s in val],'test_unloaded':True}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');torch.save({'normalized_residual':mean.cpu(),'training_groups':protocol['training_groups'],'scale':ds.scale,'data_manifest_sha256':protocol['data_manifest_sha256']},a.out/'training_mean.pt')
    references={}
    for name,fixed in [('zero',torch.zeros_like(mean)),('training_mean',mean)]:
        references[name]={}
        for split,samples in [('train',train),('validation',val)]:references[name][split],_=evaluate(None,samples,bg,ds.scale,sr,device,fixed=fixed)
    (a.out/'references.json').write_text(json.dumps(references,indent=2),encoding='utf-8')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);history=[];start=time.perf_counter()
    for update in range(4801):
        if update in protocol['evaluation_updates']:
            model.eval();training,_=evaluate(model,train,bg,ds.scale,sr,device,inputs=inputs);validation,example=evaluate(model,val,bg,ds.scale,sr,device,save_example=update==4800)
            history.append({'update':update,'training':training,'validation':validation});(a.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps({'update':update,'train_objective':training['mean_total_objective'],'train_mean_oracle':references['training_mean']['train']['mean_total_objective'],'train_scatter_trace':training['scatter_trace_relative_l2'],'validation_scatter_trace':validation['scatter_trace_relative_l2'],'validation_scatter_field':validation['scatter_field_relative_l2']}),flush=True)
            if update:
                torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'config':model.config,'updates':update,'scale':ds.scale,'time_feature_mode':'zero','target':'defect_minus_homogeneous','receiver_loss_weight':1.,'data_manifest_sha256':protocol['data_manifest_sha256']},a.out/'latest.pt')
        if update==4800:break
        index=update%12;model.train();optimizer.zero_grad(set_to_none=True);x=inputs[index];pred=model(x)
        lf,lr=supervised_losses(pred,targets[index][None,None],train[index]['receiver_indices'].to(device),ds.scale,sr);loss=lf+lr
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();del x,pred,lf,lr,loss
        if (update+1)%100==0:print(f'Update {update+1}/4800, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    checkpoint=torch.load(a.out/'latest.pt',weights_only=True,map_location='cpu');torch.save(checkpoint,a.out/'final.pt');torch.save(example,a.out/'validation_example.pt')
    report={'status':'completed','protocol':protocol,'references':references,'history':history,'final_training':training,'final_validation':validation,'elapsed_seconds':time.perf_counter()-start}
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Twelve-material fixed-source trial complete',flush=True)


if __name__=='__main__':main()

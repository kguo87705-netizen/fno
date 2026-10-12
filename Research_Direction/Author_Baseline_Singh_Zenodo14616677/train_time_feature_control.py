"""Paired eight-channel zero vs fixed source-frequency temporal feature trial."""
import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset
from time_feature_input import encode_time_features
from run_receiver_loss_control import supervised_losses
from run_scatter_target_pilot import evaluate_scatter
from train_multiworkpiece_control import state_hash
from train_scatter_multiworkpiece import evaluate_dataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    old=json.loads((a.control/'comparison.json').read_text());plan=json.loads(a.plan.read_text())
    assert plan['status']=='planned_not_run' and len(old['sample_orders'])==20
    assert sha(a.data/'manifest.json')==old['protocol']['data_manifest_sha256']
    train=OperatorPilotDataset(a.data,'train');val=OperatorPilotDataset(a.data,'validation');assert len(train)==48 and len(val)==24
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(2);legacy=BasicFNO();assert state_hash(legacy)==old['initial_state_hashes'][1];del legacy
    models=[];hashes=[];optimizers=[];encoders=[];modes=['zero','sincos']
    for mode in modes:
        torch.manual_seed(2);model=BasicFNO(input_channels=8).to(device);hashes.append(state_hash(model));models.append(model)
        optimizers.append(torch.optim.Adam(model.parameters(),lr=.001));encoders.append(partial(encode_time_features,mode=mode))
    assert hashes[0]==hashes[1]
    sample=train[0];zero=encoders[0](sample);features=encoders[1](sample)
    assert torch.equal(zero[:6],features[:6]) and not zero[6:].count_nonzero()
    assert torch.allclose(features[6].square()+features[7].square(),torch.ones_like(features[6]),atol=2e-7)
    assert int(torch.fft.rfft(features[6,0,0]).abs().argmax())==30
    del zero,features,sample
    protocol={**plan,'status':'fixed_budget_protocol','data_manifest_sha256':sha(a.data/'manifest.json'),
        'plan_sha256':sha(a.plan),'script_sha256':sha(Path(__file__)),'homogeneous_sha256':sha(a.data/'homogeneous.pt'),
        'normalization_scales':train.manifest['normalization'],'initial_state_hashes':hashes,
        'legacy_six_channel_initialization_preserved':True,'feature_checks_passed':True,'test_unloaded':True,
        'sample_orders':old['sample_orders'],'feature_modes':modes,'model_config':models[0].config}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu');histories=[[],[]];start=time.perf_counter()
    for epoch,order in enumerate(old['sample_orders'],1):
        assert sorted(order)==list(range(48));losses=[[],[]]
        for model in models:model.train()
        for step,index in enumerate(order,1):
            sample=train[index];target=(sample['target']-bg['wavefield_snapshots'][sample['shot']:sample['shot']+1]/train.scale)[None].to(device);ids=sample['receiver_indices'].to(device)
            for arm,(model,optimizer,encoder) in enumerate(zip(models,optimizers,encoders)):
                inputs=encoder(sample)[None].to(device);optimizer.zero_grad(set_to_none=True);prediction=model(inputs)
                lf,lr=supervised_losses(prediction,target,ids,train.scale,train.manifest['normalization']['trace_rms']);loss=lf+lr
                assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step();losses[arm].append((float(lf.detach()),float(lr.detach())));del inputs,prediction,lf,lr,loss
            del target
            if step%24==0:print(f'Epoch {epoch}/20 paired step {step}/48 elapsed {time.perf_counter()-start:.1f}s',flush=True)
        validations=[]
        for arm,(model,encoder) in enumerate(zip(models,encoders)):
            model.eval();validation=evaluate_dataset(model,val,bg,device,encoder);validations.append(validation)
            histories[arm].append({'epoch':epoch,'updates':epoch*48,
                'training_field_mse_step_mean':float(np.mean([x[0] for x in losses[arm]])),
                'training_receiver_mse_step_mean':float(np.mean([x[1] for x in losses[arm]])),
                'validation':{k:v for k,v in validation.items() if k!='records'}})
            torch.save({'state_dict':model.state_dict(),'config':model.config,'epoch':epoch,'updates':epoch*48,'scale':train.scale,
                'receiver_loss_weight':1.,'target':'defect_minus_homogeneous','time_feature_mode':modes[arm],
                'data_manifest_sha256':protocol['data_manifest_sha256']},a.out/f'latest_weight{arm}.pt')
        (a.out/'history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
        print(json.dumps({'epoch':epoch,'arms':[{k:v for k,v in x.items() if k not in ('records','energy_sums')} for x in validations]}),flush=True)
    trainings=[]
    for arm,(model,encoder) in enumerate(zip(models,encoders)):
        checkpoint=torch.load(a.out/f'latest_weight{arm}.pt',weights_only=True,map_location='cpu');torch.save(checkpoint,a.out/f'final_weight{arm}.pt')
        trainings.append(evaluate_dataset(model,train,bg,device,encoder))
        sample=val[0];_,example=evaluate_scatter(model,encoder(sample)[None].to(device),sample['target'][None].to(device),sample,bg);torch.save(example,a.out/f'validation_example{arm}.pt')
    report={'status':'completed','protocol':protocol,'initial_state_hashes':hashes,'sample_orders':old['sample_orders'],
        'history':histories,'final_training':trainings,'final_validation':validations,'elapsed_seconds':time.perf_counter()-start}
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Paired temporal feature trial complete',flush=True)


if __name__=='__main__':main()

"""Fixed-budget ordinary FNO pilot; select checkpoint on validation field MSE."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=6)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'best.pt').exists(),'Choose a new run output'
    torch.manual_seed(2);rng=np.random.default_rng(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=BasicFNO().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    train=OperatorPilotDataset(args.data,'train');validation=OperatorPilotDataset(args.data,'validation')
    assert {r['group_id'] for r,_ in train.entries}.isdisjoint({r['group_id'] for r,_ in validation.entries})
    history=[];best=float('inf');start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        # Shuffle workpieces and shots, retaining one-file cache for efficient I/O.
        order=[int(4*g+s) for g in rng.permutation(len(train)//4) for s in rng.permutation(4)]
        for step,index in enumerate(order,1):
            sample=train[index];inputs=encode_fno_input(sample).unsqueeze(0).to(device);target=sample['target'].unsqueeze(0).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction=model(inputs);loss=(prediction-target).square().mean()
            assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();losses.append(float(loss.detach()))
            del inputs,target,prediction,loss
            if step%8==0:print(f'Epoch {epoch}, step {step}/{len(train)}, mean MSE={np.mean(losses):.6f}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
        model.eval();val=[]
        with torch.no_grad():
            for i in range(len(validation)):
                sample=validation[i];input=encode_fno_input(sample).unsqueeze(0).to(device);target=sample['target'].unsqueeze(0).to(device)
                val.append(float((model(input)-target).square().mean()));del input,target
        row={'epoch':epoch,'training_normalized_mse':float(np.mean(losses)),'validation_normalized_mse':float(np.mean(val)),'elapsed_seconds':time.perf_counter()-start};history.append(row)
        if row['validation_normalized_mse']<best:
            best=row['validation_normalized_mse'];torch.save({'state_dict':model.state_dict(),'config':model.config,'epoch':epoch,'scale':train.scale,'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()},args.out/'best.pt')
        (args.out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    torch.save({'state_dict':model.state_dict(),'config':model.config},args.out/'last.pt')
    report={'status':'fixed_budget_training_completed','epochs':args.epochs,'updates':args.epochs*len(train),'train_groups':len(train)//4,'validation_groups':len(validation)//4,'checkpoint_selection':'lowest validation wavefield normalized MSE; test unseen by training/selection','config':model.config,'optimizer':'Adam','learning_rate':.001,'batch_size':1,'seed':2,'precision':'float32 and complex64','data':str(args.data),'history':history,'parameter_count_real_equivalent':sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),'elapsed_seconds':time.perf_counter()-start,'torch_version':torch.__version__}
    (args.out/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Training complete',flush=True)


if __name__=='__main__':main()

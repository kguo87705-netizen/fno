"""Unified transfer-U-Net path, varying only the loaded checkpoint."""
import argparse
from common import *

def native_model(seed):
    torch.manual_seed(seed)
    model=NN.Unet([1,16,32,64,128],2,GMIN,bnorm=True)
    NN.initWeights(model)  # exact author call (not model.apply)
    torch.nn.init.normal_(model.convolutionsUp[-1].weight,std=.01,mean=.7)
    model.convolutionsUp[-1].bias.data.fill_(3)
    return model.to(DEVICE)

def run(case,seed=30,epoch=0,destination='inversions',steps=35,initialization='expanded'):
    if case.startswith('test_'):raise RuntimeError('Test inversion forbidden in first round')
    started=time.perf_counter();s=setup();f=dense_source(s)
    data=np.load(data_path(case))
    meas=torch.as_tensor(data['measurement'],device=DEVICE)
    inp=torch.as_tensor(data['initial_gradient'],device=DEVICE)[None,None]
    span=inp.amax()-inp.amin()
    assert span>0 and torch.isfinite(span)
    inp=(inp-inp.amin())/span*2-1
    model=native_model(seed)
    if epoch:
        if initialization=='ellipse':
            assert epoch==100
            weights=ROOT/('.build/singh-coverage-100epochs-20261013/control/model_Unet_200' if seed==30 else f'.build/singh-training-seeds-20261013/seed{seed-30}/control/model_Unet_200')
        else:weights=BUILD/'checkpoints'/f'seed{seed}'/f'epoch{epoch}.pt'
        model.load_state_dict(torch.load(weights,map_location=DEVICE,weights_only=True))
    model.train()
    torch.use_deterministic_algorithms(True)
    opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:(.2*e+1)**(-.5))
    costs=[];material_mse=[];snapshots={};fit=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for e in range(steps):
        opt.zero_grad(set_to_none=True)
        pred=torch.ones((1,1,258,130),device=DEVICE)
        pred[:,:,1:-1,1:-1]=model(inp)
        # Existing forward at iteration e represents exactly e completed updates.
        # Store it without an extra train-mode BN forward or physical evaluation.
        if e in [0,5,10,20]:snapshots[f'gamma_step{e}']=pred[0,0,1:-1,1:-1].detach().cpu().numpy().copy()
        cost,gradient=cost_grad(s,pred,f,meas)
        assert torch.isfinite(cost) and torch.isfinite(gradient).all()
        external=torch.zeros_like(pred);external[0,0,1:-1,1:-1]=gradient*1e10
        pred.backward(external)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5e-5)
        opt.step();scheduler.step()
        costs.append(float(cost));material_mse.append(float(np.mean((pred[0,0,1:-1,1:-1].detach().cpu().numpy().astype(float)-data['truth'])**2)))
        if (e+1)%10==0 or e+1==steps:
            print('STEP',case,seed,epoch,e+1,'cost',costs[-1],flush=True)
        del pred,gradient,external,cost
    torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit
    # Author train-mode BN retained. Extra material evaluation is separate from updates.
    with torch.no_grad():
        final=model(inp).detach();full=torch.ones((1,1,258,130),device=DEVICE);full[:,:,1:-1,1:-1]=final
        wave=receiver_forward(s,full)
        j35=.5*torch.sum(s['dx']*s['dy']*torch.trapz((wave-meas)**2,dx=s['dt'],dim=2))/4
    torch.cuda.synchronize()
    finalarray=final[0,0].cpu().numpy();assert np.isfinite(finalarray).all()
    snapshots['gamma_step35']=finalarray.copy()
    out=BUILD/destination/f'seed{seed}_epoch{epoch}'/case;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'fields.npz',truth=data['truth'],labels=data['labels'],prediction=finalarray,costs=np.array(costs),mse_history=np.array(material_mse),**snapshots)
    metrics=dict(case=case,seed=seed,epoch=epoch,updates=steps,initial_cost=costs[0],final_cost=float(j35),relative_cost_reduction=1-float(j35)/costs[0],material_mse=float(np.mean((finalarray.astype(float)-data['truth'])**2)),fit_seconds=fit_seconds,wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),physical_forward_calls=steps+1,physical_adjoint_calls=steps,initial_gradient_calls='shared preprocessing in T1; one forward+one adjoint per geometry')
    metrics.update(initialization=initialization,saved_material_steps=[0,5,10,20,35])
    dump(out/'metrics.json',metrics);print('INVERSION_RESULT',json.dumps(metrics),flush=True)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',default='validation_000');p.add_argument('--seed',type=int,default=30);p.add_argument('--epoch',type=int,default=0);p.add_argument('--destination',default='inversions');p.add_argument('--initialization',choices=['expanded','ellipse'],default='expanded');a=p.parse_args()
    run(a.case,a.seed,a.epoch,a.destination,initialization=a.initialization)

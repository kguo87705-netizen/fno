"""Preregistered 131-input, frozen-branch temporal representation comparison."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from deeponet_multifrequency import MultiFrequencyDeepONet, coordinate_batch131, grid_predict
from operator_data import OperatorPilotDataset
from run_four_materials import evaluate, REGIONS
from run_receiver_loss_control import supervised_losses
from audit_four_material_response import energies, metrics
from time_reconstruction import cubic_time_reconstruction
from train_multiworkpiece_control import state_hash

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--mode',choices=['zero126','fourier126'],required=True); p.add_argument('--plan',type=Path,required=True); a=p.parse_args()
    plan=json.loads(a.plan.read_text()); assert plan['status']=='planned_not_run' and plan['updates_per_arm']==4000 and plan['arms']==['zero126','fourier126']
    a.out.mkdir(parents=True,exist_ok=True); assert not (a.out/'protocol.json').exists(); torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'cpu'
    ds=OperatorPilotDataset(a.data,'train'); s=ds[0]; assert s['group_id']=='synthetic_workpiece_000' and s['shot']==0
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu'); su=ds.scale; sr=ds.manifest['normalization']['trace_rms']; target=s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/su
    nx,ny,nt=target.shape; receiver=s['receiver_indices']; rid=(receiver[:,0,None]*ny*nt+receiver[:,1,None]*nt+torch.arange(nt)[None]).reshape(-1)
    rindices=torch.stack([rid//(ny*nt),(rid//nt)%ny,rid%nt],-1); rcoords=coordinate_batch131(s,rindices,a.mode); rtarget=target.reshape(-1)[rid.to(device)]
    torch.manual_seed(2); model=MultiFrequencyDeepONet().to(device); initial_branch_hash=state_hash(model.branch); [q.requires_grad_(False) for q in model.branch.parameters()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); generator=torch.Generator().manual_seed(19); points=32768
    protocol={'status':'fixed_budget_protocol','arm':a.mode,'branch':'frozen_initial_seed2','initial_branch_hash':initial_branch_hash,'trainable_parameters':[n for n,q in model.named_parameters() if q.requires_grad],'plan_sha256':sha(a.plan),'feature_module_sha256':sha(Path(__file__).with_name('deeponet_multifrequency.py')),'training_groups':[s['group_id']],'shot':0,'updates':4000,'seed':2,'torch_version':str(torch.__version__),'sampling_seed':19,'field_points_per_update':points,'receiver_points_per_update':int(rid.numel()),'loss':'sampled full-field MSE plus exact500-frame24receiver MSE rescaled su/sr, receiver weight1','optimizer':'Adam lr0.001','initial_state_hash':state_hash(model),'model_config':model.config,'data_manifest_sha256':sha(a.data/'manifest.json'),'homogeneous_sha256':sha(a.data/'homogeneous.pt'),'model_sha256':sha(Path(__file__).with_name('basic_deeponet.py')),'script_sha256':sha(Path(__file__)),'selection':'fixed final4000, evaluation0/400/4000; no validation/test access','scope':'Paired 131-input temporal representation control with initial branch frozen, training000/source0 only; no material generalization or physical-gradient/FWI claim.'}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8'); opt=torch.optim.Adam([q for q in model.parameters() if q.requires_grad],lr=.001); history=[]; indexhash=hashlib.sha256(); start=time.perf_counter()
    for update in range(4001):
        if update in [0,400,4000]:
            model.eval(); prediction=grid_predict(model,s,s['gamma'],device,a.mode); result,example=evaluate(None,[s],bg,su,sr,device,fixed=prediction,save_example=update==4000); history.append({'update':update,'metrics':result}); print(json.dumps({'update':update,'scatter_field':result['scatter_field_relative_l2'],'scatter_trace':result['scatter_trace_relative_l2']}),flush=True)
        if update==4000: break
        model.train(); opt.zero_grad(set_to_none=True); ids=torch.randint(nx*ny*nt,(points,),generator=generator); indexhash.update(ids.numpy().tobytes()); indices=torch.stack([ids//(ny*nt),(ids//nt)%ny,ids%nt],-1); coords=torch.cat([coordinate_batch131(s,indices,a.mode),rcoords],0).to(device); out=model(s['gamma'][None].to(device),coords)[0]
        lf=(out[:points]-target.reshape(-1)[ids.to(device)]).square().mean(); lr=((out[points:]-rtarget)*(su/sr)).square().mean(); loss=lf+lr; assert torch.isfinite(loss); loss.backward(); assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters()); assert all(q.grad is None for q in model.branch.parameters()); opt.step()
        if (update+1)%100==0: print(f'MultiFrequency update {update+1}/4000',flush=True)
    model.eval(); homogeneous=grid_predict(model,s,torch.ones_like(s['gamma']),device,a.mode); response=prediction-homogeneous; true=target; fulltimes=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9; ids=receiver.to(device); decode=lambda f:cubic_time_reconstruction(f[ids[:,0],ids[:,1],:]*su,s['time_seconds'].to(device),fulltimes)
    mr={'field':metrics(energies(response,true)),'trace':metrics(energies(decode(response),s['traces_full'].to(device)-bg['traces_full'][0].to(device))),**{n:metrics(energies(response[:,ys,:],true[:,ys,:])) for n,ys in REGIONS.items()}}
    final_branch_hash=state_hash(model.branch); assert final_branch_hash==initial_branch_hash
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':model.config,'updates':4000,'scale':su,'normalization':ds.manifest['normalization']},a.out/'final.pt'); torch.save(example,a.out/'training_example.pt')
    report={'status':'completed','protocol':protocol,'history':history,'final_training':result,'material_response_training_only':mr,'sampled_indices_sha256':indexhash.hexdigest(),'elapsed_seconds':time.perf_counter()-start,'final_branch_hash':final_branch_hash}; (a.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print('131-input frozen-branch arm complete',flush=True)
if __name__=='__main__': main()

"""Frozen initial/final optimization-state diagnostic, training000 only."""
import json,hashlib
from pathlib import Path
import torch
from basic_deeponet import BasicDeepONet,coordinate_batch
from operator_data import OperatorPilotDataset

def main():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';data=Path('.build/operator-pilot-v3-20261008');run=Path('.build/basic-deeponet-single-20261008');ds=OperatorPilotDataset(data,'train');s=ds[0];bg=torch.load(data/'homogeneous.pt',weights_only=True,map_location='cpu');target=s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale
    gen=torch.Generator().manual_seed(23);flat=torch.randint(256*128*500,(32768,),generator=gen);fieldindices=torch.stack([flat//(128*500),(flat//500)%128,flat%500],-1)
    recv=s['receiver_indices'];rid=(recv[:,0,None]*128*500+recv[:,1,None]*500+torch.arange(500)[None]).reshape(-1);ri=torch.stack([rid//(128*500),(rid//500)%128,rid%500],-1)
    coords=torch.cat([coordinate_batch(s,fieldindices),coordinate_batch(s,ri)],0).to(device);checkpoint=torch.load(run/'final.pt',weights_only=True,map_location=device);results={}
    for name in ['initial','final']:
        torch.manual_seed(2);model=BasicDeepONet().to(device)
        if name=='final':model.load_state_dict(checkpoint['state_dict'])
        model.eval();gamma=s['gamma'][None].to(device).requires_grad_();pred=model(gamma,coords)[0];lf=(pred[:32768]-target.reshape(-1)[flat.to(device)]).square().mean();lr=((pred[32768:]-target.reshape(-1)[rid.to(device)])*ds.scale/ds.manifest['normalization']['trace_rms']).square().mean();loss=lf+lr;loss.backward()
        with torch.no_grad():
            pre=model.branch[:8](gamma[:,None]);activation=model.branch[:9](gamma[:,None]);encoding=model.encode_material(gamma);uniform=model.encode_material(torch.ones_like(gamma));trunk=model.trunk(coords)
        gradnorm=lambda prefix:sum(float(p.grad.double().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5
        results[name]={'sampled_diagnostic_objective':float(loss.detach()),'branch_gradient_norm':gradnorm('branch'),'trunk_gradient_norm':gradnorm('trunk'),'material_input_gradient_norm':float(gamma.grad.double().norm()),'branch_fc_pre_min':float(pre.min()),'branch_fc_pre_max':float(pre.max()),'branch_fc_pre_below_minus6_fraction':float((pre< -6).float().mean()),'branch_activation_norm':float(activation.norm()),'material_encoding_norm':float(encoding.norm()),'homogeneous_encoding_difference_norm':float((encoding-uniform).norm()),'trunk_basis_norm':float(trunk.norm()),'prediction_rms':float(pred.detach().square().mean().sqrt()),'target_sampled_rms':float(target.reshape(-1)[flat.to(device)].square().mean().sqrt())}
    report={'scope':'Frozen initial/final read-only training000/source0, fresh fixed diagnostic point seed23, no optimizer updates. Gradient is model-computational, not FD physics. Activations and gradients diagnose optimization but do not alone establish spectral bias.','results':results,'checkpoint_sha256':hashlib.sha256((run/'final.pt').read_bytes()).hexdigest()}
    (run/'optimization_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()

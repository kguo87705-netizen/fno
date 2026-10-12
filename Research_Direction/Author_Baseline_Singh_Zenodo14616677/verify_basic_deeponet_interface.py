"""CPU interface checks only; not training, physical derivatives or validation."""
import json,hashlib
from pathlib import Path
import torch
from basic_deeponet import BasicDeepONet,coordinate_batch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset
from time_feature_input import encode_time_features
from run_deeponet_single import grid_predict

def main():
    torch.set_num_threads(4);torch.manual_seed(2);data=Path('.build/operator-pilot-v3-20261008');s=OperatorPilotDataset(data,'train')[0]
    indices=torch.tensor([[0,0,0],[255,127,499],[73,36,231],[128,65,300],[17,98,40]])
    coords=coordinate_batch(s,indices);canonical=encode_time_features(s,mode='zero');assert torch.equal(coords,canonical[1:6,indices[:,0],indices[:,1],indices[:,2]].T);del canonical
    model=BasicDeepONet().double();gamma=s['gamma'][None].double().requires_grad_();coordinates=coords.double().requires_grad_();out=model(gamma,coordinates);assert out.shape==(1,5) and torch.isfinite(out).all()
    material_grad,coordinate_grad=torch.autograd.grad(out.square().sum(),(gamma,coordinates));assert material_grad.norm()>0 and coordinate_grad.norm()>0
    generator=torch.Generator().manual_seed(19);direction=torch.randn(gamma.shape,generator=generator,dtype=torch.float64);direction/=direction.norm();epsilon=1e-3
    with torch.no_grad():fd=(model(gamma+epsilon*direction,coordinates).square().sum()-model(gamma-epsilon*direction,coordinates).square().sum())/(2*epsilon)
    ad=(material_grad*direction).sum();relative=float((fd-ad).abs()/(fd.abs()+ad.abs()).clamp_min(1e-14));assert relative<1e-5
    with torch.no_grad():encoding=model.encode_material(gamma);chunked=torch.cat([model.predict_from_encoding(encoding,coordinates[:2]),model.predict_from_encoding(encoding,coordinates[2:])],dim=1);chunk_error=float((chunked-out).abs().max());assert chunk_error<1e-12
    tiny={'gamma':torch.rand(8,8),'x_m':torch.linspace(0,.1,8),'y_m':torch.linspace(0,.05,8),'time_seconds':torch.arange(1,11,dtype=torch.float64)*120e-9,'source_xy_m':torch.tensor([.031,.05])}
    tiny_model=BasicDeepONet(grid=(8,8)).eval();flat=torch.arange(8*8*10);xyz=torch.stack([flat//80,(flat//10)%8,flat%10],-1)
    with torch.no_grad():direct=tiny_model(tiny['gamma'][None],coordinate_batch(tiny,xyz)).reshape(8,8,10);stream=grid_predict(tiny_model,tiny,tiny['gamma'],'cpu',chunk=17)
    grid_error=float((direct-stream).abs().max());assert grid_error<1e-7
    fno=BasicFNO(input_channels=8);real_fno=sum(p.numel()*(2 if p.is_complex() else 1) for p in fno.parameters());deep=sum(p.numel() for p in model.parameters())
    report={'status':'passed','scope':'CPU training000 interface and computational AD check only; no training, validation/test evaluation or physical gradient claim. Branch fixed grid256x128; trunk supports query chunks.','config':model.config,'real_scalar_parameters_deeponet':deep,'real_scalar_parameters_fno':real_fno,'parameter_ratio':deep/real_fno,'canonical_coordinate_parity':'bit-identical','gamma_and_coordinate_gradient_connectivity':'nonzero','computational_directional_derivative_relative_discrepancy':relative,'query_chunk_max_absolute_discrepancy':chunk_error,'full_grid_xyz_ordering_and_chunking_max_discrepancy':grid_error,'model_script_sha256':hashlib.sha256(Path(__file__).with_name('basic_deeponet.py').read_bytes()).hexdigest()}
    path=Path('.build/basic-deeponet-interface-20261008');path.mkdir(exist_ok=True);(path/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()

"""No-update spatial and temporal-frequency error coverage diagnostic."""
import json,torch,time,argparse
from pathlib import Path
from run_paired_depth_coverage import policy,data_setup,predict,BasicFNO,state_hash
from run_reduced_material_scan import sha
OUT=Path('.build/fno-coverage-diagnostic-20261011');PLAN=OUT/'preregistered_protocol.json';KEYS=['depth-2','depth0','depth2','depth-1','depth1','contrast03','contrast07']
def bands(x):
    z=torch.fft.rfft(x.double(),dim=-1).abs().square();n=z.shape[-1];cuts=[max(1,int(.1*n)),max(2,int(.4*n))];return [float(z[...,:cuts[0]].sum()),float(z[...,cuts[0]:cuts[1]].sum()),float(z[...,cuts[1]:].sum()),float(z.sum())]
def run():
    policy();p=json.loads(PLAN.read_text(encoding='utf-8'));assert not (OUT/'summary.json').exists()
    for f,h in p['frozen_sha256'].items():assert sha(Path(f))==h
    device,nom,bg,scale,sr,samples=data_setup();models={};start=time.perf_counter()
    for name,path in p['models'].items():
        ck=torch.load(Path(path),weights_only=True,map_location='cpu');m=BasicFNO(width=8,input_channels=8).to(device);m.load_state_dict(ck['state_dict']);m.eval();assert ('state_hash' not in ck) or state_hash(m)==ck['state_hash'];models[name]=[]
        for key in KEYS:
            pred=predict(m,{**nom,'gamma':samples[key]['gamma']},device,scale,bg);ref=samples[key]['physical'];
            out={'state':key}
            for space in ['field','trace']:
                base=bg['wavefield_snapshots'][0] if space=='field' else bg['traces_full'][0:1];err=(pred[space].double()-ref[space].double());sc=(ref[space].double()-base.double());
                out[space]={'full_relative_l2':float(err.norm()/ref[space].double().norm()),'scatter_relative_l2':float(err.norm()/sc.norm())}
                if space=='field':
                    out['field']['y_regions']={n:{'error_energy':float(err[sl,:,:].square().sum()),'scatter_energy':float(sc[sl,:,:].square().sum())} for n,sl in [('boundary',slice(0,2)),('interior',slice(2,-2)),('array_side',slice(-2,None))]}
                    out['field']['x_regions']={n:{'error_energy':float(err[:,sl,:].square().sum()),'scatter_energy':float(sc[:,sl,:].square().sum())} for n,sl in [('left',slice(0,2)),('interior',slice(2,-2)),('right',slice(-2,None))]}
                else:
                    eb=bands(err);sb=bands(sc);out['trace']['frequency_bands']={'low':{'error_energy':eb[0],'scatter_energy':sb[0]},'mid':{'error_energy':eb[1],'scatter_energy':sb[1]},'high':{'error_energy':eb[2],'scatter_energy':sb[2]},'total':{'error_energy':eb[3],'scatter_energy':sb[3]}}
            models[name].append(out)
        del m,ck;torch.cuda.empty_cache()
    raw={'status':'completed','models':models,'elapsed_seconds':time.perf_counter()-start,'calls':{'forward':28,'backward':0,'optimizer':0,'physical':0},'scope':'Existing final models and seven stored development states; spatial y/x partitions and receiver FFT residual energy; no training or physical derivative.'};torch.save(raw,OUT/'raw.pt');(OUT/'summary.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8');print('Completed')
def verify():
    p=json.loads(PLAN.read_text(encoding='utf-8'));raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert raw==json.loads((OUT/'summary.json').read_text(encoding='utf-8'));assert raw['calls']=={'forward':28,'backward':0,'optimizer':0,'physical':0};assert all(torch.isfinite(torch.tensor(v['field']['full_relative_l2'])) for rs in raw['models'].values() for v in rs);(OUT/'verification.json').write_text(json.dumps({'status':'passed','raw_summary':'Exact equality','calls':raw['calls'],'limits':'No native frequency response calibration;development states only;no physical derivative or independent final test.'},ensure_ascii=False,indent=2),encoding='utf-8');print('Verification passed')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');x=a.parse_args();verify() if x.verify else run()

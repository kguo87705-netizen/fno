"""Frozen validation material-versus-homogeneous finite response, source0 only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_four_materials import load_samples,REGIONS


def energies(pred,true):
    pred=pred.double();true=true.double()
    return {'error':float((pred-true).square().sum()),'predicted':float(pred.square().sum()),'reference':float(true.square().sum()),'dot':float((pred*true).sum())}


def metrics(v):
    return {'relative_l2':np.sqrt(v['error']/v['reference']),'norm_ratio':np.sqrt(v['predicted']/v['reference']),
        'cosine':v['dot']/np.sqrt(v['predicted']*v['reference']) if v['predicted']>0 else None,'energy_sums':v}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu';ds,_,val=load_samples(a.data);bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    c=torch.load(a.run/'final.pt',weights_only=True,map_location=device);assert c['updates']==1600 and c['scale']==ds.scale
    model=BasicFNO(input_channels=8).to(device);model.load_state_dict(c['state_dict']);model.eval();s=val[0];ids=s['receiver_indices'].to(device);times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
    decode=lambda field:cubic_time_reconstruction(field[ids[:,0],ids[:,1],:]*ds.scale,s['time_seconds'].to(device),times)
    sums={name:{k:0. for k in ['error','predicted','reference','dot']} for name in ['field','trace',*REGIONS]};records=[]
    with torch.no_grad():
        homogeneous=model(encode_time_features(s,torch.ones_like(s['gamma']),mode='zero')[None].to(device))[0,0];homogeneous_trace=decode(homogeneous)
        for s in val:
            actual=model(encode_time_features(s,mode='zero')[None].to(device))[0,0];response=actual-homogeneous
            true=s['target'][0].to(device)-bg['wavefield_snapshots'][0].to(device)/ds.scale
            trace_response=decode(actual)-homogeneous_trace;true_trace=s['traces_full'].to(device)-bg['traces_full'][0].to(device)
            values={'field':energies(response,true),'trace':energies(trace_response,true_trace),**{name:energies(response[:,ys,:],true[:,ys,:]) for name,ys in REGIONS.items()}}
            for name,v in values.items():
                for k,x in v.items():sums[name][k]+=x
            records.append({'group_id':s['group_id'],'shot':0,**{name:metrics(v) for name,v in values.items()}})
            del actual,response,true,trace_response,true_trace
    report={'scope':'Six validation workpieces, same source0; FNO(gamma)-FNO(1) versus FD(gamma)-FD(1). All-homogeneous input is a diagnostic counterfactual. Not local gradients, test or FWI.',
        'summary':{name:metrics(v) for name,v in sums.items()},'records':records}
    (a.run/'material_response.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()

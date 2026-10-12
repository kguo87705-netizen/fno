"""Frozen sin/cos-model input ablation; diagnostic, not another trained arm."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    ds=OperatorPilotDataset(a.data,'validation');device='cuda' if torch.cuda.is_available() else 'cpu'
    c=torch.load(a.run/'final_weight1.pt',weights_only=True,map_location=device);assert c['time_feature_mode']=='sincos' and c['updates']==960
    cfg=c['config'];model=BasicFNO(width=cfg['width'],modes=tuple(cfg['modes']),layers=cfg['layers'],input_channels=8).to(device);model.load_state_dict(c['state_dict']);model.eval()
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    sums={k:0. for k in ['field_change','true_scatter_field','trace_change','true_scatter_trace','zeroed_field_error','zeroed_trace_error']};records=[]
    with torch.no_grad():
        for i in range(len(ds)):
            s=ds[i];x=encode_time_features(s,mode='sincos')[None].to(device);active=model(x)[0,0];x[:,6:]=0;zero=model(x)[0,0]
            truth=s['target'][0].to(device)-bg['wavefield_snapshots'][s['shot']].to(device)/ds.scale
            ids=s['receiver_indices'].to(device);times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
            decode=lambda v:cubic_time_reconstruction(v[ids[:,0],ids[:,1],:]*ds.scale,s['time_seconds'].to(device),times)
            active_trace=decode(active);zero_trace=decode(zero);reference=s['traces_full'].to(device)-bg['traces_full'][s['shot']].to(device)
            values={'field_change':float((active-zero).double().square().sum()),'true_scatter_field':float(truth.double().square().sum()),
                'trace_change':float((active_trace-zero_trace).double().square().sum()),'true_scatter_trace':float(reference.double().square().sum()),
                'zeroed_field_error':float((zero-truth).double().square().sum()),'zeroed_trace_error':float((zero_trace-reference).double().square().sum())}
            for k,v in values.items():sums[k]+=v
            records.append({'group_id':s['group_id'],'shot':s['shot'],**values})
    report={'scope':'Frozen sin/cos model, validation24 shots; zeroing features is a counterfactual input outside the trained representation, not a separately trained causal control.',
        'feature_removal_field_change_over_true_scatter':np.sqrt(sums['field_change']/sums['true_scatter_field']),
        'feature_removal_trace_change_over_true_scatter':np.sqrt(sums['trace_change']/sums['true_scatter_trace']),
        'zeroed_scatter_field_relative_l2':np.sqrt(sums['zeroed_field_error']/sums['true_scatter_field']),
        'zeroed_scatter_trace_relative_l2':np.sqrt(sums['zeroed_trace_error']/sums['true_scatter_trace']),
        'energy_sums':sums,'records':records}
    (a.run/'feature_use.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()

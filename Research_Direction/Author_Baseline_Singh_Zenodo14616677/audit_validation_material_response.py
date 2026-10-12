"""Frozen final-model validation secant response to defect vs homogeneous material."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input
from time_reconstruction import cubic_time_reconstruction
from time_feature_input import encode_time_features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--arms',type=int,nargs='+',default=[0,1]);args=parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ds=OperatorPilotDataset(args.data,'validation')
    bg=torch.load(args.data/'homogeneous.pt',weights_only=True,map_location='cpu');results=[]
    for arm in args.arms:
        c=torch.load(args.run/f'final_weight{arm}.pt',weights_only=True,map_location=device)
        assert c['epoch']==20 and c['updates']==960 and c['scale']==ds.scale
        config=c.get('config',{});model=BasicFNO(width=config.get('width',4),modes=tuple(config.get('modes',(16,8,64))),layers=config.get('layers',4),input_channels=config.get('input_channels',6)).to(device);model.load_state_dict(c['state_dict']);model.eval()
        cache={};records=[];totals={'response_error':0.,'true_scatter':0.,'predicted_response':0.,'dot':0.}
        with torch.no_grad():
            for i in range(len(ds)):
                sample=ds[i];shot=sample['shot'];indices=sample['receiver_indices'].to(device)
                times=torch.arange(1,2001,device=device,dtype=torch.float64)*ds.manifest['dt_seconds']
                def traces(gamma):
                    inputs=encode_time_features(sample,gamma,c['time_feature_mode']) if 'time_feature_mode' in c else encode_fno_input(sample,gamma)
                    field=model(inputs[None].to(device))[0,0]
                    sparse=field[indices[:,0],indices[:,1],:]*ds.scale
                    return cubic_time_reconstruction(sparse,sample['time_seconds'].to(device),times)
                if shot not in cache:cache[shot]=traces(torch.ones_like(sample['gamma'])).cpu()
                pred=traces(sample['gamma']).cpu();response=pred-cache[shot]
                true_response=sample['traces_full']-bg['traces_full'][shot]
                values={'response_error':float((response-true_response).double().square().sum()),
                    'true_scatter':float(true_response.double().square().sum()),
                    'predicted_response':float(response.double().square().sum()),
                    'dot':float((response.double()*true_response.double()).sum())}
                for k,v in values.items():totals[k]+=v
                records.append({'group_id':sample['group_id'],'shot':shot,
                    'secant_response_relative_l2':np.sqrt(values['response_error']/values['true_scatter']),
                    'response_norm_ratio':np.sqrt(values['predicted_response']/values['true_scatter'])})
        cosine=totals['dot']/max(np.sqrt(totals['true_scatter']*totals['predicted_response']),1e-30)
        results.append({'arm':arm,'time_feature_mode':c.get('time_feature_mode','none'),'receiver_loss_weight':c.get('receiver_loss_weight',arm),'target':c.get('target','full_field'),'split':'validation','groups':6,'shots':24,
            'secant_response_relative_l2':np.sqrt(totals['response_error']/totals['true_scatter']),
            'response_norm_ratio':np.sqrt(totals['predicted_response']/totals['true_scatter']),
            'response_cosine':cosine,'records':records,'energy_sums':totals})
        del model
    report={'scope':'Frozen epoch-20 models, validation material-vs-homogeneous secant response; not a local gradient audit or a test-set score.',
        'definition':'Compare (FNO(gamma)-FNO(1)) with (FD(gamma)-FD(1)) in full receiver traces; same shot and reconstruction.',
        'homogeneous_input_caution':'All-homogeneous material is a diagnostic counterfactual, not itself a training workpiece.',
        'models':results}
    (args.run/'material_response.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({**report,'models':[{k:v for k,v in r.items() if k!='records'} for r in results]}),flush=True)


if __name__=='__main__':main()

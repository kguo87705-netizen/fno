"""Separate whole-grid and physical boundary-row errors on the memorized shot."""
import argparse
import json
from pathlib import Path
import torch
from operator_data import OperatorPilotDataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    sample=OperatorPilotDataset(args.data,'train')[0]
    saved=torch.load(args.run/'example.pt',weights_only=True,map_location='cpu')
    truth=sample['target'][0];prediction=saved['prediction_field'];difference=prediction-truth
    ny=truth.shape[1];regions={'whole_grid':torch.ones(ny,dtype=torch.bool),
        'first_two_y_rows':torch.arange(ny)<2,'last_two_y_rows':torch.arange(ny)>=ny-2,
        'interior_y_rows':(torch.arange(ny)>=2)&(torch.arange(ny)<ny-2)}
    records={};total_error=float(difference.square().sum())
    for label,mask in regions.items():
        error=float(difference[:,mask].square().sum());energy=float(truth[:,mask].square().sum())
        records[label]={'relative_l2':(error/energy)**.5,'fraction_of_whole_grid_error_energy':error/total_error,
                        'grid_cell_fraction':float(mask.float().mean())}
    ids=sample['receiver_indices'];p=prediction[ids[:,0],ids[:,1]];r=truth[ids[:,0],ids[:,1]]
    report={'group_id':sample['group_id'],'shot':sample['shot'],'split':'train',
        'receiver_y_rows':ids[:,1].unique().tolist(),'regions':records,
        'receiver_sparse_field_relative_l2':float((p-r).norm()/r.norm()),
        'caution':'Descriptive localization on one fitted training shot; not a causal boundary ablation or gradient audit.'}
    (args.run/'boundary_error.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()

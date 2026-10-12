"""Descriptive FFT coverage of scattering targets, train/validation only."""
import argparse
import json
from pathlib import Path
import torch
from operator_data import OperatorPilotDataset
from diagnose_basic_fno import spectral_audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu');device='cuda' if torch.cuda.is_available() else 'cpu';summaries={};records=[]
    with torch.no_grad():
        for split in ['train','validation']:
            ds=OperatorPilotDataset(a.data,split);total=joint=temporal=0.
            for i in range(len(ds)):
                s=ds[i];delta=s['target'][0].to(device)-bg['wavefield_snapshots'][s['shot']].to(device)/ds.scale
                stats,_=spectral_audit(delta);energy=float(delta.square().sum(dtype=torch.float64))
                total+=energy;joint+=energy*stats['joint_retained_energy_fraction'];temporal+=energy*stats['temporal_retained_energy_fraction']
                records.append({'group_id':s['group_id'],'split':split,'shot':s['shot'],'energy':energy,**stats});del delta
            summaries[split]={'shots':len(ds),'joint_retained_energy_fraction':joint/total,'temporal_retained_energy_fraction':temporal/total}
            print(f'Scatter target spectrum complete: {split}',flush=True)
    result={'scope':'Descriptive full-FFT rectangular target-energy mask, not the exact learned Hermitian rFFT output or a strict FNO error bound; local branches and nonlinearities produce additional frequencies.',
        'target':'FD(gamma)-FD(1)','modes':[16,8,64],'mask':'x[-16,15],y[-8,7],t[-63,63]','test_unloaded':True,
        'summary':summaries,'records':records}
    (a.run/'scatter_target_spectrum.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()

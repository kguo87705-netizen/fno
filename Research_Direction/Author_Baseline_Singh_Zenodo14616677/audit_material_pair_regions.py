"""Separate array-side and interior scattering errors in fitted pair outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    ex=torch.load(a.run/'example.pt',weights_only=True,map_location='cpu');truth=ex['reference_residual_fields'];regions={'opposite_boundary_two_rows':slice(0,2),'interior':slice(2,126),'array_side_two_rows':slice(126,128)}
    models=[]
    for arm,pred in enumerate(ex['prediction_residual_fields']):
        records={}
        for name,ys in regions.items():
            target=truth[:,:,ys,:];prediction=pred[:,:,ys,:];reference=target[0]-target[1];response=prediction[0]-prediction[1]
            values={'field_error':float((prediction-target).double().square().sum()),'field_reference':float(target.double().square().sum()),
                'pair_response_error':float((response-reference).double().square().sum()),'pair_response_reference':float(reference.double().square().sum()),
                'pair_response_predicted':float(response.double().square().sum())}
            records[name]={'scatter_field_relative_l2':np.sqrt(values['field_error']/values['field_reference']),
                'pair_field_response_relative_l2':np.sqrt(values['pair_response_error']/values['pair_response_reference']),
                'pair_field_response_norm_ratio':np.sqrt(values['pair_response_predicted']/values['pair_response_reference']),'energy_sums':values}
        models.append({'arm':arm,'regions':records})
    r=json.loads((a.run/'comparison.json').read_text())
    for arm,model in enumerate(models):
        total=sum(z['energy_sums']['field_error'] for z in model['regions'].values());assert np.isclose(total,r['final'][arm]['energy_sums']['field_error'],rtol=1e-6)
    result={'scope':'Frozen fitted-pair spatial diagnostic. Physical y indices0..1,2..125,126..127 partition the grid; receivers are at127. Not a causal loss-weight ablation or physical-gradient check.','models':models}
    (a.run/'region_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

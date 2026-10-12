"""CPU-only fixed nearest-material lookup diagnostic, train/validation source0."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from run_twelve_materials import load_samples
from run_four_materials import REGIONS

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);ds,train,val=load_samples(a.data);bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    a.out.mkdir(parents=True,exist_ok=True)
    td=torch.stack([s['gamma'].double()-1 for s in train]);vd=torch.stack([s['gamma'].double()-1 for s in val])
    distances=(vd[:,None]-td[None]).square().sum((2,3));nearest=distances.argmin(1)
    sums={k:0. for k in ['trace_error','trace_true','scatter_trace_true','field_error','field_true','scatter_field_true']};records=[]
    for i,s in enumerate(val):
        j=int(nearest[i]);tr=train[j];pf=tr['target'][0]-bg['wavefield_snapshots'][0]/ds.scale;tf=s['target'][0]-bg['wavefield_snapshots'][0]/ds.scale
        pt=tr['traces_full']-bg['traces_full'][0];tt=s['traces_full']-bg['traces_full'][0]
        values={'trace_error':float((pt.double()-tt).square().sum()),'trace_true':float(s['traces_full'].double().square().sum()),'scatter_trace_true':float(tt.double().square().sum()),'field_error':float((pf.double()-tf).square().sum()),'field_true':float(s['target'][0].double().square().sum()),'scatter_field_true':float(tf.double().square().sum())}
        for k,v in values.items():sums[k]+=v
        support_t=td[j]!=0;support_v=vd[i]!=0
        records.append({'group_id':s['group_id'],'nearest_training_group':tr['group_id'],'distance_squared':float(distances[i,j]),'material_perturbation_relative_distance':float(distances[i,j].sqrt()/vd[i].norm()),'material_perturbation_cosine':float((vd[i]*td[j]).sum()/(vd[i].norm()*td[j].norm())),'defect_support_iou':float((support_t&support_v).sum()/(support_t|support_v).sum()),'scatter_trace_relative_l2':np.sqrt(values['trace_error']/values['scatter_trace_true']),'scatter_field_relative_l2':np.sqrt(values['field_error']/values['scatter_field_true']),'energy_sums':values})
    summary={'scatter_trace_relative_l2':np.sqrt(sums['trace_error']/sums['scatter_trace_true']),'scatter_field_relative_l2':np.sqrt(sums['field_error']/sums['scatter_field_true']),'full_trace_relative_l2':np.sqrt(sums['trace_error']/sums['trace_true']),'full_field_relative_l2':np.sqrt(sums['field_error']/sums['field_true'])}
    entries=[r for r in ds.manifest['records'] if r['split'] in ['train','validation']]
    ranges={}
    for split in ['train','validation']:
        rr=[r for r in entries if r['split']==split];dd=[d for r in rr for d in r['defects']]
        ranges[split]={'groups':[r['group_id'] for r in rr],'defect_count_histogram':{str(n):sum(len(r['defects'])==n for r in rr) for n in [1,2]},'defect_parameter_ranges':{key:[min(d[key] for d in dd),max(d[key] for d in dd)] for key in ['x_m','y_m','axis_x_m','axis_y_m','gamma']}}
    report={'scope':'Development diagnostic, fixed nearest gamma Euclidean distance; only training gamma selects a stored training source0 response. No validation waveform influences neighbor selection, no fitted parameters, no independent test or formal OOD claim. Coordinates and source fixed. Sparse perturbation distance is a heuristic, not a wave-physics metric.','selection_rule':'argmin sum((gamma_validation-gamma_train)^2), stable manifest order breaks ties','training_groups':[s['group_id'] for s in train],'validation_groups':[s['group_id'] for s in val],'data_manifest_sha256':hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),'summary':summary,'records':records,'coverage_ranges':ranges,'distance_squared_matrix':distances.tolist()}
    for k in sums:assert np.isclose(sums[k],sum(r['energy_sums'][k] for r in records),rtol=1e-12)
    assert not set(report['training_groups'])&set(report['validation_groups'])
    report['verification']='energy aggregation, grouped split, deterministic argmin passed'
    (a.out/'coverage.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'records':[{k:v for k,v in r.items() if k!='energy_sums'} for r in records],'coverage_ranges':ranges}),flush=True)

if __name__=='__main__':main()

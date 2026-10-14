"""Descriptive snapshots with final-step thresholds, never retune thresholds."""
from evaluate import score
from common import *

if __name__=='__main__':
    bank=json.loads((BUILD/'frozen_thresholds.json').read_text())
    folders={'A':BUILD/'calibration/seed30_epoch0','B':BUILD/'calibration_B/seed30_epoch100','C':BUILD/'calibration/seed30_epoch100'}
    all_ids=[f'validation_{i:03d}' for i in range(25)]
    available={};missing={}
    for group,folder in folders.items():
        available[group]=[];missing[group]=[]
        for case in all_ids:
            p=folder/case/'fields.npz'
            if p.exists() and all(f'gamma_step{s}' in np.load(p).files for s in [0,5,10,20,35]):available[group].append(case)
            else:missing[group].append(case)
    common=sorted(set(available['B'])&set(available['C']))
    out=dict(evidence='descriptive validation only; final-step threshold reused at all steps; no H2 test',available=available,missing=missing,common_BC_cases=common,results={})
    for group,key in [('A','0'),('B','B'),('C','100')]:
        threshold=bank['results'][key]['0.01']
        if threshold is None:out['results'][group]={'status':'no final 1% working point'};continue
        paths=[folders[group]/case/'fields.npz' for case in common]
        steps=[35] if group=='A' else [0,5,10,20,35]
        out['results'][group]={}
        for step in steps:
            out['results'][group][str(step)]=score(paths,threshold['threshold'],field_key='prediction' if group=='A' else f'gamma_step{step}')
    out['transitions']={}
    for group in ['B','C']:
        if 'status' in out['results'][group]:continue
        scores=out['results'][group];counts=dict(initial_detected=0,final_detected=0,recovered_at_final=0,lost_at_final=0,never_detected_at_saved_steps=0,transient_only=0)
        for i,case in enumerate(common):
            trajectories=np.array([scores[str(s)]['cases'][i]['detected'] for s in [0,5,10,20,35]],dtype=bool)
            if trajectories.size==0:continue
            first,last=trajectories[0],trajectories[-1]
            counts['initial_detected']+=int(first.sum());counts['final_detected']+=int(last.sum())
            counts['recovered_at_final']+=int((~first&last).sum());counts['lost_at_final']+=int((first&~last).sum())
            counts['never_detected_at_saved_steps']+=int((~trajectories.any(axis=0)).sum())
            counts['transient_only']+=int((~first&~last&trajectories.any(axis=0)).sum())
        out['transitions'][group]=counts
    dump(BUILD/'intermediate_descriptive.json',out)
    print('INTERMEDIATE_DESCRIPTIVE_SAVED',dict(common_BC=len(common),missing={k:len(v) for k,v in missing.items()}),flush=True)

"""Frozen validation threshold selection and descriptive grouped scoring."""
import collections
from scipy import ndimage
from common import *

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,mid-half),min(1.,mid+half)]

def score(paths,threshold,field_key='prediction'):
    rows=json.loads((BUILD/'accepted_geometries.json').read_text());metadata={r['id']:r for r in rows}
    tp=collections.Counter();total=collections.Counter();pixel=collections.defaultdict(list)
    false=background=0;blankfalse=blankbg=0;cases=[];mses=[]
    for path in paths:
        case=path.parent.name;assert case.startswith('validation_')
        data=np.load(path);pred=data[field_key];truth=data['truth'];labels=data['labels']
        abnormal=abs(pred-1)>=threshold*(1-GMIN)
        bg=~ndimage.binary_dilation(labels>0,structure=np.ones((3,3)),iterations=2)
        nfalse=int((abnormal&bg).sum());nbg=int(bg.sum());false+=nfalse;background+=nbg
        detections=[]
        for i,p in enumerate(metadata[case]['components'],1):
            recall=float(abnormal[labels==i].mean());typ=str(p['category'])
            total[typ]+=1;tp[typ]+=int(recall>=.5);pixel[typ].append(recall);detections.append(recall>=.5)
        if metadata[case]['blank']:blankfalse+=nfalse;blankbg+=nbg
        mses.append(float(np.mean((pred.astype(float)-truth)**2)))
        cases.append(dict(case=case,false_alarm=nfalse/nbg,detected=detections))
    return dict(threshold=float(threshold),false_alarm=false/background,blank_false_alarm=blankfalse/blankbg if blankbg else None,detected=int(sum(tp.values())),total=int(sum(total.values())),mean_material_mse=float(np.mean(mses)),groups={k:dict(detected=int(tp[k]),total=int(n),recall=tp[k]/n,wilson95=wilson(tp[k],n),mean_pixel_recall=float(np.mean(pixel[k]))) for k,n in total.items()},cases=cases)

if __name__=='__main__':
    results={}
    groups=[('0',BUILD/'calibration/seed30_epoch0'),('100',BUILD/'calibration/seed30_epoch100')]
    if (BUILD/'calibration_B').exists():groups.append(('B',BUILD/'calibration_B/seed30_epoch100'))
    for epoch,folder in groups:
        paths=[folder/f'validation_{i:03d}'/'fields.npz' for i in range(25)]
        assert all(p.exists() for p in paths),'Validation calibration incomplete; cannot freeze threshold'
        scan=[score(paths,t) for t in np.round(np.arange(.1,.90001,.05),2)]
        results[str(epoch)]={}
        for cap in [.01,.005]:
            eligible=[r for r in scan if r['false_alarm']<=cap]
            best=max(eligible,key=lambda r:(r['detected'],-r['false_alarm'],r['threshold'])) if eligible else None
            results[str(epoch)][str(cap)]=best
        results[str(epoch)]['scan']=scan
    bank=BUILD/'frozen_thresholds.json'
    if bank.exists():
        previous=json.loads(bank.read_text())
        for key in ['0','100']:
            assert previous['results'][key]==results[key],'STOP: existing A/C frozen thresholds or scores changed'
    dump(bank,dict(seed=30,split='validation only',frozen=True,results=results,limitation='validation thresholds frozen before test inversion; validation results are not independent test evidence'))
    print('THRESHOLDS_FROZEN',json.dumps({e:{cap:r[cap]['threshold'] if r[cap] else None for cap in ['0.01','0.005']} for e,r in results.items()}),flush=True)

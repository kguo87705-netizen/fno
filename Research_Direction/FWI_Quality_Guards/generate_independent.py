import time,json,collections
from common import *
from scipy import ndimage
from scipy.spatial.distance import cdist

def geometries():
    BUILD.mkdir(exist_ok=True,parents=True)
    old=[]
    import pandas as pd
    for group in ['control','experiment']:
        for p in (ROOT/f'.build/singh-coverage-paired-20261012/{group}/data').glob('material*.h5'):
            old.append(pd.read_hdf(p).values<GMIN+.1)
    for i in range(1,5):old.append(pd.read_hdf(AUTHOR/f'data_casestudy/material{i}.h5').values<.5)
    _,_,source_y=0,0,0
    sx,sy=U.getSourceLocations(255,127,36,4)
    src=np.stack([(np.array(sx)-1)*.1/255,(np.array(sy)-1)*.05/127],axis=1)
    allrows=[];stats={}
    for split,count,seed in [('validation',20,2026101301),('test',60,2026101302)]:
        rng=np.random.default_rng(seed)
        types=np.tile(np.arange(8),count//2);rng.shuffle(types)
        totals=collections.Counter()
        for case in range(count+5):
            spec=[];fine_points=[]
            for typ in ([] if case>=count else types[case*4:case*4+4]):
                for attempt in range(10000):
                    p=dict(x=float(rng.uniform(.008,.092)),y=float(rng.uniform(.008,.042)),angle=float(rng.uniform(0,np.pi)),category=int(typ))
                    if typ<4:p.update(kind='hole',diameter=[.002,.003,.004,.006][typ])
                    elif typ<7:p.update(kind='slit',width=[1,2,3][typ-4]*.1/255,length=float(rng.uniform(.006,.014)))
                    else:p.update(kind='ellipse',a=float(rng.uniform(.003,.006)),b=float(rng.uniform(.0015,.003)))
                    _,lab=raster([p],511,255);ix,iy=np.where(lab>0);points=np.stack([(ix-1)*.1/511,(iy-1)*.05/255],axis=1)
                    if points[:,0].min()<.005 or points[:,0].max()>.095 or points[:,1].min()<.005 or points[:,1].max()>.045:continue
                    if cdist(points,src).min()<.010:continue
                    if any(cdist(points,q).min()<.005 for q in fine_points):continue
                    _,lc=raster([p],255,127)
                    if ndimage.label(lc>0,np.ones((3,3)))[1]!=1:continue
                    spec.append(p);fine_points.append(points);totals[str(typ)]+=1;break
                else:raise RuntimeError('Geometry rejection exhausted')
            material,labels=raster(spec,255,127)
            assert ndimage.label(labels>0,np.ones((3,3)))[1]==len(spec)
            assert not any(np.array_equal(labels[1:-1,1:-1]>0,t) for t in old)
            if spec:
                old.append(labels[1:-1,1:-1]>0)
            row=dict(id=f'{split}_{case:03d}',split=split,blank=case>=count,components=spec)
            allrows.append(row)
        stats[split]=dict(defects=dict(totals),defective_boards=count,blank_boards=5)
    dump(BUILD/'geometries.json',allrows);dump(BUILD/'geometry_checks.json',dict(passed=True,counts=stats,minimum_boundary_mm=5,minimum_source_mm=10,minimum_gap_mm=5,old_material_masks_checked=len(old)))
    return allrows

if __name__=='__main__':
    started=time.time();rows=geometries();fine=setup(511,255,4000,15e-9);coarse=setup();f=dense_source(coarse)
    times=[]
    for row in rows:
        dest=BUILD/'data'/f"{row['id']}.npz"
        if dest.exists():continue
        t=time.time();truth,labels=raster(row['components'],255,127);truthf,_=raster(row['components'],511,255)
        with torch.no_grad():measurement=receiver_forward(fine,torch.as_tensor(truthf,device=DEVICE)[None,None])[:,:,1::2].clone()
        bg=torch.ones((1,1,258,130),device=DEVICE)
        j,g=cost_grad(coarse,bg,f,measurement)
        assert torch.isfinite(measurement).all() and torch.isfinite(g).all()
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,truth=truth[1:-1,1:-1],labels=labels[1:-1,1:-1],measurement=measurement.cpu().numpy(),initial_gradient=g.cpu().numpy(),background_cost=float(j))
        times.append(dict(id=row['id'],seconds=time.time()-t,background_cost=float(j)))
        del measurement,g,bg;cleanup()
        dump(BUILD/'generation_times.json',dict(rows=times,wall_seconds=time.time()-started))
        print('GENERATED',row['id'],round(times[-1]['seconds'],3),'total',round(time.time()-started,2),flush=True)
    print('GENERATION_COMPLETE',time.time()-started,flush=True)

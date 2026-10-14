"""One permitted correction after continuous-distance validation failure."""
from common import *
from scipy import ndimage
from scipy.spatial.distance import cdist

def boundary(p):
    t=np.linspace(0,2*np.pi,1440,endpoint=False)
    if p['kind']=='hole':a=p['diameter']/2*np.cos(t);b=p['diameter']/2*np.sin(t)
    elif p['kind']=='ellipse':a=p['a']*np.cos(t);b=p['b']*np.sin(t)
    else:
        q=np.linspace(-1,1,360)
        a=np.concatenate([q*p['length']/2,np.full(360,p['length']/2),q*p['length']/2,np.full(360,-p['length']/2)])
        b=np.concatenate([np.full(360,-p['width']/2),q*p['width']/2,np.full(360,p['width']/2),q*p['width']/2])
    return np.stack([p['x']+a*np.cos(p['angle'])-b*np.sin(p['angle']),p['y']+a*np.sin(p['angle'])+b*np.cos(p['angle'])],1)

def verify(rows):
    sx,sy=U.getSourceLocations(255,127,36,4);src=np.stack([(np.array(sx)-1)*.1/255,(np.array(sy)-1)*.05/127],1)
    minima=[1.,1.,1.];bad=[]
    for r in rows:
        bs=[]
        for p in r['components']:
            pts=boundary(p);edge=min(pts[:,0].min(),.1-pts[:,0].max(),pts[:,1].min(),.05-pts[:,1].max());source=float(cdist(pts,src).min())
            minima[0]=min(minima[0],edge);minima[1]=min(minima[1],source)
            if edge<.005 or source<.01:bad.append(r['id'])
            for q in bs:
                gap=float(cdist(pts,q).min());minima[2]=min(minima[2],gap)
                if gap<.005:bad.append(r['id'])
            bs.append(pts)
        _,lab=raster(r['components'],255,127)
        if ndimage.label(lab>0,np.ones((3,3)))[1]!=len(r['components']):bad.append(r['id'])
    return sorted(set(bad)),minima

if __name__=='__main__':
    rows=json.loads((BUILD/'geometries.json').read_text());bad,before=verify(rows)
    dump(BUILD/'continuous_distance_failure.json',dict(failure_count=1,ids=bad,minima_m=before,reason='raster distance check underestimates continuous edge extent'))
    sx,sy=U.getSourceLocations(255,127,36,4);src=np.stack([(np.array(sx)-1)*.1/255,(np.array(sy)-1)*.05/127],1)
    rng=np.random.default_rng(2026101399)
    for row in rows:
        if row['id'] not in bad:continue
        placed=[]
        for old in row['components']:
            p=dict(old)
            radius=p['diameter']/2 if p['kind']=='hole' else (p['a'] if p['kind']=='ellipse' else np.hypot(p['length'],p['width'])/2)
            for attempt in range(10000):
                p['x']=float(rng.uniform(.005+radius,.095-radius));p['y']=float(rng.uniform(.005+radius,.045-radius))
                center=np.array([p['x'],p['y']])
                if np.linalg.norm(src-center,axis=1).min()<.010+radius:continue
                if any(np.linalg.norm(center-q[0])<.005+radius+q[1] for q in placed):continue
                placed.append((center,radius));break
            else:raise RuntimeError('Second T1 failure: corrected geometry placement exhausted')
            old.update(p)
    afterbad,after=verify(rows)
    assert not afterbad,('Second T1 failure',afterbad)
    # Preserve originally rejected metadata and waveforms. Use a manifest, not overwrites.
    dump(BUILD/'accepted_geometries.json',rows)
    dump(BUILD/'geometry_data_overrides.json',{i:'corrected_data/'+i+'.npz' for i in bad})
    dump(BUILD/'continuous_geometry_checks.json',dict(passed=True,corrected_ids=bad,minima_m=after,originals_preserved=True))
    fine=setup(511,255,4000,15e-9);coarse=setup();f=dense_source(coarse)
    for row in rows:
        if row['id'] not in bad:continue
        t=time.time();truth,labels=raster(row['components'],255,127);truthf,_=raster(row['components'],511,255)
        with torch.no_grad():measurement=receiver_forward(fine,torch.as_tensor(truthf,device=DEVICE)[None,None])[:,:,1::2].clone()
        j,g=cost_grad(coarse,torch.ones((1,1,258,130),device=DEVICE),f,measurement)
        dest=BUILD/'corrected_data'/f"{row['id']}.npz";dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,truth=truth[1:-1,1:-1],labels=labels[1:-1,1:-1],measurement=measurement.cpu().numpy(),initial_gradient=g.cpu().numpy(),background_cost=float(j))
        print('CORRECTED',row['id'],time.time()-t,flush=True);del measurement,g;cleanup()

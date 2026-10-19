"""Freeze balanced nonoverlapping source lists before any M2 propagation."""
import json,sys,collections,shutil
from pathlib import Path
import numpy as np,h5py
from m2_common import load_raw,array_sha,model_from_raw,write_new_json
from deepwave_compare import sha
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));root=Path(cfg['dataset_dir']);assert shutil.disk_usage(cfg['root']).free>=cfg['minimum_free_bytes']
assert not Path(cfg['manifest_path']).exists(),'Manifest already frozen; generation must reuse it.'
model=json.loads(Path(cfg['model_config']).read_text(encoding='utf-8-sig'))['model'];raw={p:load_raw(cfg['raw_dir'],p) for p in ['train','test']}
with h5py.File(cfg['confirmed_sample']) as f:assert np.array_equal(model_from_raw(raw['train'][0][0],model),f['label'][:])
rng=np.random.default_rng(cfg['seed']);pools={(p,k):rng.permutation(np.flatnonzero(raw[p][1]==k)).tolist() for p in raw for k in range(1,27)}
rows=[];seen_raw=set();seen_model=set();distribution={}
for split,total in cfg['split_counts'].items():
 part='test' if split=='test' else 'train';extra=set(rng.permutation(np.arange(1,27))[:total%26].tolist());selected=[]
 for k in range(1,27):
  target=total//26+int(k in extra)
  for _ in range(target):
   while True:
    idx=int(pools[part,k].pop());a=raw[part][0][idx];rh=array_sha(a);c=model_from_raw(a,model);ch=array_sha(c)
    if rh not in seen_raw and ch not in seen_model:break
   seen_raw.add(rh);seen_model.add(ch);selected.append(dict(split=split,emnist_source=part,emnist_index_zero_based=idx,letter_id=k,letter=chr(64+k),raw_image_sha256=rh,c_200_sha256=ch))
 for j in rng.permutation(len(selected)):
  row=selected[int(j)];row['sample_id']=f'm2_{len(rows):06d}';rows.append(row)
 distribution[split]=dict(sorted(collections.Counter(x['letter'] for x in selected).items()))
assert len(rows)==sum(cfg['split_counts'].values())==len(seen_raw)==len(seen_model)
assert len({(x['emnist_source'],x['emnist_index_zero_based']) for x in rows})==len(rows)
raw_hashes={p.name:sha(p) for p in Path(cfg['raw_dir']).glob('emnist-letters-*.gz')}
manifest=dict(seed=cfg['seed'],split_counts=cfg['split_counts'],letter_counts=distribution,raw_file_sha256=raw_hashes,model_preparation_verified_against_confirmed_sample=True,samples=rows)
write_new_json(cfg['manifest_path'],manifest);root.mkdir(parents=True,exist_ok=False)
for sub in ['samples','records','figures']: (root/sub).mkdir()
write_new_json(root/'dataset_config.json',cfg);write_new_json(root/'split_manifest.json',manifest)
write_new_json(root/'identity.json',{'config_sha256':sha(sys.argv[1]),'manifest_sha256':sha(cfg['manifest_path']),'prepare_script_sha256':sha(__file__),'free_bytes_before_start':shutil.disk_usage(cfg['root']).free})
print(json.dumps({'dataset_dir':str(root),'samples':len(rows),'distribution':distribution,'manifest_sha256':sha(cfg['manifest_path'])},indent=2))

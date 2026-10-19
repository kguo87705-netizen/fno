"""Shared deterministic EMNIST model and M2 file checks."""
import gzip,struct,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def load_raw(raw_dir,part):
 p=Path(raw_dir)
 with gzip.open(p/f'emnist-letters-{part}-images-idx3-ubyte.gz','rb') as f:
  magic,n,h,w=struct.unpack('>4I',f.read(16));assert magic==2051 and (h,w)==(28,28);images=np.frombuffer(f.read(),np.uint8).reshape(n,h,w)
 with gzip.open(p/f'emnist-letters-{part}-labels-idx1-ubyte.gz','rb') as f:
  magic,count=struct.unpack('>2I',f.read(8));assert magic==2049 and count==n;labels=np.frombuffer(f.read(),np.uint8);assert len(labels)==n and labels.min()==1 and labels.max()==26
 return images,labels

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def model_from_raw(raw,model):
 upright=raw.T.copy();resized=np.asarray(Image.fromarray(upright).resize(tuple(reversed(model['resize_shape'])),Image.Resampling.BILINEAR))
 canvas=np.zeros(model['label_shape'],np.float32);top,left=(np.array(canvas.shape)-np.array(resized.shape))//2;canvas[top:top+len(resized),left:left+resized.shape[1]]=resized
 return (model['c_background_m_s']-(model['c_background_m_s']-model['c_defect_m_s'])*canvas/255).astype(np.float32)

def write_new_json(path,value):
 with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

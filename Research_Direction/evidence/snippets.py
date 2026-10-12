from pathlib import Path
import re,json
base=Path('Research_Direction/evidence')
for key in ['P01','P02','P03','P05','P06','P07','P08','P09','P10','B01','B02']:
 p=next(p for p in base.glob(key+'*.md') if 'translate' not in p.name)
 parts=re.split(r'## PDF页 (\d+)\n',p.read_text(encoding='utf-8'))
 print('\n===',key,'===')
 for i in range(1,len(parts),2):
  page=parts[i]; s=parts[i+1]
  pats=r'(?i)(?:future work|future research|limitations?|500|1000|2000|training samples|training set|test set|dataset consists|dataset contains|training data|RTX|seconds|minutes|hours|resolution|generalization)'
  hits=list(re.finditer(pats,s))
  if hits:
   excerpts=[]
   for m in hits[:5]: excerpts.append(re.sub(r'\s+',' ',s[max(0,m.start()-160):m.end()+300]))
   print('p'+page+': '+' || '.join(excerpts))

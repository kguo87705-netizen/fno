from pathlib import Path
import json,hashlib,re,collections
root=Path.cwd(); out=root/'Research_Direction'
inv=json.loads((out/'evidence/inventory.json').read_text(encoding='utf-8'))
changed=[x['path'] for x in inv if not (root/x['path']).is_file() or hashlib.sha256((root/x['path']).read_bytes()).hexdigest()!=x['sha256']]
docs=[p for p in out.rglob('*.md') if 'evidence' not in p.relative_to(out).parts]
broken=[]
for p in docs:
 text=p.read_text(encoding='utf-8')
 for target in re.findall(r'\]\(([^)]+)\)',text):
  if target.startswith(('http:','https:','#')): continue
  target=target.split('#')[0]
  if not (p.parent/target).exists(): broken.append([p.name,target])
required=['00_Library_Index.md','01_Literature_Matrix.md','02_Research_Map.md','03_Gap_Analysis.md','04_Candidate_Directions.md','05_Final_Recommendations.md','Future_Work_Mining.md']
missing=[p for p in required if not (out/p).exists()]
scores=json.loads((out/'evidence/scores.json').read_text(encoding='utf-8'))
score_errors=[x['id'] for x in scores['ranking'] if abs(x['score']-sum(a*b for a,b in zip(x['scores'],scores['weights'])))>1e-8]
counts=collections.Counter(x['sha256'] for x in inv if x['ext']=='.pdf')
dup=[[x['path'] for x in inv if x['ext']=='.pdf' and x['sha256']==h] for h,n in counts.items() if n>1]
result={'original_files_checked':len(inv),'changed_original_files':changed,'deliverable_markdown_files':len(docs),'paper_cards':len(list((out/'papers').glob('*.md'))),'candidate_count':len(scores['ranking']),'missing_required':missing,'broken_local_links':broken,'score_errors':score_errors,'exact_pdf_duplicates':dup,'visual_QA':'P05 PDF p3 and P08 PDF p5; no Obsidian UI rendering','experiments_run':False}
(out/'evidence/validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

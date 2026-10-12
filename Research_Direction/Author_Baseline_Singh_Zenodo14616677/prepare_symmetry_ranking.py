import json
from pathlib import Path
p=Path('.build/same-quality-cost-audit-20261011/heldout_support_candidate_ranking_protocol.json')
d=json.loads(p.read_text());d['target']['material']=0.47;d['purpose']='symmetric replication of heldout support candidate ranking at a=.47';d['status']='planned_not_run';(p.parent/'heldout_support_candidate_ranking_symmetry_protocol.json').write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
s=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_heldout_support_candidate_ranking_20261011.py').read_text()
s=s.replace("O=Path('.build/heldout-support-candidate-ranking-20261011')","O=Path('.build/heldout-support-candidate-ranking-symmetry-20261011')").replace("P=Path('.build/same-quality-cost-audit-20261011/heldout_support_candidate_ranking_protocol.json')","P=Path('.build/same-quality-cost-audit-20261011/heldout_support_candidate_ranking_symmetry_protocol.json')").replace("target=make(3,.53)","target=make(3,.47)")
Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_heldout_support_candidate_ranking_symmetry_20261011.py').write_text(s,encoding='utf-8')

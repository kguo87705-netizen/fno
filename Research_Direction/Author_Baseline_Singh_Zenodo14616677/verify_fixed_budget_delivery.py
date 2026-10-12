"""Check saved documentation against machine results without new physics."""
import json,re
from pathlib import Path
from run_reduced_material_scan import sha
O=Path('.build/fno-physical-fixed-budget-control-20261009')
V=Path('PINO超声成像知识库')
title='固定八步物理更新与初始化收益审查 2026-10-09'
note=V/'05实践与复习'/f'{title}.md'
result=json.loads((O/'summary.json').read_text())
assert json.loads((O/'verification.json').read_text())['status']=='passed'
text=note.read_text(encoding='utf-8')
for target in [.35,.45,.55,.65]:
 routes=[x for x in result['routes'] if abs(x['target']-target)<1e-6]
 expected='| '+f'{target:.2f}'+' | '+' | '.join(f"{x['records'][8]['contrast_error']:.3e}" for x in routes)+' |'
 assert expected in text
 assert abs(routes[0]['records'][8]['a']-routes[1]['records'][8]['a'])<1e-14
assert max(x['records'][2]['contrast_error'] for x in result['routes'] if x['method']!='fixed05')<3.7e-8
assert sum(x['actual_physical_calls'] for x in result['routes'])==227
paths=[note,V/'00导航'/'FNO与超声全波形反演科研计划 2026-10-07.md',V/'00导航'/'相控阵FWI与神经算子后续执行路线 2026-10-08.md',V/'04专题'/'相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md',V/'05实践与复习'/'代理初值与一次物理材料更新 2026-10-09.md']
names={p.stem for p in V.rglob('*') if p.is_file()}|{p.name for p in V.rglob('*') if p.is_file()}
for p in paths:
 s=p.read_text(encoding='utf-8');assert s.startswith('---\n') and s.count('```')%2==0
 for link in re.findall(r'\[\[([^\]]+)\]\]',s):
  target=link.split('|')[0].split('#')[0]
  assert target in names,(p,target)
assert sha(O/'fixed_budget_trajectories.png')==sha(V/'90附件'/f'{title}.png')
plan=json.loads((O/'next_depth_shift_protocol.json').read_text())
assert plan['status']=='planned_not_run'
assert not Path(plan['output_directory']).exists()
for p,h in plan['frozen_evidence_sha256'].items():assert sha(Path(p))==h
audit=json.loads((O/'cost_increase_audit.json').read_text())
expected=[]
for route in result['routes']:
 for i in route['cost_increase_updates']:
  expected.append({'method':route['method'],'target':route['target'],'update':i,'absolute_increase':route['records'][i]['cost']-route['records'][i-1]['cost'],'previous_cost':route['records'][i-1]['cost']})
assert audit==expected
delivery={'status':'passed','numeric_tables':'Match all12 primary results','checkpoints':'Second-update bound verified; no truth-based stopping','wikilinks':'All5 edited notes resolved','image':'Copied SHA matches; plot visually inspected separately','next_protocol':'Planned not run; evidence hashes match','cost_increases':'All saved and recomputed','limitation':'Markdown structural QA, no native Obsidian rendering claim','files_sha256':{str(p):sha(p) for p in paths}}
assert not (O/'delivery_verification.json').exists()
(O/'delivery_verification.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
print('Delivery verification passed')

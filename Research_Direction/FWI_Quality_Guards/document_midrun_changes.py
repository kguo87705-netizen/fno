"""Requested change disclosure, recovered from actual tool events and source data."""
import re,difflib
from datetime import datetime,timezone
from common import *
ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
note=ROOT/'PINO超声成像知识库/05实践与复习/T2 独立测试探索轮 2026-10-13.md'
session=Path(r'C:/Users/Administrator/.codex/sessions/2026/10/12/rollout-2026-10-12T16-23-03-01a12f35-47c6-76c0-9529-35ebbbfcdcec.jsonl')
needle='*** Update File: D:/下载/FNO_Research_Workspace/Research_Direction/FWI_Quality_Guards/analyze_round2.py'
events=[]
for line in session.open(encoding='utf-8'):
 row=json.loads(line);p=row.get('payload',{});source=p.get('input','')
 if row.get('type')=='response_item' and p.get('type')=='custom_tool_call' and source.startswith('text(await tools.apply_patch(') and needle in source:
  match=re.search(r'tools\.apply_patch\(("(?:\\.|[^"\\])*")\)',source)
  assert match
  patch=json.loads(match.group(1))
  if patch.splitlines()[1]==needle:events.append((row['timestamp'],patch))
assert len(events)==2,len(events)
before={r['id']:r for r in json.loads((BUILD/'geometries.json').read_text())}['test_023']
after={r['id']:r for r in json.loads((BUILD/'accepted_geometries.json').read_text())}['test_023']
diff='\n'.join(difflib.unified_diff(json.dumps(before,ensure_ascii=False,indent=2).splitlines(),json.dumps(after,ensure_ascii=False,indent=2).splitlines(),fromfile='geometries.json:test_023 (original, preserved)',tofile='accepted_geometries.json:test_023',lineterm=''))
corrected=data_path('test_023');assert corrected.name=='test_023.npz' and corrected.parent.name=='corrected_data'
with np.load(corrected) as f:truth=f['truth'].copy();labels=f['labels'].copy()
checked=[]
for group in ['A','B','C']:
 for seed in range(30,35):
  epoch=0 if group=='A' else 100
  path=ROUND/('group_'+group)/f'seed{seed}_epoch{epoch}'/'test_023/fields.npz'
  assert path.exists()
  with np.load(path) as f:
   assert np.array_equal(f['truth'],truth) and np.array_equal(f['labels'],labels)
  checked.append(f'{group}/seed{seed}')
stamp=datetime.fromtimestamp(corrected.stat().st_mtime,timezone.utc).isoformat()
lines=['\n## 中途改动披露（在首次测试检出评分前写入）',
'以下前两项来自本会话实际apply_patch工具事件时间，均为UTC（北京时间加8小时）；不是凭文件最终mtime推测。两次编辑只增加汇总字段和说明，未执行评分。此时读取过运行状态、耗时、波形损失和真值/标签一致性，没有读取或计算任何测试缺陷检出结果。',
'已冻结阈值、抽样和主/次判据均未改变；没有根据测试结果选择指标。后续汇报重排在新文件format_round2_report.py中完成，不修改分析数值或这两次历史diff。']
reasons=['增加16块有缺陷板的MSE独立汇总字段，并明确总体MSE按17板等权（含空板），避免统计口径含混；未更改原MSE、检出、虚警或判据计算。','将证据标签写成“实验验证（仅本轮数值观察；探索性）”，明确H2仍为假设，不作正式判定；只改文字，未改计算。']
for i,((timestamp,patch),reason) in enumerate(zip(events,reasons),1):
 lines.extend([f'### analyze_round2.py第{i}次编辑：{timestamp}',reason,'```diff',patch,'```'])
lines.extend(['### test_023几何修正：发生在首轮，不是第二轮中途',
f'accepted_geometries.json和覆盖映射在2026-10-13 08:54:54 UTC创建，修正观测文件mtime为{stamp}；第二轮2026-10-13 12:08:51 UTC才开始。',
'原因：原来的栅格距离检查低估连续边界范围；连续轮廓复核发现test_023、validation_011、validation_015不满足既定最小间距。只重置其缺陷中心位置，类型、尺寸、数量和角度不变；用相同规则重算真值、细网格观测和粗网格初始梯度，旧文件全部保留。',
'```diff',diff,'```',
'数据入口diff：`data/test_023.npz`（原文件保留）→`corrected_data/test_023.npz`；由geometry_data_overrides.json解析。',
'受影响的已完成第二轮运行：**无失效运行**。修正早于第二轮，test_023的A/B/C×5种子共15次均使用修正文件；已逐一核对输出真值与标签，全部相等，没有任何一次使用原拒绝几何，无需重跑。核对项：'+', '.join(checked)+'.',
'这项修正同样在读取任何测试检出结果之前；固定抽样使用已经修正的接受几何池，未在抽样冻结后换板、改阈值或改判据。'])
assert '## 中途改动披露' not in note.read_text(encoding='utf-8')
with note.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print('Documented two exact code diffs, geometry diff, and 15 corrected-data checks',flush=True)

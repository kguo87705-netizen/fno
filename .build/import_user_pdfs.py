from pathlib import Path
import json,re,hashlib,shutil,zipfile
v=Path('PINO超声成像知识库'); rows=json.loads(Path('.build/import_scan.json').read_text()); rp=v/'99来源/原文文件记录.json'; rec=json.loads(rp.read_text()); by={x['id']:x for x in rec}
names={'P01':'P01_U-FNO_FWI_2026.pdf','P02':'P02_PIFNN_2026.pdf','P03':'P03_SSWP_UNet_2026.pdf','P05':'P05_DualStage_FNO_2026.pdf','P07':'P07_DualStage_FNO_IUS_2025.pdf','P08':'P08_PTUNet_2026.pdf','P10':'P10_PINN_Transcranial_2023.pdf'}
for x in rows:
 pid=x['id'];r=by[pid];dest=v/'90附件/论文原文'/names[pid];src=Path(x['source']);shutil.copy2(src,dest);assert src.read_bytes()==dest.read_bytes()
 r.pop('reason',None);r.update(status='已入库',file=str(dest.relative_to(v)),pages=x['pages'],source_file=str(src),source_url='https://doi.org/'+r['doi'],version='用户下载的正式会议论文排版版' if pid=='P07' else '用户下载的期刊论文排版版',sha256=hashlib.sha256(src.read_bytes()).hexdigest(),imported='2026-09-10',fulltext_read=False)
 p=v/r['note'];t=p.read_text();t=re.sub(r'^pdf_status:.*$', 'pdf_status: "已入库"',t,flags=re.M);t=re.sub(r'^pdf_file:.*$', 'pdf_file: '+json.dumps(r['file'],ensure_ascii=False),t,flags=re.M)
 block=f'''## 本地原文

**[[{r['file']}|打开原文 PDF]]** · [[{r['file']}#page=1|从第一页开始]]

- 来源：你下载到 Downloads 的原文件（已保留），2026-09-10归档。
- 版本：{r['version']}；{r['pages']}页。[DOI入口]({r['source_url']})。
- 已核对标题、DOI、全部页面可解析及首尾页；尚未完成全文精读。
- 原文件名：`{src.name}`。

'''
 t=re.sub(r'## 原文获取\n.*?(?=\n## )',lambda m:block,t,flags=re.S)
 t=t.replace('- [ ] 获得并核对全文版本，将 PDF 放入 `90附件`，在此添加内部链接。','- [x] 获得并核对全文版本，已归档至 `90附件/论文原文`，见本页链接。')
 if pid=='P07':
  t=t.replace('会议日程题录','题录导读（全文已入库，待精读）').replace('目前未核验技术全文。','本次已取得4页技术全文并核对题目与DOI，技术内容待精读。')
 p.write_text(t)
rp.write_text(json.dumps(rec,ensure_ascii=False,indent=2))
lines=['# 原文下载清单','','更新：2026-09-10。**原表10篇已入库8篇，只缺P04、P06。另有2篇基础论文，总计10份PDF。**','','## 已入库原文','','| 编号 | 文献笔记 | PDF | 页数 |','| --- | --- | --- | --- |']
for r in sorted(rec,key=lambda x:x['id']):
 if r['status']=='已入库':lines.append(f"| {r['id']} | [[{Path(r['note']).stem}]] | [[{r['file']}|打开原文]] | {r['pages']} |")
lines+=['','## 仅剩两篇待下载','','| 编号 | 文献 | 下载入口 |','| --- | --- | --- |','| P04 | FMC-Net相控阵超分辨成像 | [作者公开全文页面](https://www.researchgate.net/publication/374384943) · [出版社](https://doi.org/10.1121/10.0021171) |','| P06 | PI-DeepUFNet不规则缺陷波场 | [SSRN下载页面](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6967141) |','','P04作者页面曾显示公开PDF，但本次自动访问被拒绝；P06有43页预印本入口，自动下载也被拒绝。手动下载后放入 `90附件/待导入原文`，或者留在 Downloads 并告诉我即可。无需重命名。','','## 本次整理说明','','本次新增P01、P02、P03、P05、P07、P08、P10共7篇，P09使用此前机构仓储版本。按标题和DOI核对，全部页面可解析，并检查了首尾页。P07为4页正式会议论文，不是会议日程。','','下载文件夹中的原件均保留；知识库副本按编号和方法命名，已连接对应笔记。已入库状态不等于已精读，导读和个人学习进度不自动升级。校验值及原文件位置见 `99来源/原文文件记录.json`。','','返回 [[开始学习]] · [[文献地图]]']
(v/'00导航/原文下载清单.md').write_text('\n'.join(lines)+'\n')
p=v/'00导航/开始学习.md';t=p.read_text();t=t.replace('原表10篇已入库1篇（P09），另有2篇基础论文PDF；其余9篇的下载链接与协助方式见清单。','原表10篇已入库8篇，仅缺P04、P06；另有2篇基础论文PDF。').replace('P07目前核对到会议日程。','P07已取得4页正式会议全文，现有导读仍待全文精读更新。');p.write_text(t)
p=v/'README.md';t=p.read_text().replace('1篇会议题录','1篇会议题录导读（全文已补齐）');p.write_text(t)
p=v/'99来源/原始表格说明.md';t=p.read_text();t+='\n最新附件状态：你补充的7篇论文已归档，原表10篇目前仅缺P04、P06；P07已取得正式4页全文。上述早期检索状态保留为历史记录，当前状态以 [[原文下载清单]] 为准。\n';p.write_text(t)
p=v/'99来源/核验记录.md';t=p.read_text();t+='\n## 用户下载文件归档\n\n新增P01、P02、P03、P05、P07、P08、P10，核对标题、DOI、页数、首尾页及逐页解析。P07为正式4页会议论文。当前原表论文已入库8篇，仅P04和P06缺失；历史下载失败记录已被当前获取状态取代，阅读证据层级尚未升级。详见 [[原文下载清单]]。\n';p.write_text(t)
p=v/'00导航/文献地图.md';t=p.read_text().replace('| 会议日程题录 |','| 题录导读；4页全文已入库待精读 |');p.write_text(t)
files=list(v.rglob('*'));targets={p.name for p in files if p.is_file()}|{p.stem for p in files if p.suffix=='.md'};bad=[]
for p in v.rglob('*.md'):
 for raw in re.findall(r'\[\[([^\]]+)\]\]',p.read_text()):
  q=raw.split('|')[0].split('#')[0]
  if q not in targets and not (v/q).exists():bad.append((str(p),q))
assert not bad,bad
assert len(list((v/'90附件/论文原文').glob('*.pdf')))==10
with zipfile.ZipFile('PINO超声成像知识库.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in v.rglob('*'):
  if p.is_file():z.write(p,p)
print('已新增7篇，原表8/10，另2篇基础论文；全部链接有效。')

from pathlib import Path
import json, hashlib, collections, zipfile
from pypdf import PdfReader
root=Path.cwd(); out=root/'Research_Direction'; rows=[]; pdfs=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or any(x in {'.git','Research_Direction'} for x in p.relative_to(root).parts): continue
 rel=p.relative_to(root).as_posix(); row={'path':rel,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ext':p.suffix.lower()}; rows.append(row)
 if p.suffix.lower()=='.pdf' and '90附件/论文原文/' in rel:
  try:
   reader=PdfReader(p); pages=[x.extract_text() or '' for x in reader.pages]
   row.update(pages=len(pages),low_text_pages=[i+1 for i,s in enumerate(pages) if len(s.strip())<100])
   dest=out/'evidence'/(p.stem+'.md'); dest.write_text('# '+p.name+'\n\n'+ '\n\n'.join('## PDF页 '+str(i+1)+'\n\n'+s for i,s in enumerate(pages)),encoding='utf-8')
   pdfs.append({'path':rel,'pages':len(pages),'first':pages[0][:2100] if pages else '', 'low':row['low_text_pages']})
  except Exception as e: row['error']=str(e)
(out/'evidence'/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'evidence'/'pdf_overview.json').write_text(json.dumps(pdfs,ensure_ascii=False,indent=2),encoding='utf-8')
counts=collections.Counter(r['ext'] for r in rows)
lines=['# 00 文献库索引','', '扫描日期：2026-09-15。扫描当前工作目录，排除 Git 内部数据及本次输出目录；包含隐藏构建材料，单独标记，不重复计作论文。原文件只读。','', '## 文件类型统计','', '|类型|数量|','|---|---:|']+[f'|{k or "无扩展名"}|{v}|' for k,v in sorted(counts.items())]
lines+=['','## 完整文件索引','', '|路径|字节|角色|','|---|---:|---|']
for r in rows:
 role='构建/缓存材料' if r['path'].startswith(('.build/','.obsidian/')) else ('译本，非独立论文' if 'translate' in r['path'] else '待分类科研材料')
 lines.append(f'|{r["path"]}|{r["size"]}|{role}|')
lines+=['','## 扫描边界','', 'ZIP仅列成员，不解压覆盖已有知识库；PNG登记为截图附件；既有Markdown是二级笔记，不替代论文。XLSX原始题录已有JSON导出，本轮以PDF核验为准。没有发现的扩展名不代表文件解析失败。','']
for p in root.glob('*.zip'):
 with zipfile.ZipFile(p) as z: lines+=['### '+p.name, '',f'归档成员数：{len(z.infolist())}。仅检查成员名称；不把压缩包内副本重复计入核心论文。','']+['- '+n for n in z.namelist()]
(out/'00_Library_Index.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(pdfs,ensure_ascii=False,indent=2))

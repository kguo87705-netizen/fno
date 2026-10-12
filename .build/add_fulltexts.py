from pathlib import Path
import json,re,hashlib,shutil,zipfile
from pypdf import PdfReader

root=Path('/Users/akong/Documents/ChatGPT/文献阅读')
v=root/'PINO超声成像知识库'
out=v/'90附件/论文原文';out.mkdir(exist_ok=True)
inbox=v/'90附件/待导入原文';inbox.mkdir(exist_ok=True)
downloads={
 'P09':('P09_FEUNet_2024.pdf','https://dr.ntu.edu.sg/server/api/core/bitstreams/9be85139-c21d-4e48-8ea6-c815fbf66663/content','机构仓储提供的期刊排版版，JASA 155(3):2257–2269',13),
 'B01':('B01_FNO_2010.08895v3.pdf','https://arxiv.org/pdf/2010.08895','arXiv:2010.08895v3，ICLR 2021版式；首栏版本日期2021-05-17',16),
 'B02':('B02_PINO_2111.03794v4.pdf','https://arxiv.org/pdf/2111.03794','arXiv:2111.03794v4；首栏版本日期2023-07-29',27),
}
dois={'P01':'10.1016/j.ymssp.2026.114179','P02':'10.1016/j.engappai.2026.113994','P03':'10.1016/j.ndteint.2025.103597','P04':'10.1121/10.0021171','P05':'10.1016/j.ndteint.2026.103759','P06':'10.2139/ssrn.6967141','P07':'10.1109/IUS62464.2025.11201843','P08':'10.1016/j.ultras.2025.107852','P09':'10.1121/10.0025391','P10':'10.1016/j.ultras.2023.107026'}
rg={'P01':'403739104','P02':'400741378','P03':'397779684','P04':'374384943','P05':'404232531','P06':'407324736','P07':'396723677','P08':'396321558'}
reasons={
 'P01':'未取得PDF；出版社下载返回403，ResearchGate仅提供索取全文。优先用机构访问出版社。',
 'P02':'未取得PDF；ResearchGate明确显示无公开全文，OpenAlex未列出开放仓储全文。请从出版社经机构访问下载。',
 'P03':'未取得PDF；ResearchGate为索取全文入口。已通过Crossref核对DOI。请从出版社经机构访问下载。',
 'P04':'作者已在ResearchGate公开上传全文，但下载请求及AIP下载均返回403。建议先手动打开作者页面点击 Download full-text PDF。',
 'P05':'未取得PDF；出版社下载返回403，ResearchGate仅提供索取全文。优先用机构访问出版社。',
 'P06':'SSRN有43页预印本及下载按钮，但页面和PDF下载请求返回403；浏览器打开超时。建议手动访问SSRN。',
 'P07':'已通过Crossref核对正式会议DOI；IEEE PDF请求返回418，ResearchGate无全文。建议经机构访问IEEE Xplore。',
 'P08':'未取得PDF；ResearchGate为索取全文入口，OpenAlex未列出开放仓储全文。请从出版社经机构访问下载。',
 'P10':'未取得PDF；出版社下载返回403，OpenAlex未列出开放仓储全文。请从出版社经机构访问下载。',
}
records=[]
for p in sorted((v/'02文献').glob('*.md')):
 pid=p.name[:3]
 if pid not in dois and pid not in downloads:continue
 text=p.read_text()
 if pid in ['P03','P07']:
  text=re.sub(r'^doi:.*$', 'doi: '+json.dumps(dois[pid]), text, count=1,flags=re.M)
 if pid in downloads:
  fn,url,version,np=downloads[pid];src=root/'.build/downloads'/f'{pid}.pdf'
  r=PdfReader(src);assert len(r.pages)==np
  assert all(len(page.extract_text().strip())>50 for page in r.pages)
  if pid=='P09':assert '10.0025391' in r.pages[0].extract_text()
  shutil.copy2(src,out/fn)
  digest=hashlib.sha256(src.read_bytes()).hexdigest()
  block=f'''## 本地原文

**[[90附件/论文原文/{fn}|打开原文 PDF]]** · [[90附件/论文原文/{fn}#page=1|从第一页开始]]

- 下载日期：2026-09-10；页数：{np}。
- 版本：{version}。
- [下载来源]({url})。
- 已核对标题、作者、全部页面可解析及首尾页显示；这里只确认附件完整性，未标记全文精读完成。
- 获取状态：已入库。阅读与复习任务由你实际完成后勾选。

'''
  if pid=='P09':text=text.replace('- [ ] 获得并核对全文版本，将 PDF 放入 `90附件`，在此添加内部链接。','- [x] 获得并核对全文版本，PDF 已放入 `90附件/论文原文`，链接见本页。')
  record={'id':pid,'status':'已入库','file':'90附件/论文原文/'+fn,'pages':np,'source_url':url,'version':version,'sha256':digest,'downloaded':'2026-09-10','fulltext_read':False}
 else:
  url='https://doi.org/'+dois[pid]
  block=f'''## 原文获取

**尚未下载原文。** {reasons[pid]}

[出版社／DOI入口]({url})'''
  if pid in rg:block+=f' · [ResearchGate页面](https://www.researchgate.net/publication/{rg[pid]})'
  if pid=='P06':block+=' · [SSRN下载页面](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6967141)'
  block+='\n\n下载后可放入 `90附件/待导入原文`，由助手核对版本后归档。见 [[原文下载清单]]。\n\n'
  record={'id':pid,'status':'待下载','doi':dois[pid],'reason':reasons[pid],'checked':'2026-09-10','fulltext_read':False}
 # Add fields without modifying user's reading status.
 end=text.index('\n---',4)
 fm=text[:end]
 for key,val in [('pdf_status',record['status']),('pdf_file',record.get('file',''))]:
  if re.search('^'+key+':',fm,re.M):fm=re.sub('^'+key+':.*$',key+': '+json.dumps(val,ensure_ascii=False),fm,flags=re.M)
  else:fm+='\n'+key+': '+json.dumps(val,ensure_ascii=False)
 text=fm+text[end:]
 text=re.sub(r'\n## (?:本地原文|原文获取)\n.*?(?=\n## |\Z)','',text,flags=re.S)
 # Insert after the main title, before the existing introduction.
 main=re.search(r'^# .+\n',text,re.M)
 text=text[:main.end()]+'\n'+block+text[main.end():]
 if pid=='P09':text=text.replace('另找到南洋理工机构仓储 PDF 入口，尚未逐页精读。','已从南洋理工机构仓储下载 PDF 并核对版本，尚未逐页精读。').replace('机构仓储全文入口（尚未下载精读）','机构仓储全文入口（已下载，未精读）')
 p.write_text(text)
 record['note']=str(p.relative_to(v));records.append(record)

lines=['# 原文下载清单','','更新：2026-09-10。**原表10篇中已下载1篇（P09），另下载2篇补充基础论文，共3份PDF。其余9篇需要继续获取。**','','本页只报告原文获取状态，未把“下载完成”当成“精读完成”。没有用会议日程、摘要网页或其他文章替代原文。','','## 已加入知识库','','| 文献 | PDF | 版本与页数 |','| --- | --- | --- |']
for pid in ['P09','B01','B02']:
 fn,url,ver,n=downloads[pid];note=next((v/'02文献').glob(pid+'*.md')).stem
 lines.append(f'| [[{note}]] | [[90附件/论文原文/{fn}|打开PDF]] | {ver}；{n}页 |')
lines+=['','## 需要你协助下载的9篇','','建议优先下载 P04、P06，其次核心 P01、P05。出版社访问可尝试学校图书馆入口或你已配置的校园网络；不需要向助手提供账号或密码。','','| 编号 | 文献与下载入口 | 本次结果及建议 |','| --- | --- | --- |']
for pid in ['P04','P06','P01','P05','P10','P02','P03','P08','P07']:
 note=next((v/'02文献').glob(pid+'*.md')).stem
 u='https://doi.org/'+dois[pid];extra=''
 if pid=='P04':extra=' · [作者公开PDF页面](https://www.researchgate.net/publication/374384943)'
 if pid=='P06':extra=' · [SSRN页面](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6967141)'
 if pid=='P07':extra=' · [IEEE Xplore](https://ieeexplore.ieee.org/document/11201843)'
 lines.append(f'| {pid} | [[{note}]] · [DOI入口]({u}){extra} | {reasons[pid]} |')
lines+=['','403/418是本次请求被站点拒绝的结果，不能单凭状态码认定你的账户没有访问权限。ResearchGate的“Request full-text”会联系作者；本次未代你发送请求。','','## 下载后怎样交给我','','把下载好的PDF放进本库的 **`90附件/待导入原文`**，或者直接把文件发到当前对话。可保留原文件名，不必自己重命名；如果方便，在文件名前加 P01、P05 等编号。完成后告诉我“文件放好了”。','','我会检查论文标题、DOI、版本、页数与重复文件，移动到 `90附件/论文原文`，更新笔记中的PDF链接、清单和备份包。该文件夹目前不会自动监控。','','## 核验方式','','已下载文件通过PDF文件头、逐页解析、标题与版本信息核对，以及首尾页视觉检查。文件来源、版本、页数和SHA-256保存于 `99来源/原文文件记录.json`。','','P03 DOI `10.1016/j.ndteint.2025.103597`、P07 DOI `10.1109/IUS62464.2025.11201843` 已用Crossref登记元数据核对标题。原始Excel仍保留不变。','','返回 [[开始学习]] · [[文献地图]]']
(v/'00导航/原文下载清单.md').write_text('\n'.join(lines)+'\n')
(v/'99来源/原文文件记录.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
home=v/'00导航/开始学习.md';t=home.read_text()
if '[[原文下载清单]]' not in t:t=t.replace('## 学习入口','## 原文附件\n\n[[原文下载清单]]：原表10篇已入库1篇（P09），另有2篇基础论文PDF；其余9篇的下载链接与协助方式见清单。\n\n## 学习入口')
home.write_text(t)
readme=v/'README.md';t=readme.read_text()
if '[[原文下载清单]]' not in t:t+='\n原文附件获取情况与手动下载入口：[[原文下载清单]]。\n'
readme.write_text(t)
orig=v/'99来源/原始表格说明.md';t=orig.read_text()
if '[[原文下载清单]]' not in t:t+='\n2026-09-10补充：P09与两篇基础文献PDF已入库；当前附件状态见 [[原文下载清单]]。导读证据层级保持原样，下载不等于已精读。\n'
orig.write_text(t)
for pid in ['P03','P07']:
 p=next((v/'02文献').glob(pid+'*.md'));t=p.read_text()
 if pid=='P03':t=t.replace('正式卷期及 DOI 待补。','DOI已补充为10.1016/j.ndteint.2025.103597（Crossref标题核验）；正式卷期日期待补。')
 if pid=='P07':t=t.replace('正式论文或摘要的 DOI、页码与会议记录在哪里？','已找到DOI 10.1109/IUS62464.2025.11201843；继续获取正式全文并核对页码。')
 p.write_text(t)
validation=v/'99来源/核验记录.md';t=validation.read_text()
t+='\n## 原文获取补充（2026-09-10）\n\n已核对并入库P09（13页）、B01（16页）与B02（27页）。P03与P07补齐DOI，但尚未取得全文。逐篇结果及来源见 [[原文下载清单]]，文件校验记录见 `原文文件记录.json`。\n'
validation.write_text(t)

files=list(v.rglob('*'));names={p.name for p in files if p.is_file()}|{p.stem for p in files if p.suffix=='.md'}
bad=[]
for p in v.rglob('*.md'):
 for raw in re.findall(r'\[\[([^\]]+)\]\]',p.read_text()):
  q=raw.split('|')[0].split('#')[0]
  if q not in names and not (v/q).exists():bad.append((str(p),q))
assert not bad,bad
assert len(list(out.glob('*.pdf')))==3
with zipfile.ZipFile(root/'PINO超声成像知识库.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in v.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(root))
print(json.dumps({'original_downloaded':1,'supplement_downloaded':2,'pending':9,'pdfs':3,'broken_links':len(bad)},ensure_ascii=False))

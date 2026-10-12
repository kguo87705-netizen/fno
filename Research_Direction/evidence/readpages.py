from pathlib import Path
import re,sys
p=next(p for p in Path('Research_Direction/evidence').glob(sys.argv[1]+'*.md') if 'translate' not in p.name)
a=re.split(r'## PDF页 (\d+)\n',p.read_text(encoding='utf-8'))
for i in range(1,len(a),2):
 if int(a[i]) in [int(x) for x in sys.argv[2].split(',')]: print('\n## PDF页 '+a[i]+'\n'+a[i+1])

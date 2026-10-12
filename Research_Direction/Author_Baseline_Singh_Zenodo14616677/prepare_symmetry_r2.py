from pathlib import Path
s=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_heldout_support_candidate_ranking_symmetry_20261011.py').read_text()
s=s.replace("O=Path('.build/heldout-support-candidate-ranking-symmetry-20261011')","O=Path('.build/heldout-support-candidate-ranking-symmetry-r2-20261011')").replace("'target':{'shift':3,'a':.53}","'target':{'shift':3,'a':plan['target']['material']}")
Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_heldout_support_candidate_ranking_symmetry_r2_20261011.py').write_text(s,encoding='utf-8')

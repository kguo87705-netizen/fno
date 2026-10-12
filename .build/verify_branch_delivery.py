import hashlib,json,re,py_compile
from pathlib import Path
import torch
torch.set_num_threads(4)
root=Path('.build/basic-deeponet-branch-control-20261008')
code=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677')
vault=Path('PINO超声成像知识库')
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
prior=read(root/'prior_artifact_hashes.json')
assert all(sha(Path(p))==h for p,h in prior.items())
authors=read(code/'source_manifest.json')['files']
assert len(authors)==7 and all(sha(code/p)==d['sha256'] for p,d in authors.items())
data=Path('.build/operator-pilot-v3-20261008')
oldhash=read(Path('.build/basic-fno-twelve-materials-20261008/training_validation_file_hashes.json'))['files']['synthetic_workpiece_000.pt']
assert sha(data/'synthetic_workpiece_000.pt')==oldhash
reports=[read(root/a/'comparison.json') for a in ('joint','frozen_initial_branch')]
for a in ('joint','frozen_initial_branch'):
    verify=read(root/a/'artifact_verification.json')
    assert verify['status']=='passed' and verify['updates']==4000 and verify['reload_max_metric_difference']==0
assert reports[0]['protocol']['initial_state_hash']==reports[1]['protocol']['initial_state_hash']
assert reports[0]['sampled_indices_sha256']==reports[1]['sampled_indices_sha256']
assert reports[1]['final_branch_hash']==reports[1]['protocol']['initial_branch_hash']
broot=Path('.build/receiver-temporal-basis-20261008');basis=read(broot/'summary.json')
w=torch.load(broot/'waveforms.pt',weights_only=True,map_location='cpu')
reference=w['reference_full'];assert reference.shape==(24,2000) and torch.isfinite(reference).all()
assert w['target_sparse'].shape==(24,500)
for name,pred in w['predictions_full'].items():
    assert pred.shape==reference.shape and torch.isfinite(pred).all()
    err=float((pred-reference).norm()/reference.norm())
    assert abs(err-basis['bases'][name]['full2000_relative_l2'])<1e-12
    assert basis['bases'][name]['rank']==basis['bases'][name]['basis_columns']
    assert basis['bases'][name]['least_squares_normal_equation_relative_residual']<1e-10
assert basis['fourier_projection_vs_truncated_fft_max_absolute_discrepancy']<1e-11
assert basis['data_manifest_sha256']==sha(data/'manifest.json')
assert sha(root/'branch_freeze_and_temporal_basis.png')==sha(vault/'90附件/基础DeepONet分支冻结与时间基诊断.png')
notes=[vault/'00导航/FNO与超声全波形反演科研计划 2026-10-07.md',vault/'00导航/开始学习.md',vault/'05实践与复习/基础DeepONet时间特征与预算对照 2026-10-08.md',vault/'05实践与复习/基础DeepONet分支冻结与时间基诊断 2026-10-08.md']
stems={p.stem for p in vault.rglob('*') if p.is_file()};names={p.name for p in vault.rglob('*') if p.is_file()};missing=[]
for p in notes:
    s=p.read_text(encoding='utf-8');assert s.startswith('---\n') and '\ufffd' not in s
    for link in re.findall(r'\[\[([^\]]+)\]\]',s):
        target=link.split('|')[0].split('#')[0]
        if target and target not in stems and target not in names: missing.append((str(p),target))
assert not missing,missing
assert '\ufffd' not in (code/'README_local.md').read_text(encoding='utf-8')
scripts=['run_deeponet_branch_control','verify_deeponet_branch_control','audit_deeponet_branch_state','audit_receiver_temporal_basis','summarize_deeponet_branch_control']
for s in scripts: py_compile.compile(str(code/(s+'.py')),doraise=True)
plan=read(root/'next_multifrequency_protocol.json')
assert plan['status']=='planned_not_run' and plan['updates_per_arm']==4000 and plan['arms']==['zero126','fourier126']
result={'status':'passed','prior_artifacts_unchanged':len(prior),'author_files_unchanged':len(authors),'training000_file_hash':'unchanged','paired_initialization_and_sampling':'identical','frozen_branch':'unchanged','model_reload_verifiers':'both passed, metric difference0','temporal_basis_saved_waveform_metrics':'recomputed, difference below1e-12','least_squares_rank_and_normal_equation':'passed','Fourier_vs_FFT':'passed','plot_copy':'SHA identical','plot_visual_review':'actually viewed earlier in this turn, readable layout','updated_notes':len(notes),'missing_wikilinks':missing,'script_syntax_checks':len(scripts),'next_protocol':'planned_not_run','scope':'Training000/source0 only. No validation/test access or physical-gradient/FWI claim.'}
(root/'delivery_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

"""Run independent full-window audits sequentially on the available device."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
results = []
for case in (2, 3, 4):
    for update in (0, 5):
        output = Path(f".build/author-discrete-case{case}-full64-update{update}-20261008")
        print(f"Starting case {case}, update {update}", flush=True)
        subprocess.run([sys.executable, str(root / "audit_discrete_gradient.py"), "--case", str(case), "--updates", str(update), "--dtype", "float64", "--adjoint-batch", "1", "--out", str(output)], check=True, stdout=(log := open(output.with_suffix('.log'), 'w', encoding='utf-8')))
        log.close()
        report = json.loads((output / "report.json").read_text())
        errors = [v['ad_relative_error'] for d in report['directions'].values() for v in d['checks'].values()]
        results.append({"case": case, "update": update, "forward_relative_error": report['forward_relative_error'], "max_direction_relative_error": max(errors), "passes": report['discrete_ad_passes_2_percent'], "author_gradient_angle": report['gradient_angle_degrees'], "output": str(output)})
        print(json.dumps(results[-1]), flush=True)
destination = Path('.build/author-discrete-extended-20261008')
destination.mkdir(parents=True, exist_ok=True)
(destination / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

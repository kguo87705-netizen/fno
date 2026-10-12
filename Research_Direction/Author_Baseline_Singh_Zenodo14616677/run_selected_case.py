"""Run a scoped author-code case while preserving downloaded source bytes.

Only case selection, epoch count and input/output paths are changed in memory.
All original numerical/optimizer statements and settings2D.csv are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    root = Path(__file__).resolve().parent
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    original = root / "conventionalFWI.py"
    source = original.read_text(encoding="utf-8")
    replacements = {
        "cases = np.linspace(1, 4, 4, dtype=int)": f"cases = np.array([{args.case}], dtype=int)",
        "    epochs = 35": f"    epochs = {args.epochs}",
        'destinationFolder = "data_casestudy"': "destinationFolder = outputFolder",
        "if destinationFolder not in os.listdir(os.getcwd()):": "if not os.path.isdir(destinationFolder):",
        'destinationFolder + "/material"': 'inputFolder + "/material"',
        'destinationFolder + "/measurement"': 'inputFolder + "/measurement"',
        'destinationFolder + "/source.h5"': 'inputFolder + "/source.h5"',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError(f"Expected one occurrence: {before}")
        source = source.replace(before, after)
    report = {
        "status": "running", "case": args.case, "epochs": args.epochs,
        "source": "https://zenodo.org/records/14616677",
        "original_script_md5": hashlib.md5(original.read_bytes()).hexdigest(),
        "runtime_replacements": replacements,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scope": ("Selected author case at original numerical settings and original 35 epochs; numerical agreement with published figures still requires assessment." if args.epochs == 35 else "Selected author case at original numerical settings; changed epochs are an execution check, not paper-result reproduction."),
    }
    report_path = args.out / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    namespace = {"__file__": str(original), "__name__": "__main__", "inputFolder": str(root / "data_casestudy"), "outputFolder": str(args.out)}
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        exec(compile(source, str(original), "exec"), namespace)
        with torch.no_grad():
            final_wavefield = namespace["forwardSolver"].forwardNSteps(
                namespace["u0"].clone(), namespace["u1"].clone(), namespace["gammaPred"].detach(), namespace["F"],
                int(namespace["Nx"]), int(namespace["Ny"]), int(namespace["N"]), int(namespace["numberOfSources"]), namespace["device"],
            )
            residual = final_wavefield[:, namespace["selx"], namespace["sely"], 1:] - namespace["um"].to(namespace["device"])
            final_cost = 0.5 * torch.sum(namespace["dx"] * namespace["dy"] * torch.trapezoid(residual.square(), dx=namespace["dt"], dim=2)) / namespace["numberOfSources"]
            report["post_update_cost"] = float(final_cost.item())
        report["status"] = "selected_case_completed"
        report["initial_cost"] = float(namespace["historyCost"][0])
        report["last_pre_update_cost"] = float(namespace["historyCost"][-1])
        report["post_update_material_mse_half"] = float(namespace["historyMSE"][-1])
        report["grid"] = [int(namespace["Nx"]), int(namespace["Ny"])]
        report["sources"] = int(namespace["numberOfSources"])
        report["receivers"] = int(namespace["numberOfSensors"])
        report["steps"] = int(namespace["N"])
        torch.save({"truth": namespace["gamma"].detach().cpu(), "reconstruction": namespace["gammaPred"].detach().cpu()}, args.out / "fields.pt")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        if torch.cuda.is_available():
            report["peak_cuda_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

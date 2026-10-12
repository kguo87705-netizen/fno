"""Frozen, known-support transmit-design diagnostic. No network or adaptive policy."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

import FiniteDifferencePyTorchConv as FD
import Utilities

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
GEOMETRIES = [
    dict(name="left_deep", x=.030, y=.025, a=.006, b=.003),
    dict(name="right_deep", x=.070, y=.025, a=.006, b=.003),
    dict(name="left_shallow", x=.030, y=.035, a=.006, b=.003),
    dict(name="right_shallow", x=.070, y=.035, a=.006, b=.003),
    dict(name="center_small", x=.050, y=.030, a=.005, b=.002),
    dict(name="center_large", x=.050, y=.030, a=.008, b=.0035),
]
KNOTS = [.44, .48, .52, .56]
TARGETS = [.46, .50, .54]
ACTIONS = {"left": [0, 1], "right": [2, 3], "full": [0, 1, 2, 3]}
NOISE = [0., .01, .03]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def mask_for(g, nx, ny, lx, ly):
    x = torch.linspace(0, lx, nx + 1)[:, None]
    y = torch.linspace(0, ly, ny + 1)[None, :]
    return ((x - g["x"]) / g["a"]) ** 2 + ((y - g["y"]) / g["b"]) ** 2 < 1


def generate(out):
    if out.exists():
        raise FileExistsError("Use a new directory; previous results are never overwritten.")
    out.mkdir(parents=True)
    started = time.perf_counter()
    files = [Path(__file__), CODE / "settings2D.csv", CODE / "data_casestudy/source.h5",
             CODE / "FiniteDifferencePyTorchConv.py", CODE / "Utilities.py"]
    protocol = dict(
        status="frozen_before_simulation", experiment="known_support_transmit_design_value",
        geometries=GEOMETRIES, candidate_materials=KNOTS, evaluation_materials=TARGETS,
        actions=ACTIONS, noise_fractions_background_rms=NOISE, noise_seeds=list(range(100)),
        noise_model="independent Gaussian per source/receiver/time, same realization across policies",
        noise_scale="single pooled four-source background RMS; fixed before geometry evaluation",
        score="minimum adjacent candidate separation squared / gamma step squared",
        score_tie="left when scores differ by at most 1e-12 times max score",
        estimator="continuous clipped least-squares projection onto each physical candidate segment",
        primary_noise_fraction=.01, primary_metric="mean gamma absolute error, equal weight per geometry",
        gate=dict(rank_switch="left and right each win in at least one geometry",
                  relative_mae_reduction_vs_best_fixed_at_least=.10,
                  absolute_mae_reduction_at_least=.0001),
        scientific_scope="development diagnostic, known geometry/support, scalar same-grid model; no blind detection, adaptive acquisition, neural proxy or final generalization test",
        physical_budget=dict(background=4, candidate_dictionary=96, targets=72, replays=2, total=174),
        network_forwards=0, training_updates=0, time_limit_seconds=1200,
        input_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        geometry_units="metres; x lateral, y measured from lower domain boundary; array at y=0.05",
        cost_boundary="all source simulations and replays counted; offline dictionary not claimed free; no measured hardware acquisition time",
    )
    dump(out / "protocol.json", protocol)
    cfg = pd.read_csv(CODE / "settings2D.csv").iloc[0]
    nx, ny, n = (int(cfg[k]) for k in ["Nx", "Ny", "N"])
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg["distanceBetweenSources"]), 4)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg["distanceBetweenSensors"]), sx)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force = torch.tensor(pd.read_hdf(CODE / "data_casestudy/source.h5").values).reshape(4, 1, nx+3, ny+3, n).to(device)
    dump(out / "environment.json", dict(python=sys.version, platform=platform.platform(),
         torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, device=str(device),
         gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
         source_dtype=str(force.dtype), grid=[nx+1, ny+1], samples=n, dt=float(cfg["dt"]),
         source_grid_indices=list(zip(sx, sy)), receiver_grid_indices=list(zip(rx, ry))))
    ledger = dict(status="running", entries=[], physical_calls_attempted=0,
                  network_forwards=0, training_updates=0,
                  unknown_costs=["hardware acquisition time", "historic setup and training costs"])
    manifest = []

    def simulate(material, source, label, role):
        if time.perf_counter() - started > protocol["time_limit_seconds"]:
            raise TimeoutError("Frozen time budget exceeded; no retries.")
        entry = dict(label=label, source=source, role=role, status="started")
        ledger["entries"].append(entry)
        ledger["physical_calls_attempted"] += 1
        dump(out / "ledger.json", ledger)
        t0 = time.perf_counter()
        try:
            gamma = torch.ones(1, 1, nx+3, ny+3, device=device)
            gamma[0, 0, 1:-1, 1:-1] = material.to(device)
            initial = torch.zeros_like(gamma)
            solver = FD.FiniteDifference(float(cfg["dt"]), float(cfg["Lx"] / nx),
                     float(cfg["Ly"] / ny), float(cfg["c"]), float(cfg["rho"]), device)
            with torch.no_grad():
                wave = solver.forwardNSteps(initial, initial.clone(), gamma,
                       force[source:source+1], nx, ny, n, 1, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                trace = wave[0, rx, ry, 1:].detach().cpu().contiguous().clone()
            assert trace.shape == (24, 2000) and torch.isfinite(trace).all()
            entry.update(status="completed", solve_and_extract_seconds=time.perf_counter()-t0)
            return trace
        except Exception as error:
            entry.update(status="failed", error=repr(error), solve_and_extract_seconds=time.perf_counter()-t0)
            raise
        finally:
            dump(out / "ledger.json", ledger)

    try:
        t0 = time.perf_counter()
        background = torch.stack([simulate(torch.ones(nx+1, ny+1), s, "background", "background") for s in range(4)])
        torch.save(background, out / "background.pt")
        for geometry in GEOMETRIES:
            mask = mask_for(geometry, nx, ny, float(cfg["Lx"]), float(cfg["Ly"]))
            assert mask.any() and not mask[[0, -1], :].any() and not mask[:, [0, -1]].any()
            for value in sorted(KNOTS + TARGETS):
                label = f"{geometry['name']}__gamma_{value:.2f}"
                role = "candidate_dictionary" if value in KNOTS else "target"
                material = torch.where(mask, torch.tensor(value), torch.tensor(1.))
                traces = torch.stack([simulate(material, s, label, role) for s in range(4)])
                file = label + ".pt"
                torch.save(dict(traces=traces, mask=mask, geometry=geometry, gamma=value,
                                source_indices=list(range(4))), out / file)
                manifest.append(dict(file=file, sha256=sha(out/file), geometry=geometry["name"], gamma=value, role=role))
            print(f"generated {geometry['name']}; physical calls={ledger['physical_calls_attempted']}", flush=True)
        repeat_background = simulate(torch.ones(nx+1, ny+1), 0, "replay_background", "replay")
        last = torch.load(out / manifest[-1]["file"], weights_only=True)
        last_material = torch.where(last["mask"], torch.tensor(last["gamma"]), torch.tensor(1.))
        repeat_last = simulate(last_material, 3, "replay_last_source3", "replay")
        replay = dict(background_source0_bit_equal=torch.equal(repeat_background, background[0]),
                      last_geometry_source3_bit_equal=torch.equal(repeat_last, last["traces"][3]))
        assert all(replay.values())
        assert ledger["physical_calls_attempted"] == 174
        dump(out / "data_manifest.json", dict(records=manifest, background_sha256=sha(out/"background.pt"),
             replay=replay, source_simulations=174))
        ledger.update(status="completed", generation_and_replay_seconds=time.perf_counter()-t0,
                      preparation_seconds=t0-started, wall_seconds_before_analysis=time.perf_counter()-started)
        dump(out / "ledger.json", ledger)
        print("generation and replays passed", flush=True)
    except Exception as error:
        ledger.update(status="failed", error=repr(error), wall_seconds_before_failure=time.perf_counter()-started)
        dump(out / "ledger.json", ledger)
        dump(out / "failure.json", dict(error=repr(error), no_automatic_retries=True))
        raise


def fit_segments(observations, candidates):
    """Fit every segment continuously, with endpoints included; no target label used."""
    best_error = np.full(len(observations), np.inf)
    best_gamma = np.zeros(len(observations))
    for i in range(len(KNOTS)-1):
        residual = observations - candidates[i]
        delta = candidates[i+1] - candidates[i]
        denominator = delta @ delta
        if denominator <= 0:
            raise ValueError("Unidentifiable candidate segment")
        projection = residual @ delta
        weight = np.clip(projection / denominator, 0, 1)
        error = np.einsum("ij,ij->i", residual, residual) - 2*weight*projection + weight**2*denominator
        chosen = error < best_error
        best_error[chosen] = error[chosen]
        best_gamma[chosen] = KNOTS[i] + weight[chosen]*(KNOTS[i+1]-KNOTS[i])
    return best_gamma


def analyze(out):
    if (out / "summary.json").exists():
        raise FileExistsError("Analysis already exists; do not overwrite.")
    started = time.perf_counter()
    p = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert p["candidate_materials"] == KNOTS and p["evaluation_materials"] == TARGETS
    bg = torch.load(out/"background.pt", weights_only=True).numpy().astype(np.float64)
    scale = float(np.sqrt(np.mean(bg**2)))
    selections = []; rows = []; artifacts = {}; t0 = time.perf_counter()
    for gi, geometry in enumerate(p["geometries"]):
        def load(value):
            d = torch.load(out/f"{geometry['name']}__gamma_{value:.2f}.pt", weights_only=True)
            return d["traces"].numpy().astype(np.float64) - bg
        bank = np.stack([load(value) for value in KNOTS])
        scores = {}
        for name in ["left", "right"]:
            difference = np.diff(bank[:, ACTIONS[name]], axis=0).reshape(3, -1)
            scores[name] = float(np.min(np.sum(difference**2, axis=1) / np.diff(KNOTS)**2))
        tolerance = 1e-12 * max(scores.values())
        selected = "right" if scores["right"] > scores["left"] + tolerance else "left"
        selections.append(dict(geometry=geometry["name"], scores=scores, selected=selected))
        for ti, value in enumerate(TARGETS):
            truth_wave = load(value)
            for ni, fraction in enumerate(NOISE):
                # Independent cases; common realization across all policies within each case.
                estimates = np.empty((100, 4))
                for seed in p["noise_seeds"]:
                    rng = np.random.default_rng(np.random.SeedSequence([20261012, gi, ti, ni, seed]))
                    observed = truth_wave + rng.normal(0., fraction*scale, truth_wave.shape)
                    estimates[seed, 0] = fit_segments(observed[ACTIONS["left"]].reshape(1, -1), bank[:, ACTIONS["left"]].reshape(4, -1))[0]
                    estimates[seed, 1] = fit_segments(observed[ACTIONS["right"]].reshape(1, -1), bank[:, ACTIONS["right"]].reshape(4, -1))[0]
                    estimates[seed, 2] = estimates[seed, 0 if selected == "left" else 1]
                    estimates[seed, 3] = fit_segments(observed.reshape(1, -1), bank.reshape(4, -1))[0]
                key = f"{geometry['name']}_gamma_{value:.2f}_noise_{fraction:.2f}"
                artifacts[key] = estimates
                for col, policy in enumerate(["fixed_left", "fixed_right", "physics_selected", "full_four_source"]):
                    error = np.abs(estimates[:, col]-value)
                    rows.append(dict(geometry=geometry["name"], gamma=value, noise_fraction=fraction,
                                     policy=policy, source_budget=4 if col==3 else 2,
                                     mae=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                                     max_error=float(error.max()), noise_repeats=100))
        print("analyzed " + geometry["name"], flush=True)
    np.savez_compressed(out/"parameter_estimates.npz", **artifacts)
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False, encoding="utf-8")
    pooled = []
    for fraction in NOISE:
        for policy in ["fixed_left", "fixed_right", "physics_selected", "full_four_source"]:
            subset = [x for x in rows if x["noise_fraction"]==fraction and x["policy"]==policy]
            pooled.append(dict(noise_fraction=fraction, policy=policy, mae=float(np.mean([x["mae"] for x in subset]))))
    primary = {x["policy"]:x["mae"] for x in pooled if x["noise_fraction"]==.01}
    best_fixed = min(primary["fixed_left"], primary["fixed_right"])
    reduction = best_fixed-primary["physics_selected"]
    rank_switch = len({x["selected"] for x in selections}) == 2
    gates = dict(rank_switch=rank_switch, relative_mae_reduction=reduction/max(best_fixed,1e-30),
                 absolute_mae_reduction=reduction,
                 passed=rank_switch and reduction>=.0001 and reduction/max(best_fixed,1e-30)>=.10)
    summary = dict(status="completed", action_selections=selections, pooled_metrics=pooled,
                   primary_noise_fraction=.01, gate=gates, background_rms=scale,
                   independent_geometries=6, targets_per_geometry=3, noise_repeats_not_independent_workpieces=100,
                   network_forwards=0, training_updates=0, physical_calls=174,
                   scope=p["scientific_scope"],
                   limitations=["known geometry used for dictionary and action scoring", "same solver and grid for targets and candidates", "piecewise-linear approximation bias", "Gaussian independent sample noise only", "four sources have twice the deployment budget", "dictionary costs 96 source solves; not a demonstrated total-cost saving", "six planned development geometries, no independent final test"])
    dump(out/"summary.json", summary)
    ledger=json.loads((out/"ledger.json").read_text(encoding="utf-8"))
    ledger.update(analysis_and_export_seconds=time.perf_counter()-started,
                  analysis_compute_seconds_before_export=time.perf_counter()-t0)
    dump(out/"ledger.json", ledger)
    print(json.dumps(dict(gate=gates, primary=primary), indent=2), flush=True)


def verify(out):
    started=time.perf_counter()
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    manifest=json.loads((out/"data_manifest.json").read_text(encoding="utf-8"))
    ledger=json.loads((out/"ledger.json").read_text(encoding="utf-8"))
    summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    for name, expected in protocol["input_sha256"].items():
        assert sha(ROOT/name)==expected, name
    assert sha(out/"background.pt")==manifest["background_sha256"]
    assert len(manifest["records"])==42
    for row in manifest["records"]:
        assert sha(out/row["file"])==row["sha256"]
        d=torch.load(out/row["file"],weights_only=True)
        assert d["traces"].shape==(4,24,2000) and torch.isfinite(d["traces"]).all()
    assert len(ledger["entries"])==174 and all(x["status"]=="completed" for x in ledger["entries"])
    assert all(manifest["replay"].values())
    assert set(KNOTS).isdisjoint(TARGETS)
    # Estimator behavior: exact segment interiors and clipped endpoints.
    toy=np.array([[0.,0.],[1.,2.],[3.,3.],[6.,4.]])
    assert np.allclose(fit_segments(np.array([[2.,2.5]]),toy),[.50])
    assert np.allclose(fit_segments(toy[[0,3]],toy),[.44,.56])
    table=pd.read_csv(out/"metrics.csv")
    assert len(table)==216 and np.isfinite(table["mae"]).all()
    archive=np.load(out/"parameter_estimates.npz",allow_pickle=False)
    assert len(archive.files)==54
    selected={x["geometry"]:x["selected"] for x in summary["action_selections"]}
    for name in archive.files:
        values=archive[name];assert values.shape==(100,4) and np.isfinite(values).all()
        assert (values>=.44).all() and (values<=.56).all()
        geom=name.split("_gamma_")[0]
        assert np.array_equal(values[:,2],values[:,0 if selected[geom]=="left" else 1])
        material=float(name.split("_gamma_")[1].split("_noise_")[0])
        fraction=float(name.split("_noise_")[1])
        for col,policy in enumerate(["fixed_left","fixed_right","physics_selected","full_four_source"]):
            metric=table[(table.geometry==geom)&(table.gamma==material)&(table.noise_fraction==fraction)&(table.policy==policy)]
            assert len(metric)==1 and np.isclose(metric.iloc[0].mae,np.abs(values[:,col]-material).mean(),rtol=1e-10,atol=1e-14)
    dump(out/"verification.json",dict(status="passed", level="behavioral and structural",
         checks=["input and all generated data hashes", "42 four-source finite waveform files", "174 attempted/completed source solves including two bit-identical replays", "candidate and target materials disjoint", "continuous segment estimator exact-interior and endpoint checks", "216 metrics independently recalculated from 54 saved estimate arrays", "equal two-source budgets and selected-policy identity"],
         scientific_gate_passed=summary["gate"]["passed"], verification_seconds=time.perf_counter()-started,
         not_verified=["independent solver convergence", "real instrument", "unknown-support detection", "Mac experiment execution"]))
    print("verification passed; scientific gate="+str(summary["gate"]["passed"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["generate","analyze","verify"])
    parser.add_argument("--out",default=".build/tx-design-value-20261012")
    args=parser.parse_args()
    out=ROOT/args.out
    {"generate":generate,"analyze":analyze,"verify":verify}[args.phase](out)

"""No-gradient, native-wheel GPU comparison; large outputs stay outside Git."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

SCRIPT_START = time.perf_counter()

import h5py
import numpy as np
import torch
import deepwave


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


class Monitor:
    def __init__(self, interval):
        self.interval = interval
        self.rows = []
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if r.returncode == 0:
                    mem, util = map(float, r.stdout.splitlines()[0].split(","))
                    self.rows.append([time.perf_counter(), mem, util])
            except Exception as exc:
                print("Monitor warning:", str(exc), flush=True)
            self.stop.wait(self.interval)


def stats(a):
    a = np.asarray(a)
    return {"min": float(a.min()), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def reciprocity(a):
    a = a.astype(np.float64)
    i, j = np.triu_indices(a.shape[0], 1)
    norm = np.linalg.norm(a, axis=-1)
    pair = np.linalg.norm(a[i, j] - a[j, i], axis=-1) / np.maximum((norm[i, j] + norm[j, i]) / 2, 1e-300)
    return {"global_relative_l2": float(np.linalg.norm(a - a.swapaxes(0, 1)) / np.linalg.norm(a)),
            "pair_relative_l2": stats(pair), "pair_count": len(pair)}


def metrics(ref, x):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    error = np.linalg.norm(x-ref, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-300)
    r0, x0 = ref-ref.mean(axis=-1, keepdims=True), x-x.mean(axis=-1, keepdims=True)
    corr = np.sum(r0*x0, axis=-1) / np.maximum(np.linalg.norm(r0, axis=-1)*np.linalg.norm(x0, axis=-1), 1e-300)
    return {"global_relative_l2": float(np.linalg.norm(x-ref)/np.linalg.norm(ref)),
            "trace_relative_l2": stats(error), "trace_pearson_zero_lag": stats(corr)}, error, corr


def shifted(x, lag):
    y = np.zeros_like(x)
    if lag > 0:
        y[..., lag:] = x[..., :-lag]
    elif lag < 0:
        y[..., :lag] = x[..., -lag:]
    else:
        y[:] = x
    return y


def align(ref, x, max_lag):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    rr = np.sum(ref*ref)
    r0 = ref-ref.mean(axis=-1, keepdims=True)
    rn = np.linalg.norm(r0, axis=-1)
    best = None
    best_corr = np.zeros(ref.shape[:-1])
    best_trace_lag = np.zeros(ref.shape[:-1], dtype=np.int32)
    for lag in range(-max_lag, max_lag+1):
        y = shifted(x, lag)
        yy, dot = np.sum(y*y), np.sum(ref*y)
        gain = dot/yy
        loss = rr-dot*dot/yy
        if best is None or loss < best[0]:
            best = (loss, lag, gain)
        y -= y.mean(axis=-1, keepdims=True)
        corr = np.sum(r0*y, axis=-1)/np.maximum(rn*np.linalg.norm(y, axis=-1), 1e-300)
        mask = np.abs(corr) > np.abs(best_corr)
        best_corr[mask], best_trace_lag[mask] = corr[mask], lag
    _, lag, gain = best
    return shifted(x, lag)*gain, int(lag), float(gain), best_corr, best_trace_lag


def propagate(cfg, v, wavelet, sources, receivers):
    with torch.no_grad():
        out = deepwave.scalar(
            v, cfg["dx_m"], cfg["dt_s"],
            source_amplitudes=wavelet, source_locations=sources, receiver_locations=receivers,
            accuracy=cfg["accuracy"], pml_width=cfg["pml_width"], pml_freq=cfg["pml_freq_hz"],
            max_vel=cfg["max_vel_m_s"], survey_pad=None,
            python_backend=False, storage_mode="none",
        )
    return out[-1]


def plot(cfg, run, t, ref, raw, aligned):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = cfg["pairs_one_based"]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(11, 12), sharex=True)
    for row, (tx, rx) in enumerate(pairs):
        a, b, c = ref[tx-1, rx-1], raw[tx-1, rx-1], aligned[tx-1, rx-1]
        ax = axes[row, 0]
        ax.plot(t*1e6, a, lw=.8, label="k-Wave (left)")
        ax.set_ylabel(f"Tx {tx}, Rx {rx}\nk-Wave")
        ax2 = ax.twinx()
        ax2.plot(t*1e6, b, color="tab:orange", lw=.8, alpha=.8, label="Deepwave raw (right)")
        ax2.set_ylabel("Deepwave raw")
        axes[row, 1].plot(t*1e6, a, lw=.8, label="k-Wave")
        axes[row, 1].plot(t*1e6, c, lw=.8, alpha=.8, label="Deepwave global gain + lag")
    axes[0, 0].set_title("Raw values: independent vertical axes, no scaling")
    axes[0, 1].set_title("One global gain and lag: shared vertical axis")
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Acquisition time (us)")
    fig.tight_layout()
    fig.savefig(run/"deepwave_trace_overlays.png", dpi=cfg["figure_dpi"])
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    tx = cfg["bscan_transmitter_one_based"] - 1
    for ax, data, title in zip(axes, [ref[tx], raw[tx], aligned[tx]], ["k-Wave", "Deepwave raw", "Deepwave globally aligned"]):
        limit = np.max(np.abs(data))
        im = ax.imshow(data, aspect="auto", extent=[t[0]*1e6, t[-1]*1e6, 64, 1], cmap="seismic", vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=ax, shrink=.75)
        ax.set_title(title)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Receiver")
    fig.suptitle(f"Tx {tx+1}; each color scale shows its own numerical amplitude")
    fig.tight_layout()
    fig.savefig(run/"deepwave_bscan_comparison.png", dpi=cfg["figure_dpi"])
    plt.close(fig)


def main(cfgpath, stage, run):
    start = time.perf_counter()
    cfg = json.loads(Path(cfgpath).read_text(encoding="utf-8-sig"))
    run = Path(run)
    dump(run/"config.json", cfg)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    info = {"stage": stage, "python": sys.version, "executable": sys.executable,
            "versions": {n: importlib.metadata.version(n) for n in ["torch","deepwave","numpy","h5py","matplotlib"]},
            "cuda_runtime": torch.version.cuda, "code_sha256": sha(__file__), "config_sha256": sha(cfgpath)}
    assert torch.cuda.is_available(), "CUDA unavailable; do not install or compile a fallback"
    info["gpu"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
    info["torch_arch_list"] = torch.cuda.get_arch_list()
    info["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    print(json.dumps(info, ensure_ascii=False), flush=True)
    dump(run/"environment.json", info)
    assert sha(cfg["reference"]) == cfg["reference_sha256"]
    with h5py.File(cfg["reference"], "r") as f:
        wave = np.asarray(f["source_wavelet"]).reshape(-1).astype(np.float32)
        c = np.asarray(f["c"]).T.astype(np.float32).copy()
        t = np.asarray(f["t_output"]).reshape(-1)
        if stage == "full":
            ref = np.asarray(f["fmc"]).transpose(2,1,0).copy()
            raw_ref = np.asarray(f["fmc_internal"]).transpose(2,1,0)
            assert np.array_equal(ref, raw_ref[..., ::cfg["output_stride"]])
            del raw_ref
    assert wave.shape == (cfg["nt"],) and c.shape == tuple(cfg["simulation_shape"])
    monitor = Monitor(cfg["memory_poll_s"])
    monitor.worker.start()
    torch.cuda.reset_peak_memory_stats()
    try:
        if stage == "smoke":
            sc = cfg["smoke"]
            v = torch.full(sc["shape"], sc["velocity"], dtype=torch.float32, device="cuda")
            src = torch.tensor([[sc["source"]]], dtype=torch.long, device="cuda")
            rec = torch.tensor([sc["receivers"]], dtype=torch.long, device="cuda")
            wav = torch.tensor(wave[:sc["nt"]], device="cuda")[None,None,:]
            torch.cuda.synchronize()
            st = time.perf_counter()
            y = propagate(cfg, v, wav, src, rec)
            torch.cuda.synchronize()
            info["propagation_s"] = time.perf_counter()-st
            arr = y.cpu().numpy()
            assert np.isfinite(arr).all() and np.max(np.abs(arr)) > 0, "Nonfinite or zero smoke output"
            info["finite"] = True
            info["max_abs"] = float(np.max(np.abs(arr)))
            np.save(run/"smoke.npy", arr)
        else:
            v = torch.tensor(c, device="cuda")
            ac = cfg["array"]
            loc = torch.tensor([[ac["depth_index"], ac["first_lateral_index"]+j*ac["pitch_cells"]] for j in range(ac["count"])], dtype=torch.long, device="cuda")
            batch = cfg["batch_shots"]
            attempts = []
            while True:
                try:
                    parts = []
                    torch.cuda.synchronize()
                    st = time.perf_counter()
                    for j in range(0, ac["count"], batch):
                        n = min(batch, ac["count"]-j)
                        wav = torch.tensor(wave, device="cuda")[None,None,:].repeat(n,1,1)
                        y = propagate(cfg, v, wav, loc[j:j+n,None,:].contiguous(), loc[None,:,:].repeat(n,1,1))
                        parts.append(y.cpu().numpy().copy())
                        del y
                    torch.cuda.synchronize()
                    info["propagation_and_copy_s"] = time.perf_counter()-st
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    attempts.append({"batch": batch, "elapsed_s": time.perf_counter()-st, "error": str(exc)})
                    dump(run/"oom_attempts.json", attempts)
                    if batch == cfg["oom_fallback_shots"]:
                        raise
                    parts = []
                    torch.cuda.empty_cache()
                    batch = cfg["oom_fallback_shots"]
            arr = np.concatenate(parts, axis=0)
            assert arr.shape == (ac["count"],ac["count"],cfg["nt"])
            assert np.isfinite(arr).all() and np.max(np.abs(arr)) > 0, "Nonfinite or zero full output"
            info["batch_shots_used"] = batch
            info["oom_attempts"] = attempts
            raw = arr[..., ::cfg["output_stride"]].copy()
            with h5py.File(run/"deepwave_fmc.h5", "x") as f:
                f.create_dataset("fmc_internal", data=arr)
                f.create_dataset("fmc", data=raw)
                f.create_dataset("c", data=c)
                f.create_dataset("source_wavelet", data=wave)
                f.create_dataset("t_output", data=t)
                f.attrs["axis_order"] = "transmitter,receiver,time"
            info["generation_including_input_and_save_s"] = time.perf_counter()-start
            info["script_start_through_data_save_s"] = time.perf_counter()-SCRIPT_START
            info["output_bytes"] = (run/"deepwave_fmc.h5").stat().st_size
        info["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        info["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    finally:
        monitor.stop.set()
        monitor.worker.join(timeout=6)
        dump(run/"gpu_samples.json", monitor.rows)
    info["whole_gpu_peak_mib"] = max((row[1] for row in monitor.rows), default=None)
    info["gpu_utilization_mean"] = float(np.mean([row[2] for row in monitor.rows])) if monitor.rows else None
    dump(run/"timing.json", info)
    if stage == "full":
        raw_metrics, raw_l2, raw_corr = metrics(ref, raw)
        aligned, lag, gain, peak_corr, peak_lag = align(ref, raw, cfg["alignment"]["max_lag_output_samples"])
        aligned_metrics, aligned_l2, aligned_corr = metrics(ref, aligned)
        result = {"raw": raw_metrics, "aligned": aligned_metrics,
                  "global_gain": gain, "global_lag_output_samples": lag,
                  "global_lag_s": lag*cfg["dt_s"]*cfg["output_stride"],
                  "raw_peak_absolute_xcorr_signed": stats(peak_corr),
                  "raw_peak_xcorr_lag_samples": stats(peak_lag),
                  "deepwave_reciprocity": reciprocity(raw), "kwave_reciprocity": reciprocity(ref)}
        dump(run/"metrics.json", result)
        with h5py.File(run/"comparison.h5", "x") as f:
            for key, val in {"aligned_fmc":aligned, "raw_trace_l2":raw_l2, "raw_trace_pearson":raw_corr,
                             "aligned_trace_l2":aligned_l2, "aligned_trace_pearson":aligned_corr,
                             "raw_trace_peak_xcorr":peak_corr, "raw_trace_peak_lag":peak_lag}.items():
                f.create_dataset(key, data=val)
        plot(cfg, run, t, ref, raw, aligned)
        print(json.dumps(result, indent=2), flush=True)
    print(json.dumps(info, indent=2), flush=True)
    dump(run/"success.json", {"stage":stage,"success":True,"wall_s":time.perf_counter()-start})


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])
    except Exception:
        failure = traceback.format_exc()
        Path(sys.argv[3], "failure.txt").write_text(failure, encoding="utf-8")
        raise

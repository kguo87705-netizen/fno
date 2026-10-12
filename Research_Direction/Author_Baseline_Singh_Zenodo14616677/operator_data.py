"""Workpiece-grouped pilot loader and differentiable ordinary-FNO input layout."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class OperatorPilotDataset(Dataset):
    def __init__(self, folder, split):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        self.entries = [(record, shot) for record in self.manifest["records"] if record["split"] == split for shot in range(record["traces_shape"][0])]
        self.scale = self.manifest["normalization"]["wavefield_rms"]
        if not self.scale > 0:
            raise ValueError("Invalid training normalization")
        self._cached_file, self._cached_data = None, None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        record, shot = self.entries[index]
        if self._cached_file != record["file"]:
            self._cached_data = torch.load(self.folder / record["file"], weights_only=True, map_location="cpu")
            self._cached_file = record["file"]
        data = self._cached_data
        return {
            "gamma": data["gamma"], "target": data["wavefield_snapshots"][shot:shot+1] / self.scale,
            "source_xy_m": data["source_xy_m"][shot], "x_m": data["x_m"], "y_m": data["y_m"],
            "time_seconds": data["time_seconds_snapshots"], "receiver_indices": data["receiver_physical_indices"],
            "traces_full": data["traces_full"][shot], "snapshot_trace_indices": data["snapshot_trace_indices"],
            "group_id": record["group_id"], "split": record["split"], "shot": shot, "wavefield_scale": self.scale,
        }


def encode_fno_input(sample, gamma=None):
    """Six channels [gamma,x/Lx,y/Ly,t/T,sx/Lx,sy/Ly], shape [C,X,Y,T]."""
    if gamma is None:
        gamma = sample["gamma"]
    nx, ny = gamma.shape
    nt = sample["time_seconds"].numel()
    x = sample["x_m"].to(gamma)
    y = sample["y_m"].to(gamma)
    t = sample["time_seconds"].to(gamma)
    s = sample["source_xy_m"].to(gamma)
    assert x.numel() == nx and y.numel() == ny
    shape = (nx, ny, nt)
    channels = [gamma[...,None].expand(shape), (x/x[-1])[:,None,None].expand(shape), (y/y[-1])[None,:,None].expand(shape), (t/t[-1])[None,None,:].expand(shape), (s[0]/x[-1]).expand(shape), (s[1]/y[-1]).expand(shape)]
    return torch.stack(channels)


def sample_receivers(normalized_field, sample):
    """Physical scalar traces at stored times; no temporal interpolation."""
    indices = sample["receiver_indices"].to(normalized_field.device)
    return normalized_field[..., indices[:,0], indices[:,1], :] * sample["wavefield_scale"]

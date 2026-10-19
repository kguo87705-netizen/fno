"""Recreate the confirmed label from official EMNIST IDX gzip files.

Requires numpy and Pillow in an E-drive environment. Never overwrites a
different existing label. No download or installation is performed here.
"""
import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    root = Path(cfg["paths"]["root"]).resolve()
    target = Path(cfg["paths"]["label_npy"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Output must remain under the configured E-drive project")
    model = cfg["model"]
    index = model["sample_index_zero_based"]
    with gzip.open(cfg["paths"]["emnist_images"], "rb") as stream:
        magic, count, height, width = struct.unpack(">4I", stream.read(16))
        if magic != 2051 or (height, width) != (28, 28) or not 0 <= index < count:
            raise ValueError("Unexpected EMNIST IDX header or index")
        stream.seek(16 + index * height * width)
        raw = np.frombuffer(stream.read(height * width), dtype=np.uint8)
    upright = raw.reshape(height, width).T.copy()
    resized = np.asarray(Image.fromarray(upright).resize(
        tuple(reversed(model["resize_shape"])), Image.Resampling.BILINEAR))
    canvas = np.zeros(model["label_shape"], dtype=np.float32)
    top, left = (np.array(canvas.shape) - np.array(resized.shape)) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    label = model["c_background_m_s"] - (
        model["c_background_m_s"] - model["c_defect_m_s"]) * canvas / 255
    if target.exists():
        if not np.array_equal(np.load(target, allow_pickle=False), label):
            raise FileExistsError("Existing label differs; refusing overwrite")
        print("Existing label matches exactly; no file modified:", target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.save(stream, label, allow_pickle=False)
        print("Created:", target)


if __name__ == "__main__":
    main()

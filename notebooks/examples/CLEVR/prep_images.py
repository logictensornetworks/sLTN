"""Resize the CLEVR images once, for the slot model: the full 320 x 480 frame to 128 x 192 (aspect kept), RGB.

Output: notebooks/datasets/CLEVR_images128/<split>.npy, uint8 [scenes, 128, 192, 3], in the scene file's order.
Needs the images of the full CLEVR zip (18 GB) in notebooks/datasets/CLEVR_v1.0/images.

Run: python prep_images.py [--splits train val] [--workers 8]
"""

import argparse
import io
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parents[2] / "datasets" / "CLEVR_v1.0"
OUT = Path(__file__).resolve().parents[2] / "datasets" / "CLEVR_images128"
H, W = 128, 192


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR))


def load_bytes(data):
    """The same, from a PNG's bytes (slots_dh_job.py reads them from CLEVR's zip, in worker processes)."""
    return load(io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        scenes = json.load(open(DATA / "scenes" / f"CLEVR_{split}_scenes.json"))["scenes"]
        paths = [DATA / "images" / split / s["image_filename"] for s in scenes]
        arr = np.lib.format.open_memmap(OUT / f"{split}.npy", mode="w+", dtype=np.uint8, shape=(len(paths), H, W, 3))
        with Pool(a.workers) as pool:
            for i, im in enumerate(pool.imap(load, paths, chunksize=64)):
                arr[i] = im
                if i % 10000 == 0:
                    print(f"{split}: {i}/{len(paths)}", flush=True)
        arr.flush()
        print(f"{split}: wrote {OUT / f'{split}.npy'} {arr.shape}", flush=True)


if __name__ == "__main__":
    main()

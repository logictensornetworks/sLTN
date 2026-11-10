"""Cut one crop per object from the CLEVR images, once, for E1.

Each object is located by its `pixel_coords` (x, y, depth) from the scene file; the crop is a square of
half-side K / depth pixels (so objects have about the same scale whatever their distance; a large object
fills the crop, a small one about half of it), resized to R x R. Padded positions are left black.

Output: notebooks/datasets/CLEVR_crops/<split>_crops.npy, uint8 [scenes, O = 10, R, R, 3], in the scene file's order.

Needs the images of the full CLEVR zip (18 GB) in notebooks/datasets/CLEVR_v1.0/images.

Run: python prep_crops.py [--splits train val] [--workers 8]
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parents[2] / "datasets" / "CLEVR_v1.0"
OUT = Path(__file__).resolve().parents[2] / "datasets" / "CLEVR_crops"
K, R, O = 500, 48, 10


def crops_of(args):
    path, objects = args
    im = Image.open(path).convert("RGB")
    out = np.zeros((O, R, R, 3), np.uint8)
    for i, ob in enumerate(objects):
        x, y, d = ob["pixel_coords"]
        h = K / d
        out[i] = np.asarray(im.crop((x - h, y - h, x + h, y + h)).resize((R, R), Image.BILINEAR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        scenes = json.load(open(DATA / "scenes" / f"CLEVR_{split}_scenes.json"))["scenes"]
        jobs = [(DATA / "images" / split / s["image_filename"], s["objects"]) for s in scenes]
        arr = np.lib.format.open_memmap(OUT / f"{split}_crops.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(scenes), O, R, R, 3))
        with Pool(a.workers) as pool:
            for i, c in enumerate(pool.imap(crops_of, jobs, chunksize=64)):
                arr[i] = c
                if i % 10000 == 0:
                    print(f"{split}: {i}/{len(scenes)}", flush=True)
        arr.flush()
        print(f"{split}: wrote {OUT / f'{split}_crops.npy'} {arr.shape}", flush=True)


if __name__ == "__main__":
    main()

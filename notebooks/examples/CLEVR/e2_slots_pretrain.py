"""E2, step 1: pretrain the slot model (slots.py) on the CLEVR training images by reconstruction. No labels.

The paper's object-discovery recipe: Adam, learning rate 4e-4 with a linear warm-up over 10k steps and a
decay by 0.5 every 100k steps, batch 64, pixel MSE between the image and its reconstruction (images in
[-1, 1]). Here on the full frame (128 x 192, prep_images.py) with K = 11 slots.

Every --ckpt_every steps it saves results/e2_slots/<name>/ckpt.pt (or --results/<name>) (model, optimiser, step) and the MSE on
256 validation images; `--resume <name>` continues from there, so a stopped workspace loses little.

Run (GPU): python e2_slots_pretrain.py --name k11 [--steps 500000] [--amp]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clevr_sltn import RESULTS, sltn_version
from slots import K, SlotAutoEncoder

IMAGES = Path(__file__).resolve().parents[2] / "datasets" / "CLEVR_images128"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at(step, base, warmup, decay_steps, decay_rate):
    return base * min(1.0, step / warmup) * decay_rate ** (step / decay_steps)


@torch.no_grad()
def val_mse(model, val, amp):
    model.eval()
    errs = []
    for s in range(0, len(val), 32):
        x = model.to_input(val[s:s + 32].to(DEVICE))
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        errs.append(F.mse_loss(out["recon"].float(), x, reduction="none").mean(dim=(1, 2, 3)))
    model.train()
    return torch.cat(errs).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", action="store_true", help="continue results/e2_slots/<name>/ckpt.pt")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--decay_steps", type=int, default=100_000)
    ap.add_argument("--decay_rate", type=float, default=0.5)
    ap.add_argument("--slots", type=int, default=K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast (less memory, faster)")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=5_000)
    ap.add_argument("--images", default=str(IMAGES), help="folder with train.npy and val.npy")
    ap.add_argument("--results", default=str(RESULTS / "e2_slots"), help="the run folder <results>/<name> goes here")
    a = ap.parse_args()

    out = Path(a.results) / a.name
    out.mkdir(parents=True, exist_ok=True)
    train = torch.from_numpy(np.load(Path(a.images) / "train.npy"))            # uint8 [S, 128, 192, 3], in RAM
    val = torch.from_numpy(np.load(Path(a.images) / "val.npy")[:256])
    torch.manual_seed(a.seed)
    model = SlotAutoEncoder(num_slots=a.slots).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(a.seed)
    step, history = 0, []
    if a.resume:
        ck = torch.load(out / "ckpt.pt", map_location=DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, history = ck["step"], ck["history"]
        g.set_state(ck["generator"]); torch.set_rng_state(ck["torch_rng"])
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(ck["cuda_rng"])
        print(f"resumed at step {step}", flush=True)
    config = dict(vars(a), n_train=len(train), device=str(DEVICE),
                  params=sum(p.numel() for p in model.parameters()))
    version_file = Path(__file__).resolve().parents[3] / "VERSION.json"     # on the cluster (no git): the launcher's
    version = json.loads(version_file.read_text()) if version_file.exists() else sltn_version()
    (out / "config.json").write_text(json.dumps(dict(config=config, sltn=version), indent=1))
    print(config, flush=True)

    t0, run_loss = time.perf_counter(), 0.0
    while step < a.steps:
        step += 1
        for group in opt.param_groups:
            group["lr"] = lr_at(step, a.lr, a.warmup, a.decay_steps, a.decay_rate)
        x = model.to_input(train[torch.randint(0, len(train), (a.batch,), generator=g)].to(DEVICE, non_blocking=True))
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=a.amp):
            out_ = model(x)
        loss = F.mse_loss(out_["recon"].float(), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if step % a.log_every == 0:
            ms = (time.perf_counter() - t0) / a.log_every * 1000
            print(f"step {step:7d} mse {run_loss / a.log_every:.5f} lr {opt.param_groups[0]['lr']:.2e} | {ms:.0f} ms/step",
                  flush=True)
            run_loss, t0 = 0.0, time.perf_counter()
        if step % a.ckpt_every == 0 or step == a.steps:
            history.append(dict(step=step, val_mse=val_mse(model, val, a.amp)))
            print(f"step {step:7d} val mse {history[-1]['val_mse']:.5f} (256 val images)", flush=True)
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, history=history,
                      generator=g.get_state(), torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None)
            torch.save(ck, out / "ckpt.tmp")
            (out / "ckpt.tmp").replace(out / "ckpt.pt")                       # never a half-written checkpoint
            (out / "history.json").write_text(json.dumps(history, indent=1))
    torch.save(model.state_dict(), out / "slot_model.pt")
    print(f"saved {out / 'slot_model.pt'}", flush=True)


if __name__ == "__main__":
    main()

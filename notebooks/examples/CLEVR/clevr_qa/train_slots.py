"""Learn the objects, their attributes and their relations from whole CLEVR images and the answers only.

No crops, no pixel coordinates, no object count: a slot model (slot_model.SlotAutoEncoder) splits each image
into K = 11 slots, and everything is trained together, from scratch, by one knowledge base:
  - the question rules of the families in the batch, with Present learned, so written as a conjunct:
    `exists o: (Present(scene[o]) & ...)` (formulas.compile_question(present="conjunct"));
  - the reconstruction axiom, "the slots explain the image":
    forall (img, rec): forall p: Eq(rec[p], img[p]),   Eq(a, b) = exp(-alpha * ||a - b||^2)
  - with --dup_weight > 0, the duplicate axiom, "two present slots are not at the same place"
    (formulas.DUP_AXIOM), on where each slot's attention peaks (attention_peaks, detached).
With --counting, the counting questions too ("how many ...?", "are there more A than B?"): their counts
are computed from the sets' values per slot (counting.py), the only questions that say a slot is NOT an object.
loss = 1 - ((1 - w) * rules + w * axiom) (+ the duplicate axiom with its weight), where rules is the families' truth values weighted by how many of
their questions are in the batch, and w = --recon_weight (0 = answers only).

scene[o] for slot o = [0, centre x, centre y, area, emb]: the centre and area come from the slot's alpha
mask (slot_model.mask_centroids), emb = an MLP of the slot vector. C, Z, M, S, SameC & co. read emb through
one linear head per attribute (networks.Has, Same), Present a sigmoid head on emb, R the MLP of
networks.Relation on (centre x, centre y, area) of both slots, antisymmetric by default (--relation): R(o, o1, left)
and R(o1, o, left) add up to 1, and right is left with o and o1 swapped. The leading 0 only keeps scene[o]'s layout
(vocabulary.py); the presence bit isn't used.

Each step: B images, and all their in-scope questions, grouped by family. The scene files are used only to
score (evaluate): each object is matched to a slot (--match: by centre, Hungarian; "paint", the slot whose
mask is the largest at the object's centre pixel; "attention", the slot that wins that pixel in the last round of
slot attention), then concept, relation and Present accuracy are read on the matched slots; QA accuracy needs
no matching.

Needs the 128 x 192 images (../prep_images.py, or the artifact clevr-images128 on the cluster).
Run (from notebooks/examples/CLEVR; GPU):
    python -m clevr_qa.train_slots [--steps 100000] [--recon_weight 0.5] [--amp]
Several GPUs: one process per GPU, each with its own --batch images; the gradients are averaged after each
backward (one all-reduce; not DDP's wrapper, because sLTN calls the heads and R directly):
    torchrun --nproc_per_node 4 -m clevr_qa.train_slots --run_dir <folder> ...
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

import sltn
from sltn.interpretation.operators.aggregator import AggregPMeanError

from .batching import family_tensors, question_type
from .counting import ground_counts
from .data import IMAGES, RESULTS, load, load_images, scene_arrays, subset
from .formulas import DUP_AXIOM, RECON_AXIOM, make_signature
from .interpretation import base_interpretation, per_question
from .networks import Has, Relation, Same, dot
from .slot_model import K, RES, SlotAutoEncoder, mask_centroids
from .train import sltn_version
from .vocabulary import APPEARANCE, ATTRS, DIRS, POS, PRED, PRESENT, SAME, VALUES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PIX = RES[0] * RES[1]


class SlotConcepts(nn.Module):
    """Slot vector -> emb (an MLP); emb -> a softmax per attribute (heads) and Present (a sigmoid)."""

    def __init__(self, slot_dim=64, width=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(slot_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.heads = nn.ModuleDict({a: nn.Linear(width, len(VALUES[a])) for a in ATTRS})
        self.present = nn.Linear(width, 1)

    @property
    def dim(self):
        return self.present.in_features


class PresentHead(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return torch.sigmoid(self.linear(x[..., APPEARANCE]))


def make_interpretation(sig, concepts, relation, alpha, device, near_sigma=0.03):
    it = base_interpretation(sig, concepts.dim, device, n_slots=K)
    it["Present"] = PresentHead(concepts.present)
    it["R"] = relation
    for a in ATTRS:
        it[PRED[a]] = Has(concepts.heads[a])
        it[SAME[a]] = Same(concepts.heads[a])
    it["Pix"] = range(N_PIX)
    it["Rgb"] = sltn.Type("Rgb", shape=(3,))
    it["Eq"] = lambda a, b: torch.exp(-alpha * ((a - b) ** 2).sum(-1, keepdim=True))
    it.logic["forall,Obj"] = AggregPMeanError(p=2)                  # DUP_AXIOM's forall over pairs of slots
    it["Place"] = sltn.Type("Place", shape=(2,))
    it["Near"] = lambda a, b: torch.exp(-((a - b) ** 2).sum(-1, keepdim=True) / near_sigma ** 2)
    return it


def attention_peaks(attn, power=8):
    """attn [B, H*W, K] (slot attention's last iteration) -> [B, K, 2]: the centroid (x, y) in [0, 1] of each
    slot's attention raised to `power`, which keeps only where it peaks. Detached: no gradient reaches the
    slot model through it."""
    w = attn.detach().float() ** power
    w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    ys = (torch.arange(RES[0], device=w.device, dtype=w.dtype) + 0.5) / RES[0]
    xs = (torch.arange(RES[1], device=w.device, dtype=w.dtype) + 0.5) / RES[1]
    grid = torch.stack([xs.repeat(RES[0]), ys.repeat_interleave(RES[1])], -1)    # [H*W, 2], row-major
    return torch.einsum("bnk,nc->bkc", w, grid)


def perceive(model, concepts, images, amp, owner=False, peaks=False):
    """uint8 images [B, H, W, 3] -> scene features [B, K, 4 + emb], image and reconstruction [B, H*W, 3]
    (in [-1, 1]), slot centres [B, K, 2]; with owner, also the slot that paints each pixel [B, H, W]
    (the argmax of the masks) and the slot that wins it in slot attention's last round [B, H, W] (the argmax
    of the attention); with peaks, also attention_peaks [B, K, 2] (last)."""
    x = model.to_input(images)
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp):
        out = model(x)
    masks = out["masks"].float()
    centre, area = mask_centroids(masks)                                   # [B, K, 2], [B, K]
    emb = concepts.mlp(out["slots"].float())
    feats = torch.cat([torch.zeros_like(area)[..., None], centre, area[..., None], emb], -1)
    img = x.permute(0, 2, 3, 1).reshape(len(x), N_PIX, 3)
    rec = out["recon"].float().permute(0, 2, 3, 1).reshape(len(x), N_PIX, 3)
    wins = lambda: out["attn"].float().argmax(-1).reshape(len(x), *RES)
    extra = ((masks[:, :, 0].argmax(1), wins()) if owner else ()) + ((attention_peaks(out["attn"]),) if peaks else ())
    return (feats, img, rec, centre) + extra


def ground_family(it, c, t, sel, feats, rows, counts=()):
    """Family c's questions sel; their scenes are rows `rows[scene_idx]` of feats. counts: the parsed
    Count.formula of a counting family (its counts cnt0, cnt1 are grounded from them)."""
    it["scene"] = feats[rows[t["scene_idx"][sel]]]
    for name in c.variables[1:] + [c.answer]:
        it[name] = t[name][sel]
    if counts:
        ground_counts(it, c, counts)


def load_split(split, images_dir, max_scenes=None, n_questions=None, seed=0, counting=False):
    scenes, questions = load(split)
    if max_scenes:
        scenes = scenes[:max_scenes]
        questions = [q for q in questions if q["image_index"] < max_scenes]
    if n_questions:
        questions = subset(questions, n_questions, seed)
    out = {k: v.to(DEVICE) for k, v in scene_arrays(scenes).items()}
    out["images"] = load_images(split, len(scenes), images_dir).to(DEVICE)
    out["fams"] = family_tensors(questions, DEVICE, present="conjunct", counting=counting)
    return out


ORACLES = {"P": "present", "C": "concepts", "R": "relations"}


def oracle_scene(feats, concepts, match, matched, split, S, oracle):
    """scene rows for an oracle QA evaluation: [Present, POS (3), scene index, object index (-1: no object),
    one distribution per attribute (ATTRS order), zeros up to the width of scene[o]]. On the slots matched to an object, oracle "present" puts
    Present = 1 (0 on the others) and "concepts" the one-hot true attributes; elsewhere the learned values."""
    app = feats[..., APPEARANCE]
    pres = split["present"][:S]
    sidx = torch.arange(S, device=feats.device)[:, None].expand_as(match)
    present = matched.float() if "present" in oracle else torch.sigmoid(concepts.present(app)).squeeze(-1)
    oidx = torch.full(matched.shape, -1.0, device=feats.device)
    oidx[sidx[pres], match[pres]] = torch.arange(pres.shape[1], device=feats.device).expand_as(pres)[pres].float()
    dists = []
    for j, a in enumerate(ATTRS):
        d = torch.softmax(concepts.heads[a](app), -1)
        if "concepts" in oracle:
            truth = nn.functional.one_hot(split["attr"][:S][..., j].clamp_min(0), len(VALUES[a])).float()
            d[sidx[pres], match[pres]] = truth[pres]
        dists.append(d)
    scene_col = torch.arange(S, device=feats.device, dtype=torch.float)[:, None].expand_as(present)
    rows = torch.cat([present[..., None], feats[..., POS], scene_col[..., None], oidx[..., None], *dists], -1)
    return nn.functional.pad(rows, (0, feats.shape[-1] - rows.shape[-1]))   # the sort Object's width


def oracle_interpretation(it, relation, rel, oracle):
    """The entries of `it` that read oracle_scene's rows; returns the old entries, to put back. R: with oracle
    "relations", CLEVR's annotation on two slots matched to objects, the learned R elsewhere."""
    old = {n: it[n] for n in ["Present", "R", *PRED.values(), *SAME.values()]}
    it["Present"] = lambda x: x[..., 0:1]
    start = 6
    for a in ATTRS:
        sl = slice(start, start + len(VALUES[a]))
        it[PRED[a]] = (lambda sl: lambda x, v: dot(x[..., sl], v))(sl)
        it[SAME[a]] = (lambda sl: lambda x, x1: dot(x[..., sl], x1[..., sl]))(sl)
        start = sl.stop

    def R(x, x1, r):
        learned = relation(x, x1, r)
        if "relations" not in oracle:
            return learned
        s, o, o1 = torch.broadcast_tensors(x[..., 4].long(), x[..., 5].long(), x1[..., 5].long())
        truth = (rel[s, :, o.clamp_min(0), o1.clamp_min(0)] * r).sum(-1, keepdim=True)     # [..., 4] . r
        return torch.where(((o >= 0) & (o1 >= 0))[..., None], truth, learned)
    it["R"] = R
    return old


@torch.no_grad()
def evaluate(model, concepts, relation, it, sig, split, n_scenes, amp, chunk=64, oracle=(), match_by="centroid"):
    """Scores on the first n_scenes of split. The annotations are used here only.
    match_by: how each object gets its slot for the concept, Present and relation scores (and the oracles):
    "centroid" (Hungarian on the distance between the object's centre and the masks' soft centroids) or
    "paint" (the slot whose mask is the largest at the object's centre pixel; two objects can share one) or
    "attention" (the slot that wins the object's centre pixel in slot attention's last round; two objects can
    share one). "attention" does not use the decoder: for recon_weight 0, where nothing trains it to paint objects.
    oracle: a subset of ORACLES' values; the QA formulas then read the truth on the slots matched to an object:
    "present" (Present = 1 on them, 0 on the other slots), "concepts" (their true attributes), "relations"
    (CLEVR's relations between them). A check of how much of the QA error each part causes; the other scores
    don't use the oracles."""
    model.eval()
    S = min(n_scenes, len(split["images"]))
    feats, centres, paint, attend, mse = [], [], [], [], 0.0
    for s0 in range(0, S, chunk):
        f, img, rec, c, own, win = perceive(model, concepts, split["images"][s0:min(S, s0 + chunk)], amp, owner=True)
        feats.append(f)
        centres.append(c)
        px = split["pixel"][s0:s0 + len(f), :, :2].to(own.device)                 # [b, objects, 2]: column, row
        col = (px[..., 0] * RES[1]).long().clamp(0, RES[1] - 1)
        row = (px[..., 1] * RES[0]).long().clamp(0, RES[0] - 1)
        paint.append(own[torch.arange(len(f), device=own.device)[:, None], row, col].cpu())
        attend.append(win[torch.arange(len(f), device=win.device)[:, None], row, col].cpu())
        mse += float(((rec - img) ** 2).mean(dim=(1, 2)).sum())
    feats, centres, paint, attend = torch.cat(feats), torch.cat(centres), torch.cat(paint), torch.cat(attend)
    res = {"recon_mse": mse / S}

    # match every object to a slot by centre: the annotations' pixel x, y vs the masks' centroids
    pres, pixel, attr, rel = (split[k][:S] for k in ("present", "pixel", "attr", "rel"))
    match = torch.full((S, pres.shape[1]), -1, dtype=torch.long)
    dist = torch.cdist(pixel[..., :2], centres).cpu()                     # [S, objects, K], in image widths/heights
    d_all = []
    for s in range(S):
        n = int(pres[s].sum())
        r, c = linear_sum_assignment(dist[s, :n].numpy())
        match[s, r] = torch.from_numpy(c)
        d_all.append(dist[s, r, c])
    obj, oc = pres, pres.cpu()
    res["match/paint_is_centroid"] = float((paint[oc] == match[oc]).float().mean())
    res["match/attention_is_paint"] = float((attend[oc] == paint[oc]).float().mean())
    if match_by in ("paint", "attention"):
        match = torch.where(oc, paint if match_by == "paint" else attend, torch.full_like(paint, -1))
        d_all = [dist[s].gather(1, match[s].clamp_min(0)[:, None])[:, 0][oc[s]] for s in range(S)]
        taken = torch.zeros(S, K, dtype=torch.long).scatter_add_(1, match.clamp_min(0), oc.long())
        res["match/objects_sharing_a_slot"] = float((taken.gather(1, match.clamp_min(0)) > 1)[oc].float().mean())
    match = match.to(feats.device)
    res["match_dist"] = float(torch.cat(d_all).mean())                    # mean centre error of the matched slots
    sidx = torch.arange(S, device=feats.device)[:, None].expand_as(match)
    mfeat = feats[sidx[obj], match[obj]]                                   # [objects, 4 + emb]
    for j, a in enumerate(ATTRS):
        pred = concepts.heads[a](mfeat[:, APPEARANCE]).argmax(-1)
        res[f"concept_acc/{a}"] = float((pred == attr[:S][obj][:, j]).float().mean())
    p = torch.sigmoid(concepts.present(feats[..., APPEARANCE])).squeeze(-1) > 0.5   # [S, K]
    matched = torch.zeros(S, K, dtype=torch.bool, device=feats.device)
    matched[sidx[obj], match[obj]] = True
    res["present/objects_found"] = float(p[matched].float().mean())          # matched slots said present
    res["present/empty_rejected"] = float((~p[~matched]).float().mean())    # other slots said absent
    res["present/count_acc"] = float((p.sum(1) == obj.sum(1)).float().mean())
    if match_by in ("paint", "attention"):
        # why the number of present slots is wrong: merged objects (two objects, one painting slot) and extra
        # present slots (present, but painting no object's centre: duplicates, fragments, or the hidden slot of a
        # merged object)
        n_pres, n_obj = p.sum(1).cpu(), obj.sum(1).cpu()
        merged = (matched.sum(1).cpu() < n_obj)
        extra = (p & ~matched).any(1).cpu()
        res["present/count_over"] = float((n_pres > n_obj).float().mean())
        res["present/count_under"] = float((n_pres < n_obj).float().mean())
        for mg in (0, 1):
            for ex in (0, 1):
                sel = (merged == bool(mg)) & (extra == bool(ex))
                for name, ok in (("under", n_pres < n_obj), ("exact", n_pres == n_obj), ("over", n_pres > n_obj)):
                    res[f"count_split/merged{mg}_extra{ex}_{name}"] = float((sel & ok).float().mean())
    # relations on the matched pairs of distinct objects
    O = pres.shape[1]
    hits, n = torch.zeros(len(DIRS)), 0
    for s0 in range(0, S, 512):
        sl = slice(s0, min(S, s0 + 512))
        m = match[sl].clamp_min(0)
        pos = torch.gather(feats[sl], 1, m[..., None].expand(-1, -1, feats.shape[-1]))[..., 1:4]   # [s, O, 3]
        pr = relation.probs(pos[:, :, None].expand(-1, -1, O, -1), pos[:, None].expand(-1, O, -1, -1)) > 0.5
        pair = pres[sl][:, :, None] & pres[sl][:, None, :] & ~torch.eye(O, dtype=torch.bool, device=feats.device)
        hits += torch.stack([(pr[..., i] == rel[sl][:, i].bool())[pair].sum() for i in range(len(DIRS))]).cpu()
        n += int(pair.sum())
    for i, d in enumerate(DIRS):
        res[f"relation_acc/{d}"] = float(hits[i] / n)

    # QA: each question's reply from its family's `predict` formula
    rows = torch.arange(S, device=feats.device)
    old = {}
    if oracle:
        feats = oracle_scene(feats, concepts, match, matched, split, S, oracle)
        old = oracle_interpretation(it, relation, split["rel"][:S].to(feats.device), oracle)
    per = {}
    for key, (c, t) in split["fams"].items():
        keep = torch.nonzero(t["scene_idx"] < S).squeeze(1)
        phi, ok = sig.parse(c.predict), 0
        counts = [sig.parse(x.formula) for x in c.counts]
        for q0 in range(0, len(keep), 512):
            sel = keep[q0:q0 + 512]
            ground_family(it, c, t, sel, feats, rows, counts)
            v = per_question(it, phi)
            if c.kind == "yesno":
                ok += int(((v > 0.5) == (t["yes"][sel].squeeze(-1) > 0.5)).sum())
            else:
                ok += int((v.argmax(-1) == t[c.answer][sel].argmax(-1)).sum())
        typ = question_type(key, c)
        h_, n_ = per.get(typ, (0, 0))
        per[typ] = (h_ + ok, n_ + len(keep))
    res["qa_acc"] = sum(h_ for h_, _ in per.values()) / max(1, sum(n_ for _, n_ in per.values()))
    res.update({f"qa_acc/{k}": h_ / max(1, n_) for k, (h_, n_) in per.items()})
    for n, v in old.items():
        it[n] = v
    model.train()
    return res


def lr_at(step, base, warmup, decay_steps, decay_rate):
    return base * min(1.0, step / warmup) * decay_rate ** (step / decay_steps)


def setup_distributed():
    """(rank, world size). Under torchrun (WORLD_SIZE > 1): one process per GPU, NCCL; else (0, 1)."""
    global DEVICE
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        DEVICE = torch.device("cuda", local)
    backend = "nccl" if DEVICE.type == "cuda" else "gloo"          # gloo: CPU, for tests
    dist.init_process_group(backend, timeout=timedelta(hours=2))     # rank 0's evaluations can take minutes
    check = dist.new_group(backend=backend, timeout=timedelta(minutes=5))   # fail fast if the GPUs can't talk
    x = torch.ones(1, device=DEVICE)
    dist.all_reduce(x, group=check)
    assert x.item() == world, x
    print(f"rank {rank}: all-reduce across {world} processes OK", flush=True)
    return rank, world


def average_gradients(params, world):
    """One all-reduce for every gradient: each process's gradient -> the mean over the processes. A parameter
    with no gradient here (e.g. a head no question of this batch used) counts as zero."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat /= world
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p)
        i += n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64,
                    help="images per step and per GPU (all their questions are used); with N GPUs a step sees N x batch")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--decay_steps", type=int, default=100_000)
    ap.add_argument("--decay_rate", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient norm clipping (0: none)")
    ap.add_argument("--recon_weight", type=float, default=0.5, help="w: the axiom's weight (0 = answers only)")
    ap.add_argument("--alpha", type=float, default=5.0, help="Eq(a, b) = exp(-alpha ||a - b||^2)")
    ap.add_argument("--dup_weight", type=float, default=0.0,
                    help="weight of DUP_AXIOM (two present slots are not at the same place) in the loss; 0 = off")
    ap.add_argument("--near_sigma", type=float, default=0.03,
                    help="Near(a, b) = exp(-||a - b||^2 / sigma^2), positions in image widths/heights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_q", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the slot model")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=2_000)
    ap.add_argument("--eval_scenes", type=int, default=1_000)
    ap.add_argument("--ckpt_every", type=int, default=2_000)
    ap.add_argument("--resume", type=Path, default=None, help="a run folder with ckpt.pt")
    ap.add_argument("--relation", choices=["antisym", "free"], default="antisym",
                    help="R's MLP: antisymmetric (networks.Relation) or one free sigmoid per direction")
    ap.add_argument("--init", type=Path, default=None,
                    help="a ckpt.pt to start a new run from: its networks, optimiser state and step, except the "
                         "parts --fresh_relation / --fresh_present reset (ignored when the run resumes)")
    ap.add_argument("--fresh_relation", action="store_true",
                    help="with --init: R starts fresh (e.g. a saturated or a free R)")
    ap.add_argument("--run_dir", type=Path, default=None,
                    help="use this run folder (resumed if it holds ckpt.pt) instead of a new dated one")
    ap.add_argument("--counting", action="store_true",
                    help="also the counting questions (count, equal_integer, less_than, greater_than; counting.py)")
    ap.add_argument("--fresh_present", action="store_true",
                    help="with --init: Present's head starts fresh (its saturated sigmoid passes no gradient)")
    ap.add_argument("--eval_only", action="store_true",
                    help="no training: evaluate the --init checkpoint (R included) on --eval_scenes val scenes, once per "
                         "--oracles entry; writes eval_only.json in the run folder")
    ap.add_argument("--oracles", default="none,P",
                    help="with --eval_only: comma-separated oracle sets, each a combination of P (Present), C (concepts), "
                         "R (relations) or `none`, e.g. none,P,C,R,PC,PCR")
    ap.add_argument("--match", choices=["centroid", "paint", "attention"], default="centroid",
                    help="how objects get their slot for the concept/Present/relation scores "
                         "(see evaluate)")
    ap.add_argument("--images", type=Path, default=IMAGES, help="folder with train.npy and val.npy")
    ap.add_argument("--max_scenes", type=int, default=None, help="smoke tests: only the first N scenes per split")
    ap.add_argument("--results", type=Path, default=RESULTS, help="where the run folder goes")
    a = ap.parse_args(argv)
    rank, world = setup_distributed()
    main_proc = rank == 0
    say = (lambda m: print(m, flush=True)) if main_proc else (lambda m: None)
    if world > 1:
        assert a.run_dir or a.resume, "with several GPUs, give --run_dir (every process must know the folder)"
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)                                  # the same initial networks in every process
    g = torch.Generator().manual_seed(a.seed * 1000 + rank)    # but each process draws its own images
    tr = load_split("train", a.images, a.max_scenes, a.n_train_q, a.seed, a.counting)
    va = load_split("val", a.images, a.max_scenes, counting=a.counting) if main_proc else None      # only process 0 evaluates
    n_tr = sum(len(t["scene_idx"]) for _, t in tr["fams"].values())
    if main_proc:
        say(f"train: {len(tr['images'])} images, {n_tr} in-scope questions; val: {len(va['images'])} images; "
            f"{world} process(es) ({time.perf_counter() - t0:.0f}s)")

    sig = make_signature(recon=True)
    model = SlotAutoEncoder(num_slots=K).to(DEVICE)
    concepts = SlotConcepts().to(DEVICE)
    relation = Relation(torch.zeros(len(DIRS), 3), learn=True,                # reads (centre x, centre y, area)
                        antisymmetric=a.relation == "antisym").to(DEVICE)
    it = make_interpretation(sig, concepts, relation, a.alpha, DEVICE, a.near_sigma)
    params = [*model.parameters(), *concepts.parameters(), *relation.parameters()]
    opt = torch.optim.Adam(params, lr=a.lr)
    keys = list(tr["fams"])
    rules = {k: sig.parse(tr["fams"][k][0].rule) for k in keys}
    count_formulas = {k: [sig.parse(x.formula) for x in tr["fams"][k][0].counts] for k in keys}
    axiom = sig.parse(RECON_AXIOM)
    dup = sig.parse(DUP_AXIOM)
    with_q = torch.unique(torch.cat([tr["fams"][k][1]["scene_idx"] for k in keys])).cpu()   # images with questions
    rows = torch.full((len(tr["images"]),), -1, dtype=torch.long, device=DEVICE)

    step, history = 0, []
    if a.run_dir and (a.run_dir / "ckpt.pt").exists():
        a.resume = a.run_dir
    if a.resume:
        out = a.resume
        ck = torch.load(out / "ckpt.pt", map_location=DEVICE)
        for m, k in ((model, "model"), (concepts, "concepts"), (relation, "relation"), (opt, "opt")):
            m.load_state_dict(ck[k])
        step, history = ck["step"], ck["history"]
        g.manual_seed(a.seed * 1000 + rank + 7919 * step)       # fresh draws after a resume, per process
        say(f"resumed {out} at step {step}")
    else:
        tag = f"-w{a.recon_weight:g}" + (f"-n{a.n_train_q}" if a.n_train_q else "") + f"-seed{a.seed}"
        out = a.run_dir or a.results / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-slots" + tag)
        if main_proc:
            out.mkdir(parents=True, exist_ok=True)
        if a.init:
            ck = torch.load(a.init, map_location=DEVICE)
            model.load_state_dict(ck["model"])
            concepts.load_state_dict(ck["concepts"])
            opt.load_state_dict(ck["opt"])
            fresh = []                                          # the parts that start fresh, with no Adam state
            if a.fresh_relation and not a.eval_only:
                fresh.append(("R", relation.parameters()))
            else:
                relation.load_state_dict(ck["relation"])
            if a.fresh_present and not a.eval_only:
                concepts.present.reset_parameters()             # rank 0's draw is broadcast below
                fresh.append(("Present", concepts.present.parameters()))
            for _, ps in fresh:
                for p in ps:
                    opt.state.pop(p, None)
            step, history = ck["step"], ck["history"]
            g.manual_seed(a.seed * 1000 + rank + 7919 * step)
            say(f"initialised from {a.init} at step {step}; fresh: {', '.join(n for n, _ in fresh) or 'nothing'}")
    if world > 1:                                               # same start everywhere, whatever happened above
        for p in params:
            dist.broadcast(p.data, 0)
    config = dict({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, slots=K, res=list(RES),
                  n_train_inscope=n_tr, device=str(DEVICE), gpus=world, images_per_step=a.batch * world,
                  params=sum(p.numel() for p in params))
    say(f"sltn {sltn_version()['commit'][:7]} | {config}")
    if main_proc:
        (out / "config.json").write_text(json.dumps(dict(config=config, sltn=sltn_version()), indent=1))

    if a.eval_only:
        assert a.init and world == 1, "--eval_only needs --init and one process"
        out_res = {}
        for tag in a.oracles.split(","):
            oracle = () if tag == "none" else tuple(ORACLES[c] for c in tag)
            torch.manual_seed(a.seed)                       # the same initial slots in every evaluation
            r = evaluate(model, concepts, relation, it, sig, va, a.eval_scenes, a.amp, oracle=oracle, match_by=a.match)
            out_res[tag] = r
            say(f"step {step:6d} eval {tag} ({a.eval_scenes} val scenes): " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
        (out / "eval_only.json").write_text(json.dumps(dict(config=config, step=step, **out_res), indent=1))
        return

    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    zero = lambda: dict(loss=0.0, rules=0.0, axiom=0.0, dup=0.0, t_slots=0.0, t_rules=0.0, t_back=0.0)
    run, t1 = zero(), time.perf_counter()
    while step < a.steps:
        step += 1
        for group in opt.param_groups:
            group["lr"] = lr_at(step, a.lr, a.warmup, a.decay_steps, a.decay_rate)
        ta = time.perf_counter()
        batch = with_q[torch.randint(0, len(with_q), (a.batch,), generator=g)].to(DEVICE)
        rows[batch] = torch.arange(a.batch, device=DEVICE)
        feats, img, rec, _, place = perceive(model, concepts, tr["images"][batch], a.amp, peaks=True)
        sync()
        tb = time.perf_counter()
        sats, counts = [], []
        for k in keys:
            c, t = tr["fams"][k]
            sel = torch.nonzero(rows[t["scene_idx"]] >= 0).squeeze(1)
            if len(sel) == 0:
                continue
            ground_family(it, c, t, sel, feats, rows, count_formulas[k])
            sats.append(it(rules[k]).value.squeeze())
            counts.append(len(sel))
        rows[batch] = -1
        w = torch.tensor(counts, dtype=torch.float, device=DEVICE)
        rules_sat = (torch.stack(sats) * w).sum() / w.sum()
        it["img"], it["rec"] = img, rec
        axiom_sat = it(axiom).value.squeeze()
        loss = 1 - ((1 - a.recon_weight) * rules_sat + a.recon_weight * axiom_sat)
        it["scene"], it["spot"] = feats, place                # one row per image
        dup_sat = it(dup).value.squeeze()
        if a.dup_weight:
            loss = loss + a.dup_weight * (1 - dup_sat)
        sync()
        tc = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:
            average_gradients(params, world)
        if a.clip:
            torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        sync()
        td = time.perf_counter()
        run["t_slots"] += tb - ta
        run["t_rules"] += tc - tb
        run["t_back"] += td - tc
        run["loss"] += loss.item()
        run["rules"] += rules_sat.item()
        run["axiom"] += axiom_sat.item()
        run["dup"] += dup_sat.item()
        if step % a.log_every == 0 or step == 1:
            n = 1 if step == 1 else a.log_every
            if world > 1:                                           # the mean over the processes
                v = torch.tensor([run["loss"], run["rules"], run["axiom"], run["dup"]], device=DEVICE)
                dist.all_reduce(v)
                run["loss"], run["rules"], run["axiom"], run["dup"] = (v / world).tolist()
                check = torch.stack([p.detach().double().sum() for p in params]).sum().reshape(1)
                every = [torch.zeros_like(check) for _ in range(world)]
                dist.all_gather(every, check)                       # the processes must hold the same weights
                assert all(torch.equal(e, every[0]) for e in every), f"weights diverged: {torch.cat(every).tolist()}"
            ms = (time.perf_counter() - t1) / n * 1000
            say(f"step {step:6d} loss {run['loss'] / n:.4f} | rules {run['rules'] / n:.4f} axiom {run['axiom'] / n:.4f} dup {run['dup'] / n:.4f} "
                f"| lr {opt.param_groups[0]['lr']:.1e} | {ms:.0f} ms/step (slots {run['t_slots'] / n * 1000:.0f}, "
                f"formulas {run['t_rules'] / n * 1000:.0f}, backward {run['t_back'] / n * 1000:.0f})")
            run, t1 = zero(), time.perf_counter()
        if step % a.eval_every == 0 or step == a.steps:
            if main_proc:
                res = evaluate(model, concepts, relation, it, sig, va, a.eval_scenes, a.amp, match_by=a.match)
                history.append(dict(step=step, **res))
                say(f"step {step:6d} eval ({a.eval_scenes} val scenes): " + ", ".join(f"{k} {v:.3f}" for k, v in res.items()))
            if world > 1:
                dist.barrier()
            t1 = time.perf_counter()
        if (step % a.ckpt_every == 0 or step == a.steps) and main_proc:
            ck = dict(model=model.state_dict(), concepts=concepts.state_dict(), relation=relation.state_dict(),
                      opt=opt.state_dict(), step=step, history=history)
            torch.save(ck, out / "ckpt.tmp")
            (out / "ckpt.tmp").replace(out / "ckpt.pt")                      # never a half-written checkpoint
            (out / "history.json").write_text(json.dumps(history, indent=1))

    if main_proc:
        res = evaluate(model, concepts, relation, it, sig, va, len(va["images"]), a.amp, match_by=a.match)
        say("\nfinal (all val): " + ", ".join(f"{k} {v:.4f}" for k, v in res.items()))
        (out / "results.json").write_text(json.dumps(dict(config=config, sltn=sltn_version(), **res), indent=1))
        torch.save(dict(model=model.state_dict(), concepts=concepts.state_dict(), relation=relation.state_dict()),
                   out / "nets.pt")
        with torch.no_grad():                                                   # 8 val images, for pictures
            model.eval()
            o = model(model.to_input(va["images"][:8]))
            np.savez_compressed(out / "val_examples.npz", recon=o["recon"].float().cpu().numpy(),
                                rgb=o["rgb"].float().cpu().numpy(), masks=o["masks"].float().cpu().numpy())
        say(f"saved {out}; total {time.perf_counter() - t0:.0f}s")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

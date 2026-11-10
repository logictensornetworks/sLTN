"""Learn MNIST digits from "which came before which" labels only.

One training run: one model (`sltn` or the `gru` baseline), one number of training
sequences, one seed. The procedure is the one of `3_DigitsFromOrder.ipynb`, which
explains it; this script only runs it, for the seeds (`sweep.py` runs them all).

    python run.py --model sltn --n-train 1000 --seed 0

Writes `results/<model>-n<n_train>-seed<seed>/{config,metrics}.json` and returns the
metrics.
"""

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torchvision import datasets

import sltn
from sltn import Interpretation, Logic, Signature, kb_backward
from sltn.interpretation.operators.aggregator import AggregMax

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "datasets"               # MNIST, downloaded on first use
RESULTS = HERE / "results"
CONFIG = dict(
    T=6,               # frames per training sequence
    T_long=12,         # frames per sequence in the length-generalisation test
    n_train=1_000,     # training sequences
    n_val=500,
    n_test=2_000,
    batch_size=64,
    steps=3_000,       # gradient steps, the same for both models
    eval_every=250,    # validation, to keep the best checkpoint
    lr=1e-3,
    n_probe=100,       # labelled frames for the linear probe (10 per class)
)
K = 10


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result."""
    root = Path(sltn.__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


# --- Data -------------------------------------------------------------------

def load_pools(data_dir):
    def to_frames(ds):
        return ds.data.float().div(255).unsqueeze(1), ds.targets     # (N, 1, 28, 28), (N,)
    train_imgs, train_digits = to_frames(datasets.MNIST(data_dir, train=True, download=True))
    test_imgs, test_digits = to_frames(datasets.MNIST(data_dir, train=False, download=True))
    return {
        "train": (train_imgs[:55_000], train_digits[:55_000]),
        "val":   (train_imgs[55_000:], train_digits[55_000:]),
        "test":  (test_imgs, test_digits),
    }


def before_labels(d):
    # d: (N, T) true digits -> (N, K, K)
    oh = F.one_hot(d, K).float()
    earlier = (oh.cumsum(dim=1) - oh).clamp(max=1)
    return torch.einsum("nta,ntb->nab", earlier, oh).clamp(max=1)


def contains_labels(d):
    # d: (N, T) true digits -> (N, K): digit c appears somewhere
    return F.one_hot(d, K).amax(dim=1).float()


class Sequences:
    # n random sequences of T frames from one pool; frames are looked up per batch
    def __init__(self, pool, n, T, seed):
        imgs, digits = pool
        g = torch.Generator().manual_seed(seed)
        self.imgs = imgs
        self.idx = torch.randint(0, len(imgs), (n, T), generator=g)
        self.digits = digits[self.idx]                   # hidden: scoring only
        self.before = before_labels(self.digits)
        self.contains = contains_labels(self.digits)     # never used for training
        self.n, self.T = n, T

    def batch(self, b):
        return self.imgs[self.idx[b]], self.before[b]

    def chunks(self, size=250):
        for s in range(0, self.n, size):
            yield self.batch(torch.arange(s, min(s + size, self.n)))


# --- Networks ---------------------------------------------------------------

class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14 -> 7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        )

    def forward(self, img):                  # (N, 1, 28, 28) -> (N, 128)
        return self.net(img)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk, self.head = Trunk(), nn.Linear(128, K)

    def forward(self, img):                  # (N, 1, 28, 28) -> (N, K) probabilities
        return torch.softmax(self.head(self.trunk(img)), dim=-1)


class GRUBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.head = nn.Linear(128, K * K)

    def forward(self, x):                    # (B, T, 1, 28, 28) -> (B, K, K) logits
        B, T = x.shape[:2]
        _, h = self.gru(self.trunk(x.flatten(0, 1)).view(B, T, -1))
        return self.head(h[-1]).view(B, K, K)


# --- sLTN: the knowledge ----------------------------------------------------

sig = Signature("order")
for sort in ("Image", "Class", "Probs", "BeforeLabels"):
    sig.sort(sort)
sig.dimension("T")
sig.variable("x", "Image", dims=["T"])
sig.variable("y", "BeforeLabels")
sig.variable("c", "Class")
sig.structural_variable("t", "T")
sig.function("digit", ["Image"], "Probs")
sig.predicate("Is", ["Probs", "Class"])
sig.predicate("Before", ["BeforeLabels", "Class", "Class"])
sig.structural_relation("before", ["T", "T"])

rule = sig.parse(
    "forall (x, y): forall c: forall c1: (Before(y, c, c1) <-> "
    "exists t, t1 | before(t, t1): (Is(digit(x[t]), c) & Is(digit(x[t1]), c1)))"
)
answer = sig.parse("exists t, t1 | before(t, t1): (Is(digit(x[t]), c) & Is(digit(x[t1]), c1))")
# A question never trained on: does digit c appear?
contains = sig.parse("exists t: Is(digit(x[t]), c)")


def make_interp(digit_fn, image_type):
    logic = Logic().with_defaults()
    logic["exists,t"] = AggregMax()
    it = Interpretation(sig, logic=logic)
    it["Image"] = image_type
    it["Class"] = sltn.Type("Class", shape=(K,))
    it["Probs"] = sltn.Type("Probs", shape=(K,))
    it["BeforeLabels"] = sltn.Type("BeforeLabels", shape=(K, K))
    it["digit"] = digit_fn
    it["Is"] = lambda p, c: (p * c).sum(-1, keepdim=True)
    it["Before"] = lambda y, c, c1: torch.einsum("nij,ni,nj->n", y, c, c1).unsqueeze(-1)
    it["c"] = torch.eye(K)
    return it


def ground(it, x, y):
    T = x.shape[1]
    it["x"], it["y"] = x, y
    it["before"] = torch.ones(T, T, device=x.device).triu(diagonal=1)   # before[t, t1] = 1 iff t < t1


def evaluate_formula(it, formula, x, order):
    # Value of `formula` on the sequences x, with its axes reordered by name.
    ground(it, x, torch.zeros(x.shape[0], K, K, device=x.device))
    r = it(formula)
    names = [a.name for a in r.axes][:-1]
    return r.value.squeeze(-1).permute(*[names.index(n) for n in order])


def contains_from_before(before_pred):
    # (B, K, K) bool -> (B, K): with T >= 2, c appears iff it is before or after some digit
    return before_pred.any(dim=2) | before_pred.any(dim=1)


# --- Training and evaluation ------------------------------------------------

def macro_f1(y, pred):
    return f1_score(y.flatten(1).numpy(), pred.flatten(1).numpy(), average="macro", zero_division=0)


def run(model_name, n_train, seed, config=None, device=None, data_dir=None, out_dir=None):
    cfg = {**CONFIG, **(config or {}), "n_train": n_train}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pools = load_pools(data_dir or DATA)
    train = Sequences(pools["train"], n_train, cfg["T"], seed=1000 + seed)
    val = Sequences(pools["val"], cfg["n_val"], cfg["T"], seed=2000 + seed)
    tests = {T: Sequences(pools["test"], cfg["n_test"], T, seed=(3000 if T == cfg["T"] else 4000) + seed)
             for T in (cfg["T"], cfg["T_long"])}

    torch.manual_seed(seed)
    if model_name == "sltn":
        net = DigitCNN().to(device)
        interp = make_interp(net, sltn.Type("Image", shape=(1, 28, 28))).to(device)
        kb = {"Before rule": rule}

        def loss_step(opt, x, y):
            ground(interp, x, y)
            return kb_backward(opt, interp, kb)["agg_loss"]

        def predict(x):
            return evaluate_formula(interp, answer, x, ("x", "c", "c1")) > 0.5
    elif model_name == "gru":
        net = GRUBaseline().to(device)

        def loss_step(opt, x, y):
            loss = F.binary_cross_entropy_with_logits(net(x), y)
            loss.backward()
            return loss.item()

        def predict(x):
            return net(x) > 0
    else:
        raise ValueError(f"unknown model {model_name!r}")

    @torch.no_grad()
    def evaluate_f1(ds):
        preds = torch.cat([predict(x.to(device)).cpu() for x, _ in ds.chunks()])
        return macro_f1(ds.before, preds)

    @torch.no_grad()
    def probe_acc(trunk, ds):
        imgs, digits = pools["val"]
        idx = torch.cat([torch.nonzero(digits == k).flatten()[: cfg["n_probe"] // K] for k in range(K)])
        clf = LogisticRegression(max_iter=2000).fit(trunk(imgs[idx].to(device)).cpu().numpy(), digits[idx].numpy())
        feats = torch.cat([trunk(x.flatten(0, 1).to(device)).cpu() for x, _ in ds.chunks()]).numpy()
        return float((clf.predict(feats) == ds.digits.flatten().numpy()).mean())

    @torch.no_grad()
    def direct_acc(ds):
        preds = torch.cat([net(x.flatten(0, 1).to(device)).argmax(-1).cpu() for x, _ in ds.chunks()])
        return (preds == ds.digits.flatten()).float().mean().item()

    # Training: the notebook's train_model.
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(seed)
    best, curve, t0 = (-1.0, None, 0), [], time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        x, y = train.batch(torch.randint(0, train.n, (cfg["batch_size"],), generator=g))
        opt.zero_grad()
        loss = loss_step(opt, x.to(device), y.to(device))
        opt.step()
        if step % cfg["eval_every"] == 0:
            net.eval()
            f1 = evaluate_f1(val)
            net.train()
            curve.append((step, f1))
            if f1 > best[0]:
                best = (f1, copy.deepcopy(net.state_dict()), step)
            print(f"{model_name} n={n_train} seed={seed} step {step:4d}: loss {loss:.4f}, "
                  f"val F1 {f1:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    net.load_state_dict(best[1])
    net.eval()

    # Test: the notebook's section 5, plus the question never trained on.
    torch.manual_seed(seed)
    untrained = Trunk().to(device).eval()
    metrics = dict(best_step=best[2], val_before_f1=best[0],
                   minutes=round((time.perf_counter() - t0) / 60, 2),
                   frame_acc_probe=probe_acc(net.trunk, tests[cfg["T"]]),
                   frame_acc_probe_untrained=probe_acc(untrained, tests[cfg["T"]]))
    if model_name == "sltn":
        metrics["frame_acc"] = direct_acc(tests[cfg["T"]])
    for T, ds in tests.items():
        with torch.no_grad():
            before_pred = torch.cat([predict(x.to(device)).cpu() for x, _ in ds.chunks()])
            metrics[f"before_f1_T{T}"] = macro_f1(ds.before, before_pred)
            # Contains, never trained on. sLTN: the formula on the learned `digit`.
            # Both models: derived from the Before answers by hand.
            metrics[f"contains_from_before_f1_T{T}"] = macro_f1(ds.contains, contains_from_before(before_pred))
            if model_name == "sltn":
                pred = torch.cat([evaluate_formula(interp, contains, x.to(device), ("x", "c")).cpu() > 0.5
                                  for x, _ in ds.chunks()])
                metrics[f"contains_f1_T{T}"] = macro_f1(ds.contains, pred)
    metrics["curve"] = curve
    print(f"{model_name} n={n_train} seed={seed}:",
          {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items() if k != "curve"}, flush=True)

    record = dict(config={**cfg, "model": model_name, "seed": seed, "device": str(device)},
                  sltn=sltn_version(), metrics=metrics)
    run_dir = Path(out_dir or RESULTS) / f"{model_name}-n{n_train}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**record["config"], "sltn": record["sltn"]}, indent=1))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sltn", "gru"], required=True)
    p.add_argument("--n-train", type=int, default=CONFIG["n_train"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=CONFIG["steps"])
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    run(a.model, a.n_train, a.seed, config={"steps": a.steps}, out_dir=a.out_dir)

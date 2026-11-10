"""∂LTLf's exact protocol (github.com/andreoniriccardo/DiffLTLf, src/run.py, src/data/dataset_generator.py,
src/training/trainer.py; read, not run), with sLTN as the logic module.

What is reproduced exactly:
  - the dataset: their generator draws everything with Python's `random`, seeded by the first run (run 1) of
    the first model in the config, and caches it under a name without the seed, so all 10 runs share it.
    Seed of run r: 50 + 10 + 100·K + 10·r (470 for run 1 at |C| = 8; Set 1 and Set 2 share the sequences).
    Per sequence: length = randint(2, 10); per step: 3 classes = randint(0, C-1), each followed by an image
    index = randint(0, len(pool) - 1) into the MNIST training images of that class.
    Images: ToTensor -> Resize(32, 32, antialias) -> Normalize(0.1307, 0.3081), as their transform;
  - per run: set_seed(seed) (random, numpy, torch, cuda; cudnn deterministic); 80/20 random_split with a
    generator seeded by the run's seed; the CNN built right after, so with their initial weights (same layers
    in the same order); DataLoader(batch 64, shuffle=True) over the training split, batches padded to length 10;
    Adam lr 1e-4, 50 epochs, BCE without clamping; metrics after the last epoch on the 20% split.

Differences that remain:
  - sLTN evaluates the formula (product semantics, as theirs); padded steps are masked by feeding a sentinel image
    on which the network outputs 0 (repro.Masked), equivalent to their masks (checked, max diff 5e-6);
  - Set 2's U: "set2" writes it as the probabilistic sum over witnesses, which is >= their backward recursion on
    soft values (same crisp values); "set2_exact" gives their values exactly (see repro.BLOCKS);
  - dropout and float order on the GPU, so the runs are not bit-identical to theirs.

Usage: python their_protocol.py set1|set2|set2_exact [K=4] [runs=1-10]
Prints one JSON line per run. The stored runs (an A40 GPU, about 1 minute per run) are this output, one file per
set: results/their-protocol-20261104-144724/{set1,set2}.jsonl and results/their-protocol-exact-u-20261106/set2_exact.jsonl.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro
from repro import DEVICE, PAD, Masked, Perception, crisp_label, make_interp, make_signature, value
from sltn.interpretation.operators.base import AggregationOperator

MAX_LEN = 10


def _dims(x, dim):
    return tuple(range(x.dim())) if dim is None else (tuple(dim) if isinstance(dim, (tuple, list)) else (dim,))


class Prod(AggregationOperator):
    """∀ as a plain product, as their G (no clamping). Masked-out entries count as 1."""

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        x = xs if mask is None else 1 - mask * (1 - xs)
        for d in sorted(_dims(x, dim), reverse=True):
            x = x.prod(dim=d, keepdim=keepdim)
        return x

    def __repr__(self):
        return "Prod()"


class ProbSum(AggregationOperator):
    """∃ as a plain probabilistic sum 1 - ∏(1 - x), as their F (no clamping). Masked-out entries count as 0."""

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        x = 1 - (xs if mask is None else mask * xs)
        for d in sorted(_dims(x, dim), reverse=True):
            x = x.prod(dim=d, keepdim=keepdim)
        return 1 - x

    def __repr__(self):
        return "ProbSum()"


repro.AggregProd, repro.AggregProbSum = Prod, ProbSum       # make_interp picks these up


def set_seed(seed):
    # Their src/utils.py.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_of(K, run):
    return 50 + 10 + 100 * K + 10 * run


def their_dataset(fset, K, n=1000, lmin=2):
    """Their _generate, with the seed of run 1 set just before (as in run_experiment)."""
    C = 2 * K
    set_seed(seed_of(K, 1))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32), antialias=True),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    mnist = datasets.MNIST(repro.DATA, train=True, download=True)
    pools = [mnist.data[mnist.targets == i] for i in range(C)]
    seqs = torch.zeros(n, MAX_LEN, 3, 1, 32, 32)
    masks = torch.zeros(n, MAX_LEN, dtype=torch.bool)
    labels = torch.zeros(n)
    digits = torch.full((n, MAX_LEN, 3), -1, dtype=torch.long)
    for i in range(n):
        L = random.randint(lmin, MAX_LEN)
        for t in range(L):
            d = [random.randint(0, C - 1) for _ in range(3)]
            digits[i, t] = torch.tensor(d)
            for s in range(3):
                pool = pools[d[s]]
                seqs[i, t, s] = transform(pool[random.randint(0, len(pool) - 1)].numpy())
        masks[i, :L] = True
        labels[i] = float(crisp_label(digits[i, :L], fset, K))
    return seqs, masks, labels, digits


def their_value(p, fset, K):
    """∂LTLf's product-semantics evaluator, rewritten from their code: p [B, L, 3, C] class probabilities of the
    streams x, y, z. G is the product over the steps, F the probabilistic sum over this step and the later ones,
    X the value at the next step (0 at the last), and U their backward recursion U(t) = psi(t) ⊕ phi(t)·U(t+1)."""
    And, Or, Not, Imp = (lambda a, b: a * b), (lambda a, b: a + b - a * b), (lambda a: 1 - a), (lambda a, b: 1 - a + a * b)
    s = lambda stream, c: p[:, :, "xyz".index(stream), c]                         # [B, L]
    G = lambda v: v.prod(1)
    F_at = lambda v: 1 - (1 - v).flip(1).cumprod(1).flip(1)                        # F at every step, [B, L]
    X_at = lambda v: torch.cat([v[:, 1:], torch.zeros_like(v[:, :1])], 1)
    def U(phi, psi):
        u = torch.zeros_like(psi[:, 0])
        for t in reversed(range(psi.shape[1])):
            u = Or(psi[:, t], phi[:, t] * u)
        return u
    if fset == "set1":
        rules = [lambda: G(Imp(s("x", 0), F_at(s("y", 1)))),
                 lambda: G(Imp(Or(s("y", 2), s("z", 3)), Not(F_at(s("x", 3))))),
                 lambda: G(Imp(s("x", 4), X_at(s("y", 5)))),
                 lambda: G(Imp(Or(s("x", 6), s("y", 5)), Not(X_at(s("z", 7)))))]
    else:
        rules = [lambda: Or(U(Not(s("y", 1)), s("x", 0)), G(Not(s("y", 1)))),
                 lambda: G(Imp(X_at(Or(s("y", 0), s("z", 2))), s("x", 3))),
                 lambda: Imp(F_at(s("x", 4))[:, 0], F_at(s("y", 5))[:, 0]),
                 lambda: Imp(F_at(Or(s("x", 6), s("y", 4)))[:, 0], Not(F_at(s("z", 7))[:, 0]))]
    out = rules[0]()
    for rule in rules[1:K]:
        out = out * rule()
    return out


def compare_with_theirs(fset, K, n=200, seed=0):
    """sLTN's value of the formula against their_value, on random soft class probabilities, for lengths 2 to 10.
    Returns the max and mean absolute difference, and the mean signed difference (sLTN - theirs)."""
    C = 2 * K
    g = torch.Generator().manual_seed(seed)
    sig, formula = make_signature(fset, K)
    diffs = []
    for L in range(2, MAX_LEN + 1):
        probs = torch.softmax(3 * torch.randn(n * L * 3, C, generator=g), -1)    # one row per image
        table = probs.to(DEVICE)
        it = make_interp(sig, C, lambda v: table[v[:, 0, 0, 0].long()], fset)    # an "image" carries its row
        imgs = torch.arange(n * L * 3, dtype=torch.float).view(n, L, 3, 1, 1, 1).expand(n, L, 3, 1, 32, 32)
        ours = value(it, formula, imgs.contiguous().to(DEVICE)).cpu()
        diffs.append(ours - their_value(probs.view(n, L, 3, C).double(), fset, K).float())
    d = torch.cat(diffs)
    return d.abs().max().item(), d.abs().mean().item(), d.mean().item()


def run(fset, K, r, data, epochs=50, lr=1e-4, batch=64):
    C = 2 * K
    seed = seed_of(K, r)
    set_seed(seed)
    full = TensorDataset(*data)
    n_tr = int(0.8 * len(full))
    train_ds, val_ds = random_split(full, [n_tr, len(full) - n_tr], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    sig, formula = make_signature(fset, K)
    net = Perception(C).to(DEVICE)                            # built right after the split: their initial weights
    it = make_interp(sig, C, Masked(net), fset).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()

    def padded(imgs, masks):
        return torch.where(masks[:, :, None, None, None, None], imgs, torch.full_like(imgs, PAD))

    for _ in range(epochs):
        net.train()
        for imgs, masks, lab, _ in train_loader:
            imgs, masks, lab = imgs.to(DEVICE), masks.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            F.binary_cross_entropy(value(it, formula, padded(imgs, masks)), lab).backward()
            opt.step()
    net.eval()
    ok_img = n_img = ok_seq = n_seq = 0
    with torch.no_grad():
        for imgs, masks, lab, d in val_loader:
            imgs, masks, lab, d = imgs.to(DEVICE), masks.to(DEVICE), lab.to(DEVICE), d.to(DEVICE)
            pred = net(imgs.flatten(0, 2)).argmax(-1).view(d.shape)
            active = masks[:, :, None].expand_as(d)
            ok_img += (pred == d)[active].sum().item(); n_img += active.sum().item()
            ok_seq += ((value(it, formula, padded(imgs, masks)) > 0.5) == lab.bool()).sum().item(); n_seq += len(lab)
    return dict(fset=fset, K=K, run=r, seed=seed, img_acc=100 * ok_img / n_img, seq_acc=100 * ok_seq / n_seq,
                seconds=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    fset = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    runs = range(1, 11) if len(sys.argv) <= 3 else [int(x) for x in sys.argv[3].split(",")]
    data = their_dataset(fset, K)
    lab = data[2]
    print(json.dumps(dict(fset=fset, K=K, dataset_seed=seed_of(K, 1), positive_rate=round(lab.mean().item(), 4),
                          mean_length=round(data[1].sum(1).float().mean().item(), 3),
                          first_digits=data[3][0, :int(data[1][0].sum())].tolist())), flush=True)
    for r in runs:
        print(json.dumps(run(fset, K, r, data)), flush=True)

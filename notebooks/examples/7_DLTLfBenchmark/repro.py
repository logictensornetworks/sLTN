"""sLTN on the "Temporal Multi-Stream" benchmark of ∂LTLf (arXiv 2608.16443): the code of 7_DLTLfBenchmark.ipynb, as a script.

Reproduces the benchmark from the paper and its code (github.com/andreoniriccardo/DiffLTLf,
src/data/formulas.py and dataset_generator.py; read, not run):
  - 3 streams x, y, z; |C| = 2K classes = MNIST digits 0..2K-1; at every step each stream shows a
    uniformly drawn class; lengths uniform in [2, 10]; labels from the crisp formula, not balanced;
  - 1,000 sequences from the MNIST training set, 80% train / 20% test; images resized to 32x32 and
    normalised (mean 0.1307, std 0.3081);
  - the formula is a conjunction of K Declare blocks (Set 1 or Set 2), as in their formulas.py;
  - Adam lr 1e-4, batch 64, 50 epochs, BCE between the formula's value and the label;
  - their perception network (2 conv 5x5 with 32 / 64 filters, FC 1024 + dropout 0.5, FC |C|, softmax);
  - metrics: image accuracy (argmax vs true class, all streams and steps) and sequence accuracy (value >= 0.5).

sLTN writes the LTLf operators with structural quantifiers under product semantics:
  G phi        ->  ∀ t: phi(t)                          (∀ over time = product)
  F psi at t   ->  ∃ t1 | geq(t, t1): psi(t1)           (∃ = probabilistic sum)
  X psi at t   ->  ∃ t1 | nxt(t, t1): psi(t1)           (empty guard at the last step: 0, strong next)
  phi U psi    ->  ∃ t: (psi(t) & ∀ t1 | before(t1, t): phi(t1))
"set2_exact" is Set 2 with ∂LTLf's exact soft values: ∃ over the steps is a plain sum, and every ∃ t is written
as "the first step where psi holds" (see FIRST), so that the sum stays <= 1:
  F psi        ->  Σ t: psi(t) · ∏_{t1 < t} (1 - psi(t1))  =  1 - ∏_t (1 - psi(t))      (their F, exactly)
  phi U psi    ->  ∃ t: (psi(t) & ∀ t1 | before(t1, t): (phi(t1) & ¬psi(t1)))           (their recursion, exactly)
Their published numbers (∂LTLf, product, MNIST, mean of 10 runs), image / sequence accuracy:
  Set 1: |C|=2 99.9/99.8, 4 97.4/98.9, 6 95.7/98.3, 8 92.7/96.7
  Set 2: |C|=2 99.9/99.8, 4 97.4/99.0, 6 95.0/98.1, 8 92.7/98.0
"""

import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import sltn
from sltn import Interpretation, Logic, Signature
from sltn.interpretation.operators.base import AggregationOperator

DATA = Path(__file__).resolve().parents[2] / "datasets"     # MNIST, downloaded on first use


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result."""
    root = Path(sltn.__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_EVERY = 50
PUBLISHED = {  # ∂LTLf product semantics, MNIST: (image acc, sequence acc), %
    ("set1", 2): (99.9, 99.8), ("set1", 4): (97.4, 98.9), ("set1", 6): (95.7, 98.3), ("set1", 8): (92.7, 96.7),
    ("set2", 2): (99.9, 99.8), ("set2", 4): (97.4, 99.0), ("set2", 6): (95.0, 98.1), ("set2", 8): (92.7, 98.0),
}


class AggregProd(AggregationOperator):
    """∀ as the product (product semantics). Masked-out entries count as 1."""

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        x = xs if mask is None else 1 - mask * (1 - xs)
        dims = tuple(range(x.dim())) if dim is None else (tuple(dim) if isinstance(dim, (tuple, list)) else (dim,))
        return torch.exp(torch.log(x.clamp(min=1e-6)).sum(dim=dims, keepdim=keepdim))

    def __repr__(self):
        return "AggregProd()"


class AggregProbSum(AggregationOperator):
    """∃ as the probabilistic sum 1 - ∏(1 - x) (product semantics). Masked-out entries count as 0."""

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        x = xs if mask is None else mask * xs
        dims = tuple(range(x.dim())) if dim is None else (tuple(dim) if isinstance(dim, (tuple, list)) else (dim,))
        return 1 - torch.exp(torch.log1p(-x.clamp(max=1 - 1e-6)).sum(dim=dims, keepdim=keepdim))

    def __repr__(self):
        return "AggregProbSum()"


class AggregSum(AggregationOperator):
    """∃ as a plain sum, for set2_exact: every ∃ there sums terms of which at most one is true, so it stays <= 1
    (clamped at 1 only for the ~1e-6 rounding of sLTN's stable connectives). Masked-out entries count as 0."""

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        x = xs if mask is None else mask * xs
        dims = tuple(range(x.dim())) if dim is None else (tuple(dim) if isinstance(dim, (tuple, list)) else (dim,))
        return x.sum(dim=dims, keepdim=keepdim).clamp(max=1)

    def __repr__(self):
        return "AggregSum()"


# ----------------------------------------------------------------------------- formulas

def P(stream, c, t="t"):
    """Proposition "stream shows class c" at step t."""
    return f"Is(digit(s{stream}(s[{t}])), k{c})"


def F_(psi_t1):      # F psi, at t
    return f"(∃ t1 | geq(t, t1): {psi_t1})"


def X_(psi_t1):      # X psi, at t
    return f"(∃ t1 | nxt(t, t1): {psi_t1})"


def FIRST(psi):      # F psi from the first step, as "a step where psi holds, with no earlier one"; psi: step -> str
    return f"(∃ t: ({psi('t')} & (∀ t1 | before(t1, t): ¬{psi('t1')})))"


BLOCKS = {
    # their formulas.py; class indices are fixed per block, so K blocks use classes 0..2K-1
    "set1": [
        f"(∀ t: ({P('x', 0)} -> {F_(P('y', 1, 't1'))}))",                                    # G(x0 -> F y1)
        f"(∀ t: (({P('y', 2)} ∨ {P('z', 3)}) -> ¬{F_(P('x', 3, 't1'))}))",                    # G((y2 | z3) -> !F x3)
        f"(∀ t: ({P('x', 4)} -> {X_(P('y', 5, 't1'))}))",                                     # G(x4 -> X y5)
        f"(∀ t: (({P('x', 6)} ∨ {P('y', 5)}) -> ¬{X_(P('z', 7, 't1'))}))",                    # G((x6 | y5) -> !X z7)
    ],
    "set2": [
        f"((∃ t: ({P('x', 0)} & (∀ t1 | before(t1, t): ¬{P('y', 1, 't1')}))) ∨ (∀ t: ¬{P('y', 1)}))",  # (!y1 U x0) | G !y1
        f"(∀ t: ({X_('(' + P('y', 0, 't1') + ' ∨ ' + P('z', 2, 't1') + ')')} -> {P('x', 3)}))",       # G(X(y0 | z2) -> x3)
        f"((∃ t: {P('x', 4)}) -> (∃ t: {P('y', 5)}))",                                        # F x4 -> F y5
        f"((∃ t: ({P('x', 6)} ∨ {P('y', 4)})) -> ¬(∃ t: {P('z', 7)}))",                       # F(x6 | y4) -> !F z7
    ],
    # Set 2 with ∂LTLf's soft values, under ∃ over the steps = plain sum (crisp_label treats it as set2)
    "set2_exact": [
        f"((∃ t: ({P('x', 0)} & (∀ t1 | before(t1, t): (¬{P('y', 1, 't1')} & ¬{P('x', 0, 't1')})))) ∨ (∀ t: ¬{P('y', 1)}))",
        f"(∀ t: ({X_('(' + P('y', 0, 't1') + ' ∨ ' + P('z', 2, 't1') + ')')} -> {P('x', 3)}))",       # one next step: sum = ⊕
        f"({FIRST(lambda t: P('x', 4, t))} -> {FIRST(lambda t: P('y', 5, t))})",
        f"({FIRST(lambda t: '(' + P('x', 6, t) + ' ∨ ' + P('y', 4, t) + ')')} -> ¬{FIRST(lambda t: P('z', 7, t))})",
    ],
}


def crisp_label(d, fset, K):
    """d: [L, 3] classes of one sequence. The crisp value of the conjunction of the first K blocks."""
    L = len(d)
    x, y, z = d[:, 0].tolist(), d[:, 1].tolist(), d[:, 2].tolist()
    ok = []
    if fset == "set1":
        ok.append(all(x[t] != 0 or any(y[u] == 1 for u in range(t, L)) for t in range(L)))
        ok.append(all(not (y[t] == 2 or z[t] == 3) or not any(x[u] == 3 for u in range(t, L)) for t in range(L)))
        ok.append(all(x[t] != 4 or (t + 1 < L and y[t + 1] == 5) for t in range(L)))
        ok.append(all(not (x[t] == 6 or y[t] == 5) or not (t + 1 < L and z[t + 1] == 7) for t in range(L)))
    else:
        until = any(x[t] == 0 and all(y[u] != 1 for u in range(t)) for t in range(L))
        ok.append(until or all(v != 1 for v in y))
        ok.append(all(not (t + 1 < L and (y[t + 1] == 0 or z[t + 1] == 2)) or x[t] == 3 for t in range(L)))
        ok.append(not any(v == 4 for v in x) or any(v == 5 for v in y))
        ok.append(not any(x[t] == 6 or y[t] == 4 for t in range(L)) or not any(v == 7 for v in z))
    return all(ok[:K])


def make_signature(fset, K):
    C = 2 * K
    sig = Signature(f"dltlf_{fset}_{C}")
    for sort in ("Step", "Image", "Class", "Probs"):
        sig.sort(sort)
    sig.dimension("T")
    sig.structural_variable("t", "T")
    sig.variable("s", "Step", dims=["T"])
    for f in ("sx", "sy", "sz"):
        sig.function(f, ["Step"], "Image")
    sig.function("digit", ["Image"], "Probs")
    sig.predicate("Is", ["Probs", "Class"])
    for c in range(C):
        sig.constant(f"k{c}", "Class")
    for r in ("geq", "nxt", "before"):
        sig.structural_relation(r, ["T", "T"])
    formula = sig.parse(" & ".join(BLOCKS[fset][:K]))
    return sig, formula


# ----------------------------------------------------------------------------- data

def make_dataset(fset, K, g, n=1000, lmin=2, lmax=10):
    C = 2 * K
    ds = datasets.MNIST(DATA, train=True, download=True)
    imgs = F.interpolate(ds.data.float().div(255).unsqueeze(1), size=(32, 32), mode="bilinear", antialias=True)
    imgs = (imgs - 0.1307) / 0.3081
    by_class = [torch.nonzero(ds.targets == c).flatten() for c in range(C)]
    lengths = torch.randint(lmin, lmax + 1, (n,), generator=g)
    digits = torch.randint(0, C, (n, lmax, 3), generator=g)
    pick = torch.empty_like(digits)
    for c in range(C):
        m = digits == c
        pick[m] = by_class[c][torch.randint(0, len(by_class[c]), (int(m.sum()),), generator=g)]
    labels = torch.tensor([crisp_label(digits[i, :lengths[i]], fset, K) for i in range(n)])
    perm = torch.randperm(n, generator=g)
    n_tr = int(0.8 * n)
    split = lambda idx: dict(digits=digits[idx], img_idx=pick[idx], length=lengths[idx], label=labels[idx])
    return imgs, split(perm[:n_tr]), split(perm[n_tr:])


PAD = -100.0          # padded steps are images filled with this value (real images are >= -0.43)


class Data:
    def __init__(self, s, imgs):
        self.imgs = imgs.to(DEVICE)
        self.s = {k: v.to(DEVICE) for k, v in s.items()}
        self.n = len(self.s["length"])

    def batch(self, idx):
        L = int(self.s["length"][idx].max())
        return self.imgs[self.s["img_idx"][idx, :L]], self.s["label"][idx].float(), self.s["digits"][idx, :L]

    def epochs(self, batch, n_epochs, g):
        """Their training: n_epochs passes in batches of `batch`. Batches are grouped by length (no padding)."""
        lengths = self.s["length"].cpu()
        for _ in range(n_epochs):
            perm = torch.randperm(self.n, generator=g)
            order = perm[torch.argsort(lengths[perm], stable=True)]       # by length, random within a length
            chunks = [order[i:i + batch] for i in range(0, self.n, batch)]
            for j in torch.randperm(len(chunks), generator=g).tolist():
                idx = chunks[j]
                for L in lengths[idx].unique().tolist():                  # a chunk may span two lengths
                    yield self.batch(idx[lengths[idx] == L].to(DEVICE))

    def mixed_epochs(self, batch, n_epochs, g):
        """Their batching: random batches of `batch` sequences of mixed lengths, padded to the longest one.
        Padded steps hold PAD images, on which the network outputs 0 (see Masked), as their masks do."""
        for _ in range(n_epochs):
            perm = torch.randperm(self.n, generator=g)
            for i in range(0, self.n, batch):
                idx = perm[i:i + batch].to(DEVICE)
                L = int(self.s["length"][idx].max())
                imgs = self.imgs[self.s["img_idx"][idx, :L]]
                pad = torch.arange(L, device=DEVICE)[None] >= self.s["length"][idx][:, None]   # [B, L]
                imgs = torch.where(pad[:, :, None, None, None, None], torch.full_like(imgs, PAD), imgs)
                yield imgs, self.s["label"][idx].float(), self.s["digits"][idx, :L]

    def all_batches(self):
        lengths = self.s["length"].cpu()
        for L in lengths.unique().tolist():
            yield self.batch(torch.nonzero(lengths == L).flatten().to(DEVICE))


class Perception(nn.Module):
    """The perception network of ∂LTLf (src/models/perception.py, as described in the paper)."""

    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),          # 32 -> 28 -> 14
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),         # 14 -> 10 -> 5
            nn.Flatten(), nn.Linear(64 * 5 * 5, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, C))

    def forward(self, x):
        return torch.softmax(self.net(x), -1)


class Masked(nn.Module):
    """The perception network, with an all-zero output on padded images: every proposition is 0 there."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        valid = (x.flatten(1)[:, 0] > PAD / 2).float().unsqueeze(-1)
        return self.net(x) * valid


def make_interp(sig, C, net, fset=None):
    lg = Logic().with_defaults()
    lg["forall,t"] = AggregProd()
    lg["exists,t"] = AggregSum() if fset == "set2_exact" else AggregProbSum()
    it = Interpretation(sig, logic=lg)
    it["Step"] = sltn.Type("step", shape=(3, 1, 32, 32))
    it["Image"] = sltn.Type("image", shape=(1, 32, 32))
    it["Class"] = sltn.Type("class", shape=(C,))
    it["Probs"] = sltn.Type("probs", shape=(C,))
    it["sx"], it["sy"], it["sz"] = (lambda v: v[:, 0]), (lambda v: v[:, 1]), (lambda v: v[:, 2])
    it["digit"] = net
    it["Is"] = lambda p, c: (p * c).sum(-1, keepdim=True)
    eye = torch.eye(C, device=DEVICE)
    for c in range(C):
        it[f"k{c}"] = eye[c]
    return it


def ground(it, imgs):
    L = imgs.shape[1]
    it["s"] = imgs
    t = torch.arange(L, device=DEVICE)
    it["geq"] = (t[:, None] <= t[None]).float()                      # geq(t, t1): t1 >= t
    it["nxt"] = (t[:, None] + 1 == t[None]).float()                  # nxt(t, t1): t1 = t + 1
    it["before"] = (t[:, None] < t[None]).float()                    # before(t1, t): t1 < t


def value(it, formula, imgs):
    ground(it, imgs)
    r = it(formula)
    assert r.axis_names[0] == "s"
    return r.value.squeeze(-1)


def run(fset, K, seed, epochs=50, lr=1e-4, batch=64, log=None, mixed=False):
    C = 2 * K
    g = torch.Generator().manual_seed(seed)
    imgs, tr, te = make_dataset(fset, K, g)
    train, test = Data(tr, imgs), Data(te, imgs)
    sig, formula = make_signature(fset, K)
    torch.manual_seed(seed)
    net = Perception(C).to(DEVICE)
    it = make_interp(sig, C, Masked(net) if mixed else net, fset).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    batches = (train.mixed_epochs if mixed else train.epochs)(batch, epochs, torch.Generator().manual_seed(seed))
    for step, (b_imgs, lab, _) in enumerate(batches, 1):
        opt.zero_grad()
        v = value(it, formula, b_imgs).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(v, lab)
        loss.backward()
        opt.step()
        if log and step % LOG_EVERY == 0:
            log({"loss": loss.item()}, step)
    net.eval()
    ok_img = n_img = ok_seq = n_seq = 0
    with torch.no_grad():
        for b_imgs, lab, d in test.all_batches():
            pred = net(b_imgs.flatten(0, 2)).argmax(-1).view(d.shape)
            ok_img += (pred == d).sum().item(); n_img += d.numel()
            ok_seq += ((value(it, formula, b_imgs) >= 0.5) == lab.bool()).sum().item(); n_seq += len(lab)
    return dict(img_acc=100 * ok_img / n_img, seq_acc=100 * ok_seq / n_seq, positive_rate=float(tr["label"].float().mean()),
                steps=step, seconds=round(time.perf_counter() - t0, 1))


def oracle_check(fset, K, n=300):
    """With perfect perception, sLTN's formula must reproduce the crisp labels."""
    C = 2 * K
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(2, 11, (n,), generator=g)
    digits = torch.randint(0, C, (n, 10, 3), generator=g)
    labels = torch.tensor([crisp_label(digits[i, :lengths[i]], fset, K) for i in range(n)])
    sig, formula = make_signature(fset, K)
    oracle = lambda v: F.one_hot(v[:, 0, 0, 0].round().long(), C).float()   # an "image" carries its class
    it = make_interp(sig, C, oracle, fset)
    ok = 0
    for L in lengths.unique().tolist():
        idx = torch.nonzero(lengths == L).flatten()
        imgs = torch.zeros(len(idx), L, 3, 1, 32, 32)
        imgs[..., 0, 0, 0] = digits[idx, :L].float()
        ok += ((value(it, formula, imgs.to(DEVICE)) >= 0.5) == labels[idx].to(DEVICE)).sum().item()
    return ok, n, labels.float().mean().item()


if __name__ == "__main__":
    if "--oracle" in sys.argv:
        for fset in ("set1", "set2"):
            for K in (1, 2, 3, 4):
                ok, n, pos = oracle_check(fset, K)
                print(f"{fset} |C|={2 * K}: sLTN formula == crisp label on {ok}/{n} sequences; positive rate {pos:.2f}")
    else:
        fset, K = sys.argv[1], int(sys.argv[2])
        print(run(fset, K, seed=0, epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 50))

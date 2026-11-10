"""A temporal pattern over relations between images (Response(Sum, Eq)): the models of 6_DigitRelationsOverTime.ipynb, as a script.

Each sequence has 2-10 steps; each step shows three MNIST digits x, y, z. Two relational atoms:
  Sum: x + y = z            Eq: x = y
One yes/no label per sequence: Response(Sum, Eq), "every step where x + y = z is followed, at
that step or later, by a step where x = y". Labels are balanced. No digit labels: the digit CNN
is learned from the sequence labels alone.

Models:
  sltn-prod   sLTN, product semantics: ∀ over time = product, ∃ = probabilistic sum (defined here)
  sltn-godel  sLTN, Gödel semantics: ∀ = min, ∃ = max
  gru         the same CNN on each image, a GRU over the steps, BCE on the labels

Run (CPU, short): conda run -n sltn python prototype.py --steps 200
All the settings and seeds (GPU): sweep.py.
"""

import argparse
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import sltn
from sltn import KB, Interpretation, Logic, Signature
from sltn.interpretation.operators.aggregator import AggregMax, AggregMin
from sltn.interpretation.operators.base import AggregationOperator

DATA = Path(__file__).resolve().parents[2] / "datasets"     # MNIST, downloaded on first use
K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_EVERY = 100


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result."""
    root = Path(sltn.__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


# ----------------------------------------------------------------------------- product semantics

class AggregProd(AggregationOperator):
    """∀ as the product of the values (product semantics). Masked-out entries count as 1."""

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


# ----------------------------------------------------------------------------- data

def load_pool(train):
    ds = datasets.MNIST(DATA, train=train, download=True)
    return ds.data.float().div(255).unsqueeze(1), ds.targets


def atom(d, relation):
    """The first atom of the pattern at every step: x + y = z ("sum") or x < y ("less")."""
    return d[..., 0] + d[..., 1] == d[..., 2] if relation == "sum" else d[..., 0] < d[..., 1]


def response_label(d, length, relation="sum"):
    """d: [T, 3] digits of one sequence (first `length` steps). Response(atom, Eq)."""
    d = d[:length]
    sm = atom(d, relation)
    eq = d[:, 0] == d[:, 1]
    later_eq = torch.flip(torch.cummax(torch.flip(eq.int(), [0]), 0).values, [0]).bool()   # eq at t or later
    return bool((~sm | later_eq).all())


def make_sequences(n, pool, g, lmin=2, lmax=10, p_sum=0.25, p_eq=0.2, relation="sum"):
    """n sequences, half with label yes, padded to lmax. Steps are drawn so that both atoms are frequent:
    with prob p_sum a step satisfies x + y = z, with prob p_eq it has x = y, otherwise it is uniform."""
    imgs_pool, digits_pool = pool
    by_class = [torch.nonzero(digits_pool == k).flatten() for k in range(K)]
    seqs, lengths, labels = [], [], []
    want = {True: n // 2, False: n - n // 2}
    while want[True] or want[False]:
        L = int(torch.randint(lmin, lmax + 1, (1,), generator=g))
        u = torch.rand(L, generator=g)
        x = torch.randint(0, K, (L,), generator=g)
        y = torch.randint(0, K, (L,), generator=g)
        z = torch.randint(0, K, (L,), generator=g)
        s_mode, e_mode = u < p_sum, (u >= p_sum) & (u < p_sum + p_eq)
        if relation == "sum":
            y_sum = (torch.rand(L, generator=g) * (K - x)).long()             # y in [0, 9 - x]
            y = torch.where(s_mode, y_sum, torch.where(e_mode, x, y))
            z = torch.where(s_mode, x + y, z)
        else:                                                                # a step with x < y
            x_l = (torch.rand(L, generator=g) * (K - 1)).long()              # x in [0, 8]
            y_l = x_l + 1 + (torch.rand(L, generator=g) * (K - 1 - x_l)).long()   # y in [x + 1, 9]
            x = torch.where(s_mode, x_l, x)
            y = torch.where(s_mode, y_l, torch.where(e_mode, x, y))
        d = torch.stack([x, y, z], 1)
        lab = response_label(d, L, relation)
        if want[lab]:
            want[lab] -= 1
            seqs.append(F.pad(d, (0, 0, 0, lmax - L))); lengths.append(L); labels.append(lab)
    digits = torch.stack(seqs)                                                 # [n, lmax, 3]
    pick = torch.empty_like(digits)
    for k in range(K):
        m = digits == k
        pick[m] = by_class[k][torch.randint(0, len(by_class[k]), (int(m.sum()),), generator=g)]
    perm = torch.randperm(n, generator=g)
    return dict(digits=digits[perm], img_idx=pick[perm], length=torch.tensor(lengths)[perm],
                label=torch.tensor(labels)[perm])


class Data:
    """Sequences and the image pool, on DEVICE. Batches hold sequences of one length."""

    def __init__(self, seqs, pool):
        self.pool = pool[0].to(DEVICE)
        self.s = {k: v.to(DEVICE) for k, v in seqs.items()}
        self.n = len(self.s["length"])

    def batch(self, idx):
        L = int(self.s["length"][idx].max())
        imgs = self.pool[self.s["img_idx"][idx, :L]]                          # [B, L, 3, 1, 28, 28]
        return imgs, self.s["label"][idx].float(), self.s["digits"][idx, :L]

    def random_batches(self, batch, steps, g):
        lengths = self.s["length"].cpu()
        for _ in range(steps):
            L = int(lengths[torch.randint(0, self.n, (1,), generator=g)])     # a length, as frequent as in the data
            pool = torch.nonzero(lengths == L).flatten()
            yield self.batch(pool[torch.randint(0, len(pool), (batch,), generator=g)].to(DEVICE))

    def all_batches(self, batch=200):
        lengths = self.s["length"].cpu()
        for L in lengths.unique().tolist():
            idx = torch.nonzero(lengths == L).flatten()
            for s in range(0, len(idx), batch):
                yield self.batch(idx[s:s + batch].to(DEVICE))


# ----------------------------------------------------------------------------- networks

class Trunk(nn.Module):
    """The image network of ∂LTLf (two 5x5 conv layers, 32 and 64 filters, a 1024-unit layer)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2),          # 28 -> 24 -> 12
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),         # 12 -> 8 -> 4
            nn.Flatten(), nn.Linear(64 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        return self.net(x)


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk, self.head = Trunk(), nn.Linear(1024, K)

    def forward(self, x):                                               # [N, 1, 28, 28] -> [N, K]
        return torch.softmax(self.head(self.trunk(x)), -1)


class GRUBaseline(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.trunk = Trunk()
        self.gru = nn.GRU(3 * 1024, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, imgs):                                            # [B, L, 3, 1, 28, 28] -> [B] logits
        B, L = imgs.shape[:2]
        f = self.trunk(imgs.flatten(0, 2)).view(B, L, 3 * 1024)
        _, h = self.gru(f)
        return self.out(h[-1]).squeeze(-1)


# ----------------------------------------------------------------------------- sLTN

sig = Signature("relational_temporal")
for sort in ("Step", "Image", "Class", "Probs", "Label"):
    sig.sort(sort)
sig.dimension("T")
sig.structural_variable("t", "T")
sig.variable("s", "Step", dims=["T"])           # a sequence; one Step = the three images of a step
sig.variable("l", "Label")
sig.variable("c", "Class")
sig.variable("li", "Image")                      # a few labelled digit images (the same 100 the probe uses)
sig.variable("ld", "Class")                      # their digits, one-hot
for f in ("sx", "sy", "sz"):                     # the image of stream x, y or z
    sig.function(f, ["Step"], "Image")
sig.function("digit", ["Image"], "Probs")
sig.predicate("Is", ["Probs", "Class"])
sig.predicate("Add", ["Class", "Class", "Class"])
sig.predicate("Less", ["Class", "Class"])
sig.predicate("Yes", ["Label"])
sig.structural_relation("geq", ["T", "T"])     # geq(t, t1): t1 is t or later

ATOM = {
    "sum": "(∃ c, c1, c2: (Add(c, c1, c2) & Is(digit(sx(s[t])), c) & Is(digit(sy(s[t])), c1) & Is(digit(sz(s[t])), c2)))",
    "less": "(∃ c, c1: (Less(c, c1) & Is(digit(sx(s[t])), c) & Is(digit(sy(s[t])), c1)))",
}
EQ = "(∃ c3: (Is(digit(sx(s[t1])), c3) & Is(digit(sy(s[t1])), c3)))"
ANSWER = {r: sig.parse(f"∀ t: ({a} -> (∃ t1 | geq(t, t1): {EQ}))") for r, a in ATOM.items()}
RULE = {r: sig.parse(f"∀ (s, l): (Yes(l) <-> (∀ t: ({a} -> (∃ t1 | geq(t, t1): {EQ}))))") for r, a in ATOM.items()}
# Every class appears somewhere in the batch. Without it, the CNN can merge digits into a few classes that
# satisfy most labels (a reasoning shortcut); with all 10 classes in use, the true labelling is the only one.
DIGIT_LABELS = sig.parse("∀ (li, ld): Is(digit(li), ld)")
COVER = sig.parse("∀ c: (∃ s: (∃ t: (Is(digit(sx(s[t])), c) ∨ Is(digit(sy(s[t])), c) ∨ Is(digit(sz(s[t])), c))))")

ADD = torch.zeros(K, K, K)
for a in range(K):
    for b in range(K - a):
        ADD[a, b, a + b] = 1


def make_interp(semantics, net):
    lg = Logic().with_defaults()
    if semantics == "prod":
        lg["forall,t"], lg["exists,t"], lg["exists"] = AggregProd(), AggregProbSum(), AggregProbSum()
    else:
        lg["forall,t"], lg["exists,t"], lg["exists"] = AggregMin(), AggregMax(), AggregMax()
    it = Interpretation(sig, logic=lg)
    it["Step"] = sltn.Type("step", shape=(3, 1, 28, 28))
    it["Image"] = sltn.Type("image", shape=(1, 28, 28))
    it["Class"] = sltn.Type("class", shape=(K,))
    it["Probs"] = sltn.Type("probs", shape=(K,))
    it["Label"] = sltn.Type("label", shape=(1,))
    it["sx"], it["sy"], it["sz"] = (lambda v: v[:, 0]), (lambda v: v[:, 1]), (lambda v: v[:, 2])
    it["digit"] = net
    it["Is"] = lambda p, c: (p * c).sum(-1, keepdim=True)
    add = ADD.to(DEVICE)
    it["Add"] = lambda a, b, c: torch.einsum("ijk,ni,nj,nk->n", add, a, b, c).unsqueeze(-1)
    less = torch.ones(K, K, device=DEVICE).triu(1)                          # less[i, j] = 1 iff i < j
    it["Less"] = lambda a, b: torch.einsum("ij,ni,nj->n", less, a, b).unsqueeze(-1)
    it["Yes"] = lambda v: v
    it["c"] = torch.eye(K, device=DEVICE)
    return it


def ground(it, imgs, labels):
    L = imgs.shape[1]
    it["s"] = imgs
    it["l"] = labels.unsqueeze(-1)
    it["geq"] = torch.ones(L, L, device=DEVICE).triu()


# ----------------------------------------------------------------------------- training

def labelled_digits(pools, n_per_class=10):
    """The 100 labelled training images (10 per class) that the probe also uses."""
    imgs, digits = pools["train"]
    idx = torch.cat([torch.nonzero(digits == k).flatten()[:n_per_class] for k in range(K)])
    return imgs[idx].to(DEVICE), F.one_hot(digits[idx], K).float().to(DEVICE)


def train_sltn(train, a, seed, semantics, log=None, cover=False, labels=None, rule=True):
    torch.manual_seed(seed)
    net = DigitNet().to(DEVICE)
    it = make_interp(semantics, net).to(DEVICE)
    clauses = ([] if rule is False else [RULE[a.relation]]) + ([COVER] if cover else []) \
        + ([DIGIT_LABELS] if labels is not None else [])
    kb = KB(clauses)
    if labels is not None:
        it["li"], it["ld"] = labels
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    for step, (imgs, lab, _) in enumerate(train.random_batches(a.batch, a.steps, torch.Generator().manual_seed(seed)), 1):
        ground(it, imgs, lab)
        opt.zero_grad()
        sat = it(kb).value.squeeze()
        (1 - sat).backward()
        opt.step()
        if log and step % LOG_EVERY == 0:
            with torch.no_grad():
                d = {"loss": 1 - sat.item(), "sat_rule": it(RULE[a.relation]).value.item()}   # monitored even if not trained
                if labels is not None:
                    d["sat_digit_labels"] = it(DIGIT_LABELS).value.item()
                if cover:
                    d["sat_cover"] = it(COVER).value.item()
            log(d, step)
    net.eval()

    @torch.no_grad()
    def answer(imgs, lab):
        ground(it, imgs, lab)
        r = it(ANSWER[a.relation])
        assert r.axis_names[0] == "s"
        return r.value.squeeze(-1) > 0.5

    return net, answer


def train_gru(train, a, seed, log=None):
    torch.manual_seed(seed)
    net = GRUBaseline().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=getattr(a, "gru_lr", a.lr))
    for step, (imgs, lab, _) in enumerate(train.random_batches(a.batch, a.steps, torch.Generator().manual_seed(seed)), 1):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(net(imgs), lab)
        loss.backward()
        opt.step()
        if log and step % LOG_EVERY == 0:
            log({"loss": loss.item()}, step)
    net.eval()
    return net, torch.no_grad()(lambda imgs, lab: net(imgs) > 0)


@torch.no_grad()
def seq_acc(answer, data):
    ok = n = 0
    for imgs, lab, _ in data.all_batches():
        ok += (answer(imgs, lab) == lab.bool()).sum().item(); n += len(lab)
    return ok / n


@torch.no_grad()
def image_acc(digit_net, data, confusion=False):
    ok = n = 0
    conf = torch.zeros(K, K, dtype=torch.long)                         # [true digit, predicted class]
    for imgs, _, d in data.all_batches():
        pred = digit_net(imgs.flatten(0, 2)).argmax(-1).view(d.shape)
        ok += (pred == d).sum().item(); n += d.numel()
        conf += torch.bincount((d * K + pred).flatten().cpu(), minlength=K * K).view(K, K)
    return (ok / n, conf.tolist()) if confusion else ok / n


@torch.no_grad()
def probe_acc(trunk, pools, data, n_per_class=10):
    """Linear probe on the frozen trunk: logistic regression on 100 labelled training images."""
    from sklearn.linear_model import LogisticRegression
    imgs, digits = pools["train"]
    idx = torch.cat([torch.nonzero(digits == k).flatten()[:n_per_class] for k in range(K)])
    clf = LogisticRegression(max_iter=2000).fit(trunk(imgs[idx].to(DEVICE)).cpu().numpy(), digits[idx].numpy())
    ok = n = 0
    for b, _, d in data.all_batches():
        pred = torch.tensor(clf.predict(trunk(b.flatten(0, 2)).cpu().numpy())).view(d.shape)
        ok += (pred == d.cpu()).sum().item(); n += d.numel()
    return ok / n


def run(model, a, seed, pools, sets, log=None):
    """Train one model; returns its test metrics."""
    t0 = time.perf_counter()
    if model == "untrained":                                                 # the probe's floor
        torch.manual_seed(seed)
        trunk = Trunk().to(DEVICE).eval()
        return dict(seq_acc=None, seq_acc_long=None, img_acc=None, probe_acc=probe_acc(trunk, pools, sets["test"]),
                    seconds=round(time.perf_counter() - t0, 1))
    if model.startswith("sltn"):                             # sltn-<semantics>[-cover][-lab | -labonly]
        parts = model.split("-")
        net, answer = train_sltn(sets["train"], a, seed, parts[1], log, cover="cover" in parts,
                                 labels=labelled_digits(pools) if {"lab", "labonly"} & set(parts) else None,
                                 rule="labonly" not in parts)
        trunk = net.trunk
        img, conf = image_acc(net, sets["test"], confusion=True)
    else:
        net, answer = train_gru(sets["train"], a, seed, log)
        trunk, img, conf = net.trunk, None, None
    return dict(seq_acc=seq_acc(answer, sets["test"]), seq_acc_long=seq_acc(answer, sets["test_long"]),
                img_acc=img, confusion=conf, probe_acc=probe_acc(trunk, pools, sets["test"]),
                seconds=round(time.perf_counter() - t0, 1))


def make_sets(a, seed, pools):
    r = a.relation
    return dict(
        train=Data(make_sequences(a.n_train, pools["train"], torch.Generator().manual_seed(1000 + seed), relation=r),
                   pools["train"]),
        test=Data(make_sequences(a.n_test, pools["test"], torch.Generator().manual_seed(5000 + seed), relation=r),
                  pools["test"]),
        test_long=Data(make_sequences(a.n_test, pools["test"], torch.Generator().manual_seed(7000 + seed),
                                      lmin=20, lmax=20, relation=r), pools["test"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", nargs="+", default=["sltn-prod", "sltn-godel", "gru"])
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--relation", choices=["sum", "less"], default="sum")
    a = ap.parse_args()
    pools = {"train": load_pool(True), "test": load_pool(False)}
    print(f"sltn {sltn.__version__}, device {DEVICE}, {vars(a)}")
    for seed in a.seeds:
        sets = make_sets(a, seed, pools)
        print(f"seed {seed}: train {sets['train'].n} sequences, "
              f"{sets['train'].s['label'].float().mean():.2f} positive", flush=True)
        for m in a.models:
            r = run(m, a, seed, pools, sets)
            print(f"  {m:10s} {r}", flush=True)


if __name__ == "__main__":
    main()

"""Graph colouring from occluded MNIST digits: the models and training of 5_GraphColouring.ipynb, as a script.

Each graph has 6-15 nodes, properly 3-coloured: colours are drawn first, then edges are
added only between nodes of different colours (average degree ~3). Each node shows its
colour as an MNIST digit 0, 1 or 2 with an 18x18 square blanked at a random place, so a
single node is often ambiguous. A fraction `label_frac` of the training nodes is labelled.

Models, all with the same CNN per node:
  cnn    cross-entropy on the labelled nodes
  gnn    the CNN, then two GraphSAGE-style layers (node and neighbour mean kept separate)
  sltn   the label clause and the rule "neighbours have different colours"
At test time every model predicts with its network alone (no test-time use of the rule).

Run: conda run -n sltn python prototype.py [--seeds 0] [--fracs 0.01 0.02 0.05]
(sweep.py runs it over seeds, label fractions and wrong edges.)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import sltn
from sltn import KB, Interpretation, Logic, Signature
from sltn.interpretation.operators.aggregator import AggregPMeanError

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "datasets"     # MNIST, downloaded on first use
sys.path.insert(0, str(HERE))
from calibrate import CNN, degrade  # noqa: E402

K = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_EVERY = 100          # steps between two calls of `log`, an optional callback (loss, satisfaction)
N_MAX = 15


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result."""
    root = Path(sltn.__file__).resolve().parent.parent
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


# ----------------------------------------------------------------------------- data

def load_pool(train):
    ds = datasets.MNIST(DATA, train=train, download=True)
    keep = ds.targets < K
    return ds.data[keep].float().div(255).unsqueeze(1), ds.targets[keep]


def make_graphs(n_graphs, pool, label_frac, g, degree=3.0, occlude=18, wrong=0.0):
    """Padded to N_MAX nodes. Returns images [G, N, 1, 28, 28], colours [G, N], adjacency
    [G, N, N] (symmetric, zero diagonal, zero on padding), valid and labelled masks [G, N]."""
    imgs_pool, digits_pool = pool
    by_class = [torch.nonzero(digits_pool == k).flatten() for k in range(K)]
    sizes = torch.randint(6, N_MAX + 1, (n_graphs,), generator=g)
    valid = torch.arange(N_MAX)[None] < sizes[:, None]
    colour = torch.randint(0, K, (n_graphs, N_MAX), generator=g)
    differ = (colour[:, :, None] != colour[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    # each node has about (2/3)(n-1) nodes of another colour: pick an edge probability for `degree`
    p_edge = (degree / ((2 / 3) * (sizes - 1).float())).clamp(max=1)[:, None, None]
    upper = torch.rand(n_graphs, N_MAX, N_MAX, generator=g).triu(1)
    adj = ((upper > 0) & (upper < p_edge) & differ).float()
    adj = adj + adj.transpose(1, 2)
    # one image per node, of its colour's digit, then occluded
    pick = torch.empty(n_graphs, N_MAX, dtype=torch.long)
    for k in range(K):
        m = colour == k
        pick[m] = by_class[k][torch.randint(0, len(by_class[k]), (int(m.sum()),), generator=g)]
    imgs = degrade(imgs_pool[pick.flatten()], 0.0, occlude, g).view(n_graphs, N_MAX, 1, 28, 28)
    imgs = imgs * valid[:, :, None, None, None]
    labelled = (torch.rand(n_graphs, N_MAX, generator=g) < label_frac) & valid
    # Wrong knowledge: add edges between nodes of the same colour, so that a fraction `wrong` of
    # the edges breaks "neighbours have different colours". Drawn last, so that wrong = 0 gives
    # the same graphs, images and labels as before.
    same_pairs = ((colour[:, :, None] == colour[:, None, :]) & valid[:, :, None] & valid[:, None, :]).float().triu(1)
    n_diff = adj.triu(1).sum((1, 2))
    want = wrong / (1 - wrong) * n_diff                     # same-colour edges per graph
    p_same = (want / same_pairs.sum((1, 2)).clamp(min=1)).clamp(max=1)[:, None, None]
    extra = (torch.rand(n_graphs, N_MAX, N_MAX, generator=g) < p_same).float() * same_pairs
    wrong_edge = extra + extra.transpose(1, 2)
    adj = adj + wrong_edge
    return dict(imgs=imgs, colour=colour * valid, adj=adj, valid=valid, labelled=labelled,
                wrong_edge=wrong_edge)


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def to_device(d):
    return {k: v.to(DEVICE) for k, v in d.items()}


# ----------------------------------------------------------------------------- models

class DigitNet(nn.Module):
    """The CNN with a softmax: one probability vector per node."""

    def __init__(self):
        super().__init__()
        self.cnn = CNN()

    def forward(self, x):
        return torch.softmax(self.cnn(x), -1)


class SageNet(nn.Module):
    """CNN features, then two GraphSAGE-style layers: h' = relu(W_s h + W_n mean_neighbours(h))."""

    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = CNN()
        self.cnn.net[-1] = nn.Identity()                  # keep the 64 features, drop the head
        self.s1, self.n1 = nn.Linear(64, hidden), nn.Linear(64, hidden)
        self.s2, self.n2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, K)

    def forward(self, imgs, adj):                          # [G, N, 1, 28, 28], [G, N, N] -> logits [G, N, K]
        G, N = imgs.shape[:2]
        h = self.cnn(imgs.flatten(0, 1)).view(G, N, -1)
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        h = F.relu(self.s1(h) + self.n1(adj @ h / deg))
        h = F.relu(self.s2(h) + self.n2(adj @ h / deg))
        return self.out(h)


sig = Signature("colouring")
for sort in ("Image", "Class", "Probs"):
    sig.sort(sort)
sig.dimension("Graph")
sig.structural_variable("g", "Graph")
sig.dimension("Node")
sig.structural_variable("n", "Node")
sig.constant("img", "Image", dims=["Graph", "Node"])
sig.constant("lab", "Class", dims=["Graph", "Node"])
sig.variable("c", "Class")
sig.function("digit", ["Image"], "Probs")
sig.predicate("Is", ["Probs", "Class"])
sig.structural_relation("Edge", ["Graph", "Node", "Node"])
sig.structural_relation("Labelled", ["Graph", "Node"])
sig.structural_relation("HasLabel", ["Graph"])

LABEL = sig.parse("∀ g | HasLabel(g): ∀ n | Labelled(g, n): Is(digit(img[g, n]), lab[g, n])")
RULE = sig.parse("∀ c: ∀ g: ∀ n, n1 | Edge(g, n, n1): ¬(Is(digit(img[g, n]), c) & Is(digit(img[g, n1]), c))")


def make_interp(p_agg, net):
    it = Interpretation(sig, logic=Logic().with_defaults())
    it.logic["∀,Graph"] = AggregPMeanError(p=p_agg)
    it.logic["∀,Node"] = AggregPMeanError(p=p_agg)
    it["Image"] = sltn.Type("image", shape=(1, 28, 28))
    it["Class"] = sltn.Type("class", shape=(K,))
    it["Probs"] = sltn.Type("probs", shape=(K,))
    it["digit"] = net
    it["Is"] = lambda p, c: (p * c).sum(-1, keepdim=True)
    it["c"] = torch.eye(K, device=DEVICE)
    return it


def ground(it, b):
    lab = b["labelled"].float()
    it["img"] = b["imgs"]
    it["lab"] = F.one_hot(b["colour"], K).float()
    it["Edge"] = b["adj"]
    it["Labelled"], it["HasLabel"] = lab, lab.amax(1)


# ----------------------------------------------------------------------------- training

def batches(train, batch, steps, g):
    for _ in range(steps):
        yield subset(train, torch.randint(0, len(train["imgs"]), (batch,), generator=g))


def train_cnn(train, a, seed, log=None):
    torch.manual_seed(seed)
    net = DigitNet().to(DEVICE)
    lab = train["labelled"]
    x, y = train["imgs"][lab], train["colour"][lab]        # only the labelled nodes matter
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(1, a.steps + 1):
        opt.zero_grad()
        loss = F.cross_entropy(net.cnn(x), y)
        loss.backward()
        opt.step()
        if log and step % LOG_EVERY == 0:
            log({"loss": loss.item()}, step)
    return lambda b: net(b["imgs"].flatten(0, 1)).view(*b["colour"].shape, K)


def train_gnn(train, a, seed, log=None):
    torch.manual_seed(seed)
    net = SageNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    has = train["labelled"].any(1)                         # graphs with a label: the only ones with a loss
    lab_graphs = subset(train, torch.nonzero(has).flatten())
    for step, b in enumerate(batches(lab_graphs, a.batch, a.steps, torch.Generator().manual_seed(seed)), 1):
        opt.zero_grad()
        logits = net(b["imgs"], b["adj"])
        loss = F.cross_entropy(logits[b["labelled"]], b["colour"][b["labelled"]])
        loss.backward()
        opt.step()
        if log and step % LOG_EVERY == 0:
            log({"loss": loss.item()}, step)
    return lambda b: torch.softmax(net(b["imgs"], b["adj"]), -1)


def train_sltn(train, a, seed, log=None):
    torch.manual_seed(seed)
    net = DigitNet().to(DEVICE)
    it = make_interp(a.p_agg, net).to(DEVICE)
    kb = KB([LABEL, RULE])
    opt = torch.optim.Adam(it.parameters(), lr=1e-3)
    for step, b in enumerate(batches(train, a.batch, a.steps, torch.Generator().manual_seed(seed)), 1):
        ground(it, b)
        opt.zero_grad()
        w = getattr(a, "rule_weight", None)
        if w is None:                                   # the KB: label clause and rule aggregated together
            loss = 1 - it(kb).value.squeeze()
        else:                                           # a softer rule: weighted against the labels
            loss = (1 - it(LABEL).value.squeeze()) + w * (1 - it(RULE).value.squeeze())
        loss.backward()
        opt.step()
        if log and step % LOG_EVERY == 0:
            with torch.no_grad():
                log({"loss": loss.item(), "sat_label": it(LABEL).value.item(), "sat_rule": it(RULE).value.item()}, step)
    return lambda b: net(b["imgs"].flatten(0, 1)).view(*b["colour"].shape, K)


@torch.no_grad()
def scores(predict, test, chunk=200):
    preds = torch.cat([predict(subset(test, slice(s, s + chunk))).argmax(-1)
                       for s in range(0, len(test["imgs"]), chunk)])
    v, col, adj = test["valid"], test["colour"], test["adj"]
    ok = (preds == col) | ~v
    same = (preds[:, :, None] == preds[:, None, :]).float() * adj
    right_edge = adj - test["wrong_edge"]                          # edges whose ends differ in colour
    touched = test["wrong_edge"].sum(-1) > 0                        # nodes on a same-colour edge
    acc_on = lambda m: (preds == col)[m].float().mean().item() if m.any() else float("nan")
    return dict(acc=(preds == col)[v].float().mean().item(),
                edge_viol=((same * right_edge).sum() / right_edge.sum()).item(),
                graph_ok=ok.all(1).float().mean().item(),
                acc_wrong_nodes=acc_on(touched & v), acc_other_nodes=acc_on(~touched & v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02])
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--p_agg", type=float, default=2)
    ap.add_argument("--models", nargs="+", default=["cnn", "gnn", "sltn"])
    a = ap.parse_args()
    pools = {"train": load_pool(True), "test": load_pool(False)}
    meta = dict(config=vars(a), sltn=sltn_version(), occlude=18, degree=3.0)
    out_dir = HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"sltn {meta['sltn']['commit'][:7]} dirty={meta['sltn']['dirty']} | {vars(a)}")
    for frac in a.fracs:
        for s in a.seeds:
            # the same graphs for every fraction; the labelled nodes are nested (same random draw)
            train = to_device(make_graphs(a.n_train, pools["train"], frac, torch.Generator().manual_seed(1000 + s)))
            test = to_device(make_graphs(a.n_test, pools["test"], 0.0, torch.Generator().manual_seed(5000 + s)))
            n_nodes = int(train["valid"].sum())
            print(f"labels {frac}, seed {s}: {n_nodes} nodes, {int(train['labelled'].sum())} labelled; "
                  f"mean degree {(train['adj'].sum() / n_nodes):.2f}", flush=True)
            for m in a.models:
                t0 = time.perf_counter()
                predict = {"cnn": train_cnn, "gnn": train_gnn, "sltn": train_sltn}[m](train, a, s)
                r = scores(predict, test)
                rows.append(dict(frac=frac, seed=s, model=m, n_labelled=int(train["labelled"].sum()), **r))
                (out_dir / "results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
                print(f"  {m:5s}: node acc {r['acc']:.3f} | edges with the same colour {r['edge_viol']:.3f} | "
                      f"graphs fully right {r['graph_ok']:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("\nmean ± std over seeds: node acc | edges with the same colour | graphs fully right")
    for frac in a.fracs:
        for m in a.models:
            sel = [x for x in rows if x["frac"] == frac and x["model"] == m]
            cell = lambda k: torch.tensor([x[k] for x in sel])
            ms = lambda t: f"{t.mean():.3f}±{t.std():.3f}" if len(t) > 1 else f"{t.mean():.3f}"
            print(f"{frac:<5} {m:5s} | {ms(cell('acc'))} | {ms(cell('edge_viol'))} | {ms(cell('graph_ok'))}")
        sl = {m: torch.tensor([x["acc"] for x in rows if x["frac"] == frac and x["model"] == m]) for m in a.models}
        if "sltn" in sl:
            for m in ("cnn", "gnn"):
                if m in sl:
                    print(f"      sltn beats {m} on {int((sl['sltn'] > sl[m]).sum())}/{len(sl[m])} seeds, "
                          f"smallest gain {100 * (sl['sltn'] - sl[m]).min():+.1f} points")
    print(f"saved {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

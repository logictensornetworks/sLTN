"""How much of a run's result comes from its dataset, and how much from its training seed.

repro.run draws the dataset and trains with the same seed. Here the two are separated: each dataset seed
draws one benchmark dataset (repro.make_dataset), and each training seed sets the initial weights and the batch
order on it. Everything else is repro's protocol (batches grouped by length, Adam 1e-4, batch 64, 50 epochs).

For each dataset, some statistics that may tell a hard dataset from an easy one are printed with every run:
the positive rate, and how often each block of the formula is satisfied only vacuously (its trigger never occurs).
Each run also reports the image accuracy per class.

Usage: python dataset_vs_training.py set1|set2|set2_exact K data_seed train_seeds   (train_seeds: comma-separated)
Prints one JSON line per run. The stored runs (an A40 GPU) are this output, one file per set and dataset seed:
results/dataset-vs-training-20261104-152631/ (Set 1 and Set 2) and results/dataset-vs-training-exact-u-20261106/
(set2_exact; its training seeds were run in two parts, kept in parts/, and joined).
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro import DEVICE, Data, Perception, make_dataset, make_interp, make_signature, sltn_version, value


def triggered(d, fset, k):
    """Whether block k's trigger occurs in the sequence d [L, 3] (otherwise the block holds vacuously)."""
    x, y, z = d[:, 0].tolist(), d[:, 1].tolist(), d[:, 2].tolist()
    if fset == "set1":
        return [0 in x, 2 in y or 3 in z, 4 in x, 6 in x or 5 in y][k]
    return [1 in y, 0 in y[1:] or 2 in z[1:], 4 in x, 6 in x or 4 in y][k]


def train_and_score(fset, K, data, train_seed, epochs=50, lr=1e-4, batch=64, run=None):
    C = 2 * K
    imgs, tr, te = data
    train, test = Data(tr, imgs), Data(te, imgs)
    sig, formula = make_signature(fset, K)
    torch.manual_seed(train_seed)
    net = Perception(C).to(DEVICE)
    it = make_interp(sig, C, net, fset).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    for step, (b_imgs, lab, _) in enumerate(train.epochs(batch, epochs, torch.Generator().manual_seed(train_seed)), 1):
        opt.zero_grad()
        loss = F.binary_cross_entropy(value(it, formula, b_imgs).clamp(1e-6, 1 - 1e-6), lab)
        loss.backward()
        opt.step()
        if run is not None and step % 50 == 0:
            run.log({"loss": loss.item()}, step=step)
    net.eval()
    ok = torch.zeros(C); n = torch.zeros(C); ok_seq = n_seq = 0
    with torch.no_grad():
        for b_imgs, lab, d in test.all_batches():
            pred = net(b_imgs.flatten(0, 2)).argmax(-1).view(d.shape)
            for c in range(C):
                m = d == c
                ok[c] += (pred[m] == c).sum().item(); n[c] += m.sum().item()
            ok_seq += ((value(it, formula, b_imgs) >= 0.5) == lab.bool()).sum().item(); n_seq += len(lab)
    return dict(img_acc=round(100 * ok.sum().item() / n.sum().item(), 2), seq_acc=round(100 * ok_seq / n_seq, 2),
                img_acc_per_class=[round(100 * (ok[c] / n[c]).item(), 1) for c in range(C)],
                seconds=round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    fset, K, data_seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    train_seeds = [int(s) for s in sys.argv[4].split(",")]
    data = make_dataset(fset, K, torch.Generator().manual_seed(data_seed))
    tr = data[1]
    lengths, digits = tr["length"], tr["digits"]
    seqs = [digits[i, :lengths[i]] for i in range(len(lengths))]
    stats = dict(positive_rate=round(tr["label"].float().mean().item(), 4),
                 mean_length=round(lengths.float().mean().item(), 3),
                 triggered_rate=[round(sum(triggered(d, fset, k) for d in seqs) / len(seqs), 4) for k in range(K)])
    for ts in train_seeds:
        config = dict(fset=fset, K=K, data_seed=data_seed, train_seed=ts, protocol="repro, grouped batches")
        res = train_and_score(fset, K, data, ts)
        print(json.dumps(dict(**config, **stats, **res, sltn=(sltn_version()["commit"] or "?")[:7], device=str(DEVICE))), flush=True)

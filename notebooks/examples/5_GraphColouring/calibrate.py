"""Graph colouring (5_GraphColouring.ipynb): how much to degrade the digits so that one node alone is ambiguous.

Trains a small CNN on n labelled digits (0, 1, 2; equal per class) under a degradation and
reports its test accuracy. Target: ~0.70-0.80 at the label counts of the experiment.

Run: conda run -n sltn python calibrate.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

DATA = Path(__file__).resolve().parents[2] / "datasets"     # MNIST, downloaded on first use
K = 3


def load(train):
    ds = datasets.MNIST(DATA, train=train, download=True)
    keep = ds.targets < K
    return ds.data[keep].float().div(255).unsqueeze(1), ds.targets[keep]


def degrade(x, noise, occlude, g):
    """Gaussian noise of std `noise`, then a random square of side `occlude` set to 0."""
    x = (x + noise * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if occlude:
        n = x.shape[0]
        r = torch.randint(0, 28 - occlude + 1, (n,), generator=g)
        c = torch.randint(0, 28 - occlude + 1, (n,), generator=g)
        rows = torch.arange(28)[None, :, None]
        cols = torch.arange(28)[None, None, :]
        box = ((rows >= r[:, None, None]) & (rows < (r + occlude)[:, None, None])
               & (cols >= c[:, None, None]) & (cols < (c + occlude)[:, None, None]))
        x = x * (~box).unsqueeze(1)
    return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, K))

    def forward(self, x):
        return self.net(x)


def run(n_lab, noise, occlude, seed):
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = load(True)
    idx = torch.cat([torch.nonzero(ytr == k).flatten()[torch.randperm((ytr == k).sum(), generator=g)[: n_lab // K]]
                     for k in range(K)])
    x, y = degrade(xtr[idx], noise, occlude, g), ytr[idx]
    xte, yte = load(False)
    xte = degrade(xte, noise, occlude, torch.Generator().manual_seed(99))
    torch.manual_seed(seed)
    m = CNN()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
    with torch.no_grad():
        return (m(xte).argmax(-1) == yte).float().mean().item()


if __name__ == "__main__":
    print("labelled digits | degradation | test accuracy, 3 seeds")
    for n_lab in (30, 90):
        for noise, occ in [(0, 0), (0.5, 0), (0.8, 0), (1.0, 0), (0, 14), (0, 18), (0.5, 14)]:
            accs = [run(n_lab, noise, occ, s) for s in range(3)]
            print(f"{n_lab:5d} | noise {noise:.1f}, occlusion {occ:2d} | "
                  + " ".join(f"{a:.3f}" for a in accs) + f"  mean {sum(accs) / 3:.3f}", flush=True)

"""Prototype of 4_FeverWard.ipynb: sparse day labels, persistence rules, stays of different lengths.

A ward: each patient is admitted with fever, which would last 1-6 days. 60% of the patients
are treated on one of the first three days of fever (day 0, 1 or 2), and a treatment brings the
fever down within two days: no fever from two days after it. The patient stays 1-4 more days
after recovery. With probability `relapse`, a 1-2 day relapse happens after recovery, never on
the day the treatment rule speaks about (so only the "no relapse" rule is then wrong).
One noisy temperature per day (std `--noise`); a fraction `label_frac` of the days (never day 0)
is labelled. The treatment days are observed: every model gets them as an input, next to the
temperatures, and the "treatment works" rule says what they imply. `Fever` is learned with the labels only, or with the labels and the rules.

Run: conda run -n sltn python prototype.py [--seeds 0 1 2] [--quick]
"""

import argparse
import itertools
import time

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sltn
from sltn import KB, Interpretation, Logic, Signature
from sltn.interpretation.operators.aggregator import AggregPMeanError


# ----------------------------------------------------------------------------- data

def make_ward(n, label_frac, relapse, fixed_len, g, noise=0.5):
    """Stays padded to the longest one. Returns a dict of [n, T] tensors."""
    dur = torch.randint(1, 7, (n,), generator=g)                  # fever days 1..6, untreated
    treat_day = torch.randint(0, 3, (n,), generator=g)            # day 0, 1 or 2
    treated = (torch.rand(n, generator=g) < 0.6) & (treat_day < dur)   # treated during the fever
    dur = torch.where(treated, torch.minimum(dur, treat_day + 2), dur)  # no fever from treat_day + 2
    stay_after = torch.randint(1, 5, (n,), generator=g)           # days after recovery 1..4
    has_relapse = torch.rand(n, generator=g) < relapse
    rel_len = torch.randint(1, 3, (n,), generator=g)              # relapse lasts 1..2 days
    rel_gap = torch.randint(1, 3, (n,), generator=g)              # 1..2 days without fever first
    # a relapse starts after its gap, and for a treated patient after treat_day + 2
    rel_start = torch.where(treated, torch.maximum(dur + rel_gap, treat_day + 3), dur + rel_gap)
    length = torch.where(has_relapse, rel_start + rel_len, dur) + stay_after
    if fixed_len is not None:
        length = torch.full((n,), fixed_len)
    T = int(length.max())
    days = torch.arange(T)[None]
    valid = days < length[:, None]
    fever = days < dur[:, None]
    rs = rel_start[:, None]
    fever |= has_relapse[:, None] & (days >= rs) & (days < rs + rel_len[:, None])
    if fixed_len is not None:                                     # keep the last day without fever
        fever &= days < fixed_len - 1
    fever &= valid
    temp = 36.9 + 1.2 * fever.float() + noise * torch.randn(n, T, generator=g)
    labelled = (torch.rand(n, T, generator=g) < label_frac) & valid
    labelled[:, 0] = False
    treated_on = treated[:, None] & (days == treat_day[:, None]) & valid
    return dict(temp=temp, valid=valid, fever=fever, labelled=labelled, length=length,
                has_relapse=has_relapse, treated_on=treated_on, recovery=dur)


# ----------------------------------------------------------------------------- model

class FeverNet(nn.Module):
    """Reads a whole stay [B, Day, 3] = (temperature, treated, valid) and gives one degree per day.
    Padded days are skipped by packing, and get 0."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 16, batch_first=True, bidirectional=True)
        self.out = nn.Linear(32, 1)

    def forward(self, v):
        temp = torch.cat([v[..., :1] - 37.5, v[..., 1:2]], -1)    # temperature, treated
        valid = v[..., 2]
        lengths = valid.sum(1).long().clamp(min=1).cpu()
        h, _ = self.gru(pack_padded_sequence(temp, lengths, batch_first=True, enforce_sorted=False))
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=v.shape[1])
        return torch.sigmoid(self.out(h)) * valid.unsqueeze(-1)


sig = Signature("ward")
sig.sort("Reading")
sig.dimension("Patient")
sig.structural_variable("p", "Patient")
sig.dimension("Day")
sig.structural_variable("d", "Day")
sig.constant("obs", "Reading", dims=["Patient", "Day"])
sig.predicate("Fever", ["Reading"], input_dims=["Day"], output_dims=["Day"])
for name, dims in [("Sick", ["Patient", "Day"]), ("Healthy", ["Patient", "Day"]),
                   ("HasSick", ["Patient"]), ("HasHealthy", ["Patient"]),
                   ("Next", ["Patient", "Day", "Day"]), ("AfterTreatment", ["Patient", "Day", "Day"]),
                   ("HasTreated", ["Patient"])]:
    sig.structural_relation(name, dims)

LABELS = {
    # patients with no labelled day are left out: otherwise they count as satisfied and dilute the labels
    "sick": sig.parse("∀ p | HasSick(p): ∀ d | Sick(p, d): Fever(obs[p, d])[d]"),
    "healthy": sig.parse("∀ p | HasHealthy(p): ∀ d | Healthy(p, d): ¬Fever(obs[p, d])[d]"),
}
RULES = {
    "admitted with fever": sig.parse("∀ p: Fever(obs[p, d])[d][d=0]"),
    "no relapse": sig.parse("∀ p: ∀ d, d1 | Next(p, d, d1): (Fever(obs[p, d1])[d1] -> Fever(obs[p, d])[d])"),
    "treatment works": sig.parse(
        "∀ p | HasTreated(p): ∀ d, d1 | AfterTreatment(p, d, d1): ¬Fever(obs[p, d1])[d1]"),
}


def ground(it, w):
    n, T = w["temp"].shape
    valid = w["valid"].float()
    days = torch.arange(T)
    nxt = (days[:, None] + 1 == days[None]).float()
    two = (days[:, None] + 2 == days[None]).float()
    treated = w["treated_on"].float()
    sick = (w["labelled"] & w["fever"]).float()
    healthy = (w["labelled"] & ~w["fever"]).float()
    it["obs"] = torch.stack([w["temp"] * valid, w["treated_on"].float(), valid], -1)
    it["Sick"], it["Healthy"] = sick, healthy
    it["HasSick"], it["HasHealthy"] = sick.amax(1), healthy.amax(1)
    it["Next"] = nxt[None] * valid[:, None, :]          # consecutive days, both real
    # d is a treatment day and d1 = d + 2 is in the stay. One precomputed relation, not the
    # compound guard `TwoLater(p, d, d1) & Treated(p, d)`: under the stable product an empty
    # compound guard keeps every pair with weight 1e-6, so it averages over all pairs instead.
    after = treated[:, :, None] * two[None] * valid[:, None, :]
    it["AfterTreatment"], it["HasTreated"] = after, after.amax((1, 2))


def make_interp(p_agg=4):
    it = Interpretation(sig, logic=Logic().with_defaults())
    it.logic["∀,Patient"] = AggregPMeanError(p=p_agg)
    it.logic["∀,Day"] = AggregPMeanError(p=p_agg)
    it["Reading"] = sltn.Type("reading", shape=(3,))          # temperature, treated, valid
    it["Fever"] = FeverNet()
    return it


@torch.no_grad()
def scores(it, w):
    ground(it, w)
    pred = it("Fever(obs[p, d])[d]").value.squeeze(-1) > 0.5          # axes p, d
    v = w["valid"]
    acc = (pred == w["fever"])[v].float().mean().item()
    # rule violations, thresholded, on the predictions
    adm = (~pred[:, 0]).float().mean().item()
    starts = (pred[:, 1:] & ~pred[:, :-1] & v[:, 1:]).any(1)            # a fever that starts after day 0
    tr = w["treated_on"]
    after = torch.zeros_like(tr)
    after[:, 2:] = tr[:, :-2]                                           # two days after a treatment
    treat = (pred & after & v).any(1)[tr.any(1)].float().mean().item()  # among treated patients
    # accuracy on the days each rule speaks about
    ok = pred == w["fever"]
    T = v.shape[1]
    day0 = ok[:, 0].float().mean().item()                               # admitted with fever
    treat_day = ok[after & v].float().mean().item()                     # two days after a treatment
    post = (torch.arange(T)[None] >= w["recovery"][:, None]) & v        # no relapse: after the first recovery
    post_rec = ok[post].float().mean().item()
    return dict(acc=acc, viol_relapse=starts.float().mean().item(), viol_admit=adm, viol_treat=treat,
                acc_day0=day0, acc_treat_day=treat_day, acc_post_recovery=post_rec)


def train(w, clauses, seed, epochs):
    torch.manual_seed(seed)
    it = make_interp()
    ground(it, w)
    kb = KB(list(clauses.values()))
    opt = torch.optim.Adam(it.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = 1 - it(kb).value.squeeze()
        loss.backward()
        opt.step()
    return it, loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.5)      # std of a reading, in degrees
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    fracs = [0.05, 0.02, 0.01]
    relapses = [0.0, 0.1, 0.3]
    lens = [None, 10]                      # variable lengths, and the fixed-length control
    if a.quick:
        fracs, relapses, lens = [0.05], [0.0], [None]
    print(f"sltn {sltn.__version__}, n_train {a.n_train}, noise {a.noise}, epochs {a.epochs}, seeds {a.seeds}")
    print(f"{'length':8s} {'labels':6s} {'relapse':7s} | {'acc, labels only':16s} {'acc, + rules':12s} | "
          f"violations (relapse / admit / treatment), labels only -> + rules")
    for fixed, frac, rel in itertools.product(lens, fracs, relapses):
        res = {"labels": [], "rules": []}
        t0 = time.perf_counter()
        for s in a.seeds:
            g = torch.Generator().manual_seed(1000 + s)
            tr = make_ward(a.n_train, frac, rel, fixed, g, a.noise)
            te = make_ward(2000, frac, rel, fixed, torch.Generator().manual_seed(5000 + s), a.noise)
            for name, cl in [("labels", LABELS), ("rules", {**LABELS, **RULES})]:
                it, _ = train(tr, cl, s, a.epochs)
                res[name].append(scores(it, te))
        m = lambda k, key: torch.tensor([r[key] for r in res[k]])
        fmt = lambda t: f"{t.mean():.3f}±{t.std():.3f}" if len(t) > 1 else f"{t.mean():.3f}"
        viol = lambda k: " / ".join(f"{m(k, v).mean():.2f}" for v in ("viol_relapse", "viol_admit", "viol_treat"))
        print(f"{'var' if fixed is None else fixed:8} {frac:<6} {rel:<7} | {fmt(m('labels', 'acc')):16s} "
              f"{fmt(m('rules', 'acc')):12s} | {viol('labels')} -> {viol('rules')}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()

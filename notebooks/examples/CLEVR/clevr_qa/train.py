"""Learn colour, size, material and shape from CLEVR images, from the answers only.

A CNN (networks.ConceptNet) embeds each object's crop; the attribute predicates read its heads. Each step
samples families in proportion to their size, a batch of questions from each, and maximises the mean
truth value of the families' rules (loss = 1 - mean). R is either given (CLEVR's own relations, computed by
its rule from the 3D coordinates), or, with --learn_relations, a small network learned at the same time from
the objects' pixel coordinates (column, row, depth) only. No attribute or relation label is used;
the scene files' annotations only score the result (evaluation.py) on the val split.

Needs the crops of prepare_crops.py. Run (from notebooks/examples/CLEVR; GPU):
    python -m clevr_qa.train [--steps 4500] [--learn_relations] [--seed 0] [--no_wandb]
Each run writes results.json and the networks to RESULTS/<date>-<tag>.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

import sltn

from .batching import family_tensors
from .data import CROPS, RESULTS, camera_directions, load, load_crops, scene_arrays, subset
from .evaluation import concept_acc, qa_acc, relation_acc
from .formulas import make_signature
from .interpretation import embed_scenes, ground, learned_interpretation
from .networks import ConceptNet, Relation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split, crops_dir, pos, max_scenes=None, n_questions=None, seed=0, device=DEVICE):
    """{scene_arrays' tensors, "pos", crops, fams} of a split, on device, and the camera directions.
    pos: which position goes into scene[o], "pixel" or "xyz" (networks.Relation.reads)."""
    scenes, questions = load(split)
    directions = camera_directions(scenes)
    if max_scenes:
        scenes = scenes[:max_scenes]
        questions = [q for q in questions if q["image_index"] < max_scenes]
    if n_questions:
        questions = subset(questions, n_questions, seed)
    out = {k: v.to(device) for k, v in scene_arrays(scenes).items()}
    out["pos"] = out[pos]
    out["crops"] = load_crops(split, len(scenes), crops_dir).to(device)
    out["fams"] = family_tensors(questions, device)
    return out, directions


def sltn_version():
    """Commit, branch and dirty flag of the sLTN checkout in use, saved with every result. On the cluster (no
    git) they come from the VERSION.json the launcher writes at the root of the code's copy."""
    root = Path(sltn.__file__).resolve().parent.parent
    if (root / "VERSION.json").exists():
        return json.loads((root / "VERSION.json").read_text())
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain", "--untracked-files=no")))


def start_tracking(config, seed, enabled):
    """W&B through sltn-applications' common.tracking (GPU runs only); None if disabled or unavailable."""
    if not enabled:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "sltn-applications"))
        from common.tracking import start_run
    except ImportError as e:
        print(f"no W&B tracking ({e})", flush=True)
        return None
    return start_run("clevr-e3", config, seed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--fams_per_step", type=int, default=8)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_q", type=int, default=None, help="train on N in-scope questions (default: all)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_scenes", type=int, default=2000, help="val scenes for the periodic concept accuracy")
    ap.add_argument("--learn_relations", action="store_true", help="learn R from the pixel coordinates")
    ap.add_argument("--crops", type=Path, default=CROPS, help="folder with <split>_crops.npy")
    ap.add_argument("--max_scenes", type=int, default=None, help="smoke tests: keep only the first N scenes per split")
    ap.add_argument("--results", type=Path, default=RESULTS, help="where the run folder goes")
    ap.add_argument("--no_wandb", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    pos = "pixel" if a.learn_relations else "xyz"                  # what R reads (networks.Relation.reads)
    tr, directions = load_split("train", a.crops, pos, a.max_scenes, a.n_train_q, a.seed)
    va, _ = load_split("val", a.crops, pos, a.max_scenes)
    for name, sp in (("train", tr), ("val", va)):
        n_q = sum(len(t["scene_idx"]) for _, t in sp["fams"].values())
        print(f"{name}: {len(sp['crops'])} scenes, {n_q} in-scope questions ({time.perf_counter() - t0:.0f}s)", flush=True)

    sig = make_signature()
    net = ConceptNet().to(DEVICE)
    relation = Relation(directions, a.learn_relations).to(DEVICE)
    assert relation.reads == pos
    it = learned_interpretation(sig, net, relation, DEVICE)
    opt = torch.optim.Adam(list(net.parameters()) + list(relation.parameters()), lr=a.lr)
    keys = list(tr["fams"])
    rules = {k: sig.parse(tr["fams"][k][0].rule) for k in keys}
    sizes = torch.tensor([len(tr["fams"][k][1]["scene_idx"]) for k in keys], dtype=torch.float)
    config = dict({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                  n_train_inscope=int(sizes.sum()), crop=48, crop_k=500, relation_input=pos, device=str(DEVICE))
    tag = ("-relations" if a.learn_relations else "") + (f"-n{a.n_train_q}" if a.n_train_q else "") + f"-seed{a.seed}"
    out = a.results / (datetime.now().strftime("%Y%m%d-%H%M%S") + tag)
    out.mkdir(parents=True, exist_ok=True)
    print(f"sltn {sltn_version()['commit'][:7]} | {config}", flush=True)
    run = start_tracking(config, a.seed, DEVICE.type == "cuda" and not a.no_wandb)

    run_loss, t1 = 0.0, time.perf_counter()
    for step in range(1, a.steps + 1):
        fam_idx = torch.multinomial(sizes, min(a.fams_per_step, len(keys)), replacement=False, generator=g).tolist()
        picks = []
        for fi in fam_idx:
            t = tr["fams"][keys[fi]][1]
            picks.append((fi, torch.randint(0, len(t["scene_idx"]), (a.batch,), generator=g).to(DEVICE)))
        scene_ids = torch.cat([tr["fams"][keys[fi]][1]["scene_idx"][idx] for fi, idx in picks])
        emb = embed_scenes(net, tr["crops"][scene_ids], tr["present"][scene_ids])   # every object, once
        sats = []
        for j, (fi, idx) in enumerate(picks):
            c, t = tr["fams"][keys[fi]]
            ground(it, c, t, idx, emb[j * a.batch:(j + 1) * a.batch], tr)
            sats.append(it(rules[keys[fi]]).value.squeeze())
        loss = 1 - torch.stack(sats).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run_loss += loss.item()
        if step % a.eval_every == 0:
            acc = concept_acc(net, va, a.eval_scenes)
            racc = relation_acc(relation, va, a.eval_scenes) if a.learn_relations else {}
            ms = (time.perf_counter() - t1) / step * 1000
            print(f"step {step:5d} loss {run_loss / a.eval_every:.4f} | concept acc (val, {a.eval_scenes} scenes): "
                  + ", ".join(f"{k} {v:.3f}" for k, v in acc.items())
                  + "".join(f", {k} {v:.3f}" for k, v in racc.items()) + f" | {ms:.0f} ms/step", flush=True)
            if run is not None:
                run.log({"loss": run_loss / a.eval_every, **{f"concept_acc/{k}": v for k, v in acc.items()},
                         **{f"relation_acc/{k}": v for k, v in racc.items()}}, step=step)
            run_loss = 0.0

    acc = concept_acc(net, va)
    qa, per = qa_acc(net, it, sig, va)
    racc = relation_acc(relation, va) if a.learn_relations else {}
    summary = {**{f"concept_acc/{k}": v for k, v in acc.items()}, **{f"relation_acc/{k}": v for k, v in racc.items()},
               "qa_acc": qa, **{f"qa_acc/{k}": v for k, v in per.items()}}
    print("\nfinal concept acc (all val objects): " + ", ".join(f"{k} {v:.4f}" for k, v in acc.items()))
    if racc:
        print("final relation acc (all val pairs): " + ", ".join(f"{k} {v:.4f}" for k, v in racc.items()))
    print(f"final QA acc (val, in scope): {qa:.4f} | " + ", ".join(f"{k} {v:.4f}" for k, v in per.items()))
    (out / "results.json").write_text(json.dumps(dict(config=config, sltn=sltn_version(), **summary), indent=1))
    torch.save(net.state_dict(), out / "concept_net.pt")
    if a.learn_relations:
        torch.save(relation.state_dict(), out / "relation_net.pt")
    if run is not None:
        run.summary.update(summary)
        run.finish()
    print(f"saved {out}; total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

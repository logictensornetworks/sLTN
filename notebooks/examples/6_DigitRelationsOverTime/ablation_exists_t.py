# Ablation for 6_DigitRelationsOverTime.ipynb (Section 4): does the ∃ over the later steps need to spread its gradient?
# Rule + 100 labelled digits, 800 sequences, product semantics everywhere except "exists,t":
# the probabilistic sum (as in the notebook) vs max. It runs the notebook's own cells, so it uses the same code.
#
#   python ablation_exists_t.py less 0,1,2,3,4 less.json
#   python ablation_exists_t.py sum 0,1,2 sum.json
#
# Results of 2026-11-06 (A40): results/ablation-exists-t-20261106/.
import json, os, sys
from pathlib import Path

relation, seeds, out = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], Path(sys.argv[3]).resolve()
os.chdir(Path(__file__).resolve().parent)        # the notebook finds the repo root from the working directory
nb = json.load(open("6_DigitRelationsOverTime.ipynb"))
g = {"__name__": "ablation"}
for i in (1, 3, 7, 9, 11, 13, 17):
    exec("".join(nb["cells"][i]["source"]), g)
EXISTS_T = {"probsum": g["AggregProbSum"], "max": g["AggregMax"]}
# Rebuild make_interp's source with the swap, so nothing is set after construction.
src = "".join(nb["cells"][13]["source"])
old = 'lg["forall,t"], lg["exists,t"], lg["exists"] = AggregProd(), AggregProbSum(), AggregProbSum()'
assert src.count(old) == 1
src = src.replace(old, 'lg["forall,t"], lg["exists,t"], lg["exists"] = AggregProd(), EXISTS_T_CLS(), AggregProbSum()')
g["EXISTS_T_CLS"] = None
exec(src, g)

rows = []
pools = g["pools"]
for seed in seeds:
    sets = g["make_sets"](relation, 800, seed, pools)
    for v in ("probsum", "max"):
        g["EXISTS_T_CLS"] = EXISTS_T[v]
        it_check = g["make_interp"]("prod", g["DigitNet"]())
        r = g["evaluate"](f"{relation} seed {seed}, ∃ over steps = {v}", sets, relation, labels=True, seed=seed)
        r = {k: val for k, val in r.items() if k != "confusion"}
        r.update(relation=relation, seed=seed, exists_t=v, logic=repr(it_check.logic))
        rows.append(r)
        out.write_text(json.dumps(rows, indent=1))
print("done", flush=True)

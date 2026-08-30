"""Build a dev set for iteration (eval set stays frozen & rarely touched).

Dev queries: the 150 most-popular src anime NOT in the eval set having >= 10
rec-graph targets; truth = their top-10 targets by vote count.
Then rewrite train_pairs.parquet with edges incident to eval OR dev removed,
so supervised models can't memorize either set's truth.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_metadata  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
N_DEV = 150

eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
pairs = pd.read_parquet(DATA / "rec_pairs.parquet")
meta = load_metadata()

counts = pairs.groupby("src").size()
cand = [s for s in counts[counts >= 10].index
        if s not in eval_ids and s in meta and meta[s]["popularity"]]
cand.sort(key=lambda s: meta[s]["popularity"])
dev_ids = cand[:N_DEV]

dev = {}
for s in dev_ids:
    g = pairs[pairs.src == s].sort_values("votes", ascending=False)
    dev[str(s)] = [int(x) for x in g.dst.head(10)]
json.dump({"queries": dev}, open(DATA / "dev_set.json", "w"), indent=1)
print(f"dev set: {len(dev)} queries")

# NOTE: train-pair holdout files are written ONLY by build_enriched_meta.py
# (single-writer rule; a build_eval_set.py rerun once silently restored
# dev edges into training).

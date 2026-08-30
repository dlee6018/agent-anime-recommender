"""Freeze the eval set from scraped userrecs + top-200 popularity list.

Takes the 100 most popular anime having >= MIN_RECS scraped recommendations,
stores their top-10 lists as immutable ground truth, and writes the
training-pair file with ALL edges incident to eval anime removed
(the rec graph is symmetric — filter both directions).
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
MIN_RECS = 10
N_EVAL = 100

top200 = json.load(open(DATA / "top200_popularity.json"))
userrecs = json.load(open(DATA / "userrecs.json"))

queries = {}
for a in top200:
    mid = str(a["mal_id"])
    recs = userrecs.get(mid)
    if recs and len(recs) >= MIN_RECS:
        queries[mid] = recs[:10]
    if len(queries) == N_EVAL:
        break

out = {"frozen": date.today().isoformat(), "min_recs": MIN_RECS,
       "queries": queries}
json.dump(out, open(DATA / "eval_set.json", "w"), indent=1)
print(f"eval set frozen: {len(queries)} queries")

eval_ids = {int(q) for q in queries}
pairs = pd.read_parquet(DATA / "rec_pairs.parquet")
before = len(pairs)
pairs = pairs[~pairs.src.isin(eval_ids) & ~pairs.dst.isin(eval_ids)]
pairs.to_parquet(DATA / "train_pairs.parquet")
print(f"train pairs: {before} -> {len(pairs)} "
      f"({before - len(pairs)} edges incident to eval anime removed)")

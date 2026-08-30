"""Merge freshly-scraped userrecs lists (order only) into the rec-pair graph.

Scraped lists get pseudo-votes from the empirical ayan rank->median-vote
curve. Real ayan votes win on edge collision. Regenerates the two
holdout-filtered training files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

ayan = pd.read_parquet(DATA / "rec_pairs.parquet")

# empirical rank -> median vote curve from ayan lists
curves = {}
for s, g in ayan.groupby("src"):
    for r, v in enumerate(g.sort_values("votes", ascending=False)
                          .votes.head(40)):
        curves.setdefault(r, []).append(v)
rank_vote = {r: max(1, int(np.median(v))) for r, v in curves.items()}
print("rank->pseudo-vote:", {r: rank_vote[r] for r in list(rank_vote)[:10]})

userrecs = json.load(open(DATA / "userrecs.json"))
rows = []
for src, lst in userrecs.items():
    if not lst:
        continue
    for r, dst in enumerate(lst[:40]):
        rows.append((int(src), int(dst), rank_vote.get(r, 1)))
scr = pd.DataFrame(rows, columns=["src", "dst", "votes"])
print(f"scraped: {len(scr)} edges from {scr.src.nunique()} srcs")

merged = pd.concat([ayan, scr]).sort_values("votes", ascending=False) \
    .drop_duplicates(["src", "dst"], keep="first")
merged.to_parquet(DATA / "rec_pairs_merged.parquet")
print(f"merged: {len(merged)} edges, {merged.src.nunique()} srcs "
      f"(ayan was {len(ayan)}/{ayan.src.nunique()})")

eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
dev_ids = {int(q) for q in json.load(open(DATA / "dev_set.json"))["queries"]}
pe = merged[~merged.src.isin(eval_ids) & ~merged.dst.isin(eval_ids)]
pe.to_parquet(DATA / "train_pairs_eval.parquet")
held = eval_ids | dev_ids
pdv = merged[~merged.src.isin(held) & ~merged.dst.isin(held)]
pdv.to_parquet(DATA / "train_pairs.parquet")
print(f"train_pairs_eval {len(pe)}, train_pairs(dev) {len(pdv)}")

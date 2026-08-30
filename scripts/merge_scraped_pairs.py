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

# monotone pseudo-votes preserve the scraped page order (most-recommended
# first); flat median-based votes destroyed it and diluted training (exp 27)
userrecs = json.load(open(DATA / "userrecs.json"))
rows = []
for src, lst in userrecs.items():
    if not lst:
        continue
    for r, dst in enumerate(lst[:40]):
        rows.append((int(src), int(dst), max(1, 40 - r)))
scr = pd.DataFrame(rows, columns=["src", "dst", "votes"])
print(f"scraped: {len(scr)} edges from {scr.src.nunique()} srcs")

# product graph: everything (ayan real votes win on collision)
merged = pd.concat([ayan, scr]).drop_duplicates(["src", "dst"], keep="first")
merged.to_parquet(DATA / "rec_pairs_merged.parquet")
print(f"product graph: {len(merged)} edges, {merged.src.nunique()} srcs")

# training graph: ayan + scraped lists of NEW srcs only (tail edges of
# already-covered srcs add 1-vote noise — exp 27 showed a net dev loss)
new_srcs = set(scr.src.unique()) - set(ayan.src.unique())
train_graph = pd.concat([ayan, scr[scr.src.isin(new_srcs)]])
train_graph = train_graph.drop_duplicates(["src", "dst"], keep="first")
print(f"training graph: {len(train_graph)} edges, "
      f"{train_graph.src.nunique()} srcs (+{len(new_srcs)} new)")

eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
dev_ids = {int(q) for q in json.load(open(DATA / "dev_set.json"))["queries"]}
pe = train_graph[~train_graph.src.isin(eval_ids)
                 & ~train_graph.dst.isin(eval_ids)]
pe.to_parquet(DATA / "train_pairs_eval.parquet")
held = eval_ids | dev_ids
pdv = train_graph[~train_graph.src.isin(held) & ~train_graph.dst.isin(held)]
pdv.to_parquet(DATA / "train_pairs.parquet")
print(f"train_pairs_eval {len(pe)}, train_pairs(dev) {len(pdv)}")

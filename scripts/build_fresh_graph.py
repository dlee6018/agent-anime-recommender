"""Rebuild the rec graph with FRESH per-pair vote counts (userrecs_votes.json).

For scraped srcs (top ~2,500 by popularity) the fresh page is authoritative
and replaces the March ayan list entirely; unscraped srcs keep ayan rows.
Regenerates: rec_pairs_fresh.parquet (product + feature graph) and the four
holdout-filtered training files. eval_set.json / dev_set.json stay frozen.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

ayan = pd.read_parquet(DATA / "rec_pairs.parquet")
fresh_raw = json.load(open(DATA / "userrecs_votes.json"))

rows = []
for src, lst in fresh_raw.items():
    if not lst:
        continue
    for dst, votes in lst[:60]:
        rows.append((int(src), int(dst), int(votes)))
fresh = pd.DataFrame(rows, columns=["src", "dst", "votes"])
scraped = set(fresh.src.unique())
keep_ayan = ayan[~ayan.src.isin(scraped)]
graph = pd.concat([fresh, keep_ayan]).drop_duplicates(["src", "dst"])
graph.to_parquet(DATA / "rec_pairs_fresh.parquet")
print(f"fresh graph: {len(graph)} edges, {graph.src.nunique()} srcs "
      f"({len(fresh)} fresh-vote edges from {len(scraped)} scraped srcs)")

eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
dev_ids = {int(q) for q in json.load(open(DATA / "dev_set.json"))["queries"]}
held = eval_ids | dev_ids

graph[~graph.src.isin(eval_ids) & ~graph.dst.isin(eval_ids)] \
    .to_parquet(DATA / "train_pairs_eval.parquet")
graph[~graph.src.isin(held) & ~graph.dst.isin(held)] \
    .to_parquet(DATA / "train_pairs.parquet")
graph[~graph.src.isin(held)] \
    .to_parquet(DATA / "train_pairs_srconly_dev.parquet")
for f in ("train_pairs_eval", "train_pairs", "train_pairs_srconly_dev"):
    n = len(pd.read_parquet(DATA / f"{f}.parquet"))
    print(f"{f}: {n}")

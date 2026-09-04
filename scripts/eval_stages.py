"""Stage-isolated evaluation: retrieval pool vs reranker vs oracle ceiling.

For each dev query (under the chosen holdout condition) reports:
  recall@pool   — fraction of truth top-10 present in the candidate pool
                  (the reranker's hard ceiling)
  P@5 retrieval — pool in tower-cosine order (no reranker)
  P@5 rerank    — after LightGBM ordering
  P@5 oracle    — perfect ordering of whatever truth is in the pool
  NDCG@10 for retrieval and rerank (binary gains, log2 discount)

Usage: eval_stages.py [--mode src_only|strict|bare] [--booster FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata  # noqa: E402
from src.franchise import franchise_filter, same_franchise  # noqa: E402
from src.models.rerank import FeatureBuilder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["src_only", "strict", "bare"],
                default="src_only")
ap.add_argument("--booster", default=None)
ap.add_argument("--emb", default="tt_ens_emb.npz")
ap.add_argument("--maxrank", type=int, default=8000)
ap.add_argument("--top_cand", type=int, default=250)
ap.add_argument("--union_extra", type=int, default=100)
args = ap.parse_args()

DATA = ROOT / "data"
booster_file = args.booster or (
    "reranker_strict2.txt" if args.mode == "bare" else
    "reranker_union_strict.txt" if args.mode == "strict" else
    "reranker_union.txt")
booster = lgb.Booster(model_file=str(DATA / booster_file))

meta = load_metadata()
ids = np.load(DATA / "content_emb.npz")["ids"]
emb = np.load(DATA / args.emb)["emb"].astype(np.float32)
fb = FeatureBuilder(ids)
if args.mode != "bare":
    al = DATA / "anilist_recs.json"
    if al.exists():
        fb.set_anilist(json.load(open(al)))

pairs = pd.read_parquet(DATA / "rec_pairs.parquet")
dev = {int(q): [int(r) for r in v]
       for q, v in json.load(open(DATA / "dev_set.json"))["queries"].items()}
ev_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
held = ev_ids | set(dev)
if args.mode == "src_only":
    graph = pairs[~pairs.src.isin(held)]
else:  # strict/bare: symmetric removal for dev+eval
    graph = pairs[~pairs.src.isin(held) & ~pairs.dst.isin(held)]
fb.set_graph(graph)

idx = fb.idx
pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999)
                for a in ids])
rmask = pop <= args.maxrank


def pool_and_orders(q):
    tq = emb[idx[q]]
    sim = emb @ tq
    sim[~rmask] = -np.inf
    sim[idx[q]] = -np.inf
    top = np.argpartition(-sim, args.top_cand)[:args.top_cand]
    top = top[np.argsort(-sim[top])]
    cands = [int(ids[i]) for i in top]
    seen = set(cands)
    extras = [fb.cooc_top(q, args.union_extra, rmask),
              fb.content_top(q, args.union_extra, rmask)]
    if args.mode != "bare":
        extras.insert(0, [d for d, _ in fb.al_out.get(q, [])])
        extras.append([s for s, _ in sorted(fb.in_lists.get(q, ()),
                                            key=lambda x: -x[1])
                       [:args.union_extra]])
    for ex in extras:
        cands.extend(c for c in ex if c not in seen and c in idx)
        seen.update(ex)
    keep = franchise_filter([q])
    cands = [c for c in cands if keep(c)]
    Xr = fb.rows(q, cands, tq, emb)
    scores = booster.predict(Xr)
    rerank_order = [cands[i] for i in np.argsort(-scores)]
    return cands, rerank_order


def dedupe_franchise(lst, k):
    out = []
    for c in lst:
        if any(same_franchise(c, o) for o in out):
            continue
        out.append(c)
        if len(out) == k:
            break
    return out


def ndcg10(ranked, t10):
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(ranked[:10])
              if r in t10)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(t10))))
    return dcg / ideal


agg = {k: [] for k in ("recall_pool", "p5_ret", "p5_rr", "p5_oracle",
                       "ndcg_ret", "ndcg_rr")}
for q, truth in dev.items():
    if q not in idx:
        continue
    t10 = set(truth[:10])
    pool, rr = pool_and_orders(q)
    ret5 = dedupe_franchise(pool, 5)
    rr5 = dedupe_franchise(rr, 5)
    in_pool = [c for c in pool if c in t10]
    agg["recall_pool"].append(len(set(pool) & t10) / 10)
    agg["p5_ret"].append(sum(c in t10 for c in ret5) / 5)
    agg["p5_rr"].append(sum(c in t10 for c in rr5) / 5)
    agg["p5_oracle"].append(min(len(in_pool), 5) / 5)
    agg["ndcg_ret"].append(ndcg10(dedupe_franchise(pool, 10), t10))
    agg["ndcg_rr"].append(ndcg10(dedupe_franchise(rr, 10), t10))

print(f"mode={args.mode}  booster={booster_file}  n={len(agg['p5_rr'])}")
for k, v in agg.items():
    print(f"  {k:12} {np.mean(v):.3f}")
print(f"  uplift rerank-over-retrieval: "
      f"{np.mean(agg['p5_rr']) - np.mean(agg['p5_ret']):+.3f}")
print(f"  headroom oracle-over-rerank:  "
      f"{np.mean(agg['p5_oracle']) - np.mean(agg['p5_rr']):+.3f}")

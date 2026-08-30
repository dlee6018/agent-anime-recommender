"""LightGBM LambdaRank reranker over two-tower retrieval.

Training rows: (src, cand) for rec-graph srcs; cand = top-80 retrieved by a
two-tower whose training EXCLUDED all edges incident to that src's fold
(5 folds), so the tt-score feature carries no leakage. Labels: 3 = vote-rank
1-3, 2 = rank 4-10, 1 = in rec list beyond 10, 0 = retrieved noise.
Then evaluates retrieval-vs-rerank on the dev set.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_metadata  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.franchise import with_franchise_filter  # noqa: E402
from src.models import two_tower as tt  # noqa: E402
from src.models.rerank import (FEATS, FeatureBuilder,  # noqa: E402
                               make_rerank_recommender)

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--top_cand", type=int, default=80)
ap.add_argument("--maxrank", type=int, default=1500)
ap.add_argument("--union_extra", type=int, default=0,
                help="also union top-N cooc + top-N content candidates")
ap.add_argument("--out", default="reranker.txt")
ap.add_argument("--n_seeds", type=int, default=1,
                help="towers per fold; embeddings concatenated (mean cosine)")
args = ap.parse_args()

DATA = ROOT / "data"
N_FOLDS = 5
TOP_CAND = args.top_cand
MAXRANK_RETRIEVE = args.maxrank

meta = load_metadata()
ids, X, _ = build_features("content_emb_qwen.npz")
fb = FeatureBuilder(ids)
idx = fb.idx
pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999)
                for a in ids])
retrieve_mask = pop <= MAXRANK_RETRIEVE

pairs = pd.read_parquet(DATA / "rec_pairs.parquet")  # full graph for labels
train_pairs = pd.read_parquet(DATA / "train_pairs.parquet")

counts = train_pairs.groupby("src").size()
srcs = [s for s in counts[counts >= 8].index if int(s) in idx]
rng = np.random.default_rng(0)
folds = rng.integers(0, N_FOLDS, len(srcs))
print(f"reranker training srcs: {len(srcs)}", flush=True)

by_src_votes = {s: dict(zip(g.dst, g.votes)) for s, g in pairs.groupby("src")}
Xrows, ylab, groups = [], [], []

for f in range(N_FOLDS):
    fold_srcs = {int(srcs[i]) for i in range(len(srcs)) if folds[i] == f}
    tp = train_pairs[~train_pairs.src.isin(fold_srcs)
                     & ~train_pairs.dst.isin(fold_srcs)]
    print(f"fold {f}: tower on {len(tp)} pairs", flush=True)
    embs = []
    for s in range(args.n_seeds):
        _, e = tt.train_two_tower(ids, X, tp, epochs=12, d_out=512,
                                  seed=f * 100 + s, device="cuda")
        embs.append(e)
    emb = np.concatenate(embs, axis=1) / np.sqrt(len(embs))
    fb.set_graph(tp)  # 2-hop features must not see fold srcs' edges
    for s_id in sorted(fold_srcs):
        truth = by_src_votes.get(s_id, {})
        ranked = sorted(truth, key=truth.get, reverse=True)
        top3, top10 = set(ranked[:3]), set(ranked[:10])
        tq = emb[idx[s_id]]
        sim = emb @ tq
        sim[~retrieve_mask] = -np.inf
        sim[idx[s_id]] = -np.inf
        top = np.argpartition(-sim, TOP_CAND)[:TOP_CAND]
        top = top[np.argsort(-sim[top])]
        cands = [int(ids[i]) for i in top]
        if args.union_extra:
            seen = set(cands)
            for extra in (fb.cooc_top(s_id, args.union_extra, retrieve_mask),
                          fb.content_top(s_id, args.union_extra,
                                         retrieve_mask)):
                cands.extend(c for c in extra if c not in seen)
                seen.update(extra)
        Xrows.append(fb.rows(s_id, cands, tq, emb))
        ylab.append([3 if c in top3 else 2 if c in top10
                     else 1 if c in truth else 0 for c in cands])
        groups.append(len(cands))

Xall = np.concatenate(Xrows)
yall = np.concatenate(ylab)
print(f"reranker dataset: {Xall.shape}, positives: {(yall > 0).mean():.2%}",
      flush=True)

rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400,
                    learning_rate=0.05, num_leaves=63, min_child_samples=20,
                    label_gain=[0, 1, 3, 7], random_state=0, verbose=-1)
rk.fit(Xall, yall, group=groups)
rk.booster_.save_model(str(DATA / args.out))
imp = sorted(zip(FEATS, rk.feature_importances_), key=lambda x: -x[1])
print("feature importance:", imp, flush=True)

# dev evaluation: retrieval-only vs reranked (champion tower embeddings)
fb.set_graph(train_pairs)  # dev-honest graph
emb_file = "tt_ens_emb.npz" if args.n_seeds > 1 else "two_tower_emb.npz"
d = np.load(DATA / emb_file)
tt_emb = d["emb"].astype(np.float32)
dev_raw = json.load(open(DATA / "dev_set.json"))["queries"]
dev_set = {int(q): [int(r) for r in v] for q, v in dev_raw.items()}
fn = with_franchise_filter(make_rerank_recommender(
    ids, tt_emb, rk.booster_, fb, MAXRANK_RETRIEVE, TOP_CAND,
    union_extra=args.union_extra))
res = evaluate(fn, dev_set)
print(f"RERANK dev P@5={res['precision_at_k']:.3f} mrr={res['mrr']:.3f}",
      flush=True)

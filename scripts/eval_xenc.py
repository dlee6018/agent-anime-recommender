"""Fuse cross-encoder scores with the pipeline's ranking; tune weight on dev.

Pipeline = champion src-only stack (dev-honest graph); xenc trained on the
same dev-honest pairs. Fusion: score(c) = (1-w)/(rank_pipeline+3)
+ w*sigmoid(xenc(q,c)); sweep w.
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
from src.data import load_metadata, year_of  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.franchise import with_franchise_filter  # noqa: E402
from src.models.rerank import FeatureBuilder, make_rerank_recommender  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/xenc")
ap.add_argument("--n_cand", type=int, default=30)
ap.add_argument("--eval_set", default="data/dev_set.json")
ap.add_argument("--graph", default="data/train_pairs_srconly_dev.parquet")
ap.add_argument("--booster", default="data/reranker_final.txt")
args = ap.parse_args()

meta = load_metadata()


def doc(aid: int) -> str:
    m = meta[aid]
    bits = [m["name"]]
    if m["english"] and m["english"] != m["name"]:
        bits.append(m["english"])
    bits.append(f"({year_of(aid) or '?'})")
    tags = ", ".join((m["genres"] + m["themes"] + m["demographics"])[:8])
    if tags:
        bits.append(tags)
    syn = (m["synopsis"] or "").replace("\n", " ")
    if "no description available" in syn.lower()[:60]:
        syn = ""
    bits.append(syn[:300])
    return ". ".join(bits)


ids, X, _ = build_features("content_emb_qwen.npz")
d = np.load(ROOT / "data" / "tt_ens_emb.npz")
emb = d["emb"].astype(np.float32)
booster = lgb.Booster(model_file=str(ROOT / args.booster))
fb = FeatureBuilder(ids)
fb.set_graph(pd.read_parquet(ROOT / args.graph))
base = with_franchise_filter(make_rerank_recommender(
    ids, emb, booster, fb, 8000, 250, union_extra=100))

ev = {int(q): [int(r) for r in v]
      for q, v in json.load(open(ROOT / args.eval_set))["queries"].items()}

from sentence_transformers import CrossEncoder  # noqa: E402

xenc = CrossEncoder(str(ROOT / args.model), max_length=384, device="cuda")

# score all (query, candidate) pairs once
cands_by_q, scores_by_q = {}, {}
for q in ev:
    cands = base([q], args.n_cand)
    cands_by_q[q] = cands
    sc = xenc.predict([(doc(q), doc(c)) for c in cands],
                      show_progress_bar=False)
    scores_by_q[q] = 1 / (1 + np.exp(-np.asarray(sc)))

for w in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    def rec(qids, k, w=w):
        q = qids[0]
        cands, xs = cands_by_q[q], scores_by_q[q]
        fused = [(1 - w) / (r + 3) + w * float(xs[r]) / 10
                 for r in range(len(cands))]
        order = np.argsort(fused)[::-1]
        return [cands[i] for i in order[:k]]
    p5 = evaluate(rec, ev)["precision_at_k"]
    print(f"w={w}: P@5 = {p5:.3f}", flush=True)

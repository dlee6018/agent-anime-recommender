"""Milestone read with cross-encoder fusion (exp 47).

Eval-condition components: towers + graph on train_pairs_eval, xenc trained
on train_pairs_eval (models/xenc_eval), fusion weight chosen on dev (0.82).
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

import wandb  # noqa: E402

from src.data import load_metadata, titles, year_of  # noqa: E402
from src.evaluate import evaluate, load_eval_set, print_report  # noqa: E402
from src.features import build_features  # noqa: E402
from src.franchise import with_franchise_filter  # noqa: E402
from src.models import two_tower as tt  # noqa: E402
from src.models.rerank import FeatureBuilder, make_rerank_recommender  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n_seeds", type=int, default=9)
ap.add_argument("--w", type=float, default=0.82)
ap.add_argument("--n_cand", type=int, default=30)
ap.add_argument("--xenc", default="models/xenc_eval")
ap.add_argument("--reranker", default="reranker_final.txt")
ap.add_argument("--name", default="milestone-xenc")
args = ap.parse_args()

DATA = ROOT / "data"
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
full = pd.read_parquet(DATA / "rec_pairs.parquet")
eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
pairs = full[~full.src.isin(eval_ids)]  # src-only eval condition
print(f"eval-condition pairs: {len(pairs)}", flush=True)

embs = []
for s in range(args.n_seeds):
    _, e = tt.train_two_tower(ids, X, pairs, epochs=12, d_out=512,
                              seed=1000 + s, device="cuda")
    embs.append(e)
emb = np.concatenate(embs, axis=1) / np.sqrt(len(embs))

fb = FeatureBuilder(ids)
fb.set_graph(pairs)
al_path = DATA / "anilist_recs.json"
if al_path.exists():
    fb.set_anilist(json.load(open(al_path)))
booster = lgb.Booster(model_file=str(DATA / args.reranker))
base = with_franchise_filter(make_rerank_recommender(
    ids, emb, booster, fb, 8000, 250, union_extra=100))

from sentence_transformers import CrossEncoder  # noqa: E402

xenc = CrossEncoder(str(ROOT / args.xenc), max_length=384, device="cuda")


def recommend(qids, k):
    q = qids[0]
    cands = base(qids, args.n_cand)
    sc = xenc.predict([(doc(q), doc(c)) for c in cands],
                      show_progress_bar=False)
    xs = 1 / (1 + np.exp(-np.asarray(sc)))
    fused = [(1 - args.w) / (r + 3) + args.w * float(xs[r]) / 10
             for r in range(len(cands))]
    order = np.argsort(fused)[::-1]
    return [cands[i] for i in order[:k]]


res = evaluate(recommend, load_eval_set())
print(f"\nMILESTONE-XENC eval P@5={res['precision_at_k']:.3f} "
      f"P@1={res['precision_at_1']:.3f} MRR={res['mrr']:.3f}")
print_report(res, titles(), n_worst=8)
json.dump({str(q): p for q, p in res["per_query"].items()},
          open(DATA / "milestone_last.json", "w"))

run = wandb.init(project="anime-rec", name=args.name,
                 config={**vars(args), "milestone": True})
run.log({"eval/p5": res["precision_at_k"], "eval/p1": res["precision_at_1"],
         "eval/mrr": res["mrr"]})
run.finish()

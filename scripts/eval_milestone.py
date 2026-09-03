"""Milestone read of the FROZEN eval set (use sparingly — dev is for tuning).

Eval condition: models may use everything except edges incident to eval
anime. So towers + graph features are rebuilt on train_pairs_eval.parquet
(includes dev-incident edges); the reranker booster itself is reused from
dev-honest training (conservative).
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

from src.data import titles  # noqa: E402
from src.evaluate import evaluate, load_eval_set, print_report  # noqa: E402
from src.features import build_features  # noqa: E402
from src.franchise import with_franchise_filter  # noqa: E402
from src.models import two_tower as tt  # noqa: E402
from src.models.rerank import FeatureBuilder, make_rerank_recommender  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n_seeds", type=int, default=3)
ap.add_argument("--name", default="milestone")
ap.add_argument("--holdout", choices=["symmetric", "src_only"],
                default="symmetric",
                help="symmetric: no edge may touch an eval anime (strict). "
                     "src_only: only eval anime's OWN lists hidden; edges "
                     "pointing TO them from other lists stay visible "
                     "(the literal 'never saw its rec page' reading).")
ap.add_argument("--reranker", default=None,
                help="override booster file from best_pipeline.json")
args = ap.parse_args()

DATA = ROOT / "data"
cfg = json.load(open(DATA / "best_pipeline.json"))
if args.reranker:
    cfg["reranker"] = args.reranker
ids, X, _ = build_features("content_emb_qwen.npz")
full = pd.read_parquet(DATA / "rec_pairs.parquet")
eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
if args.holdout == "symmetric":
    pairs = full[~full.src.isin(eval_ids) & ~full.dst.isin(eval_ids)]
else:
    pairs = full[~full.src.isin(eval_ids)]
print(f"eval-condition training pairs ({args.holdout}): {len(pairs)}",
      flush=True)

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
booster = lgb.Booster(model_file=str(DATA / cfg["reranker"]))
fn = with_franchise_filter(make_rerank_recommender(
    ids, emb, booster, fb, cfg["maxrank"], cfg["top_cand"],
    union_extra=cfg["union_extra"]))

res = evaluate(fn, load_eval_set())
print(f"\nMILESTONE eval P@5={res['precision_at_k']:.3f} "
      f"P@1={res['precision_at_1']:.3f} MRR={res['mrr']:.3f}")
print_report(res, titles(), n_worst=10)
json.dump({str(q): p for q, p in res["per_query"].items()},
          open(DATA / "milestone_last.json", "w"))

run = wandb.init(project="anime-rec", name=args.name,
                 config={**cfg, "n_seeds": args.n_seeds, "milestone": True})
run.log({"eval/p5": res["precision_at_k"], "eval/p1": res["precision_at_1"],
         "eval/mrr": res["mrr"]})
run.finish()

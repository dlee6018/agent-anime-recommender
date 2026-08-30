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
args = ap.parse_args()

DATA = ROOT / "data"
cfg = json.load(open(DATA / "best_pipeline.json"))
ids, X, _ = build_features("content_emb_qwen.npz")
pairs = pd.read_parquet(DATA / "train_pairs_eval.parquet")
print(f"eval-condition training pairs: {len(pairs)}", flush=True)

embs = []
for s in range(args.n_seeds):
    _, e = tt.train_two_tower(ids, X, pairs, epochs=12, d_out=512,
                              seed=1000 + s, device="cuda")
    embs.append(e)
emb = np.concatenate(embs, axis=1) / np.sqrt(len(embs))

fb = FeatureBuilder(ids)
fb.set_graph(pairs)
booster = lgb.Booster(model_file=str(DATA / cfg["reranker"]))
fn = with_franchise_filter(make_rerank_recommender(
    ids, emb, booster, fb, cfg["maxrank"], cfg["top_cand"],
    union_extra=cfg["union_extra"]))

res = evaluate(fn, load_eval_set())
print(f"\nMILESTONE eval P@5={res['precision_at_k']:.3f} "
      f"P@1={res['precision_at_1']:.3f} MRR={res['mrr']:.3f}")
print_report(res, titles(), n_worst=10)

run = wandb.init(project="anime-rec", name=args.name,
                 config={**cfg, "n_seeds": args.n_seeds, "milestone": True})
run.log({"eval/p5": res["precision_at_k"], "eval/p1": res["precision_at_1"],
         "eval/mrr": res["mrr"]})
run.finish()

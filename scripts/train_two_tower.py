"""Train the two-tower model on rec pairs; save item embeddings + eval.

Usage: train_two_tower.py [--epochs 40] [--sample_by_votes] [--n_hard_neg 256]
                          [--asym] [--maxrank 700] [--pop_weight 0.15]
                          [--name run-name] [--no_eval]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wandb  # noqa: E402

from src.data import load_metadata, titles  # noqa: E402
from src.evaluate import evaluate, load_eval_set, print_report  # noqa: E402
from src.features import build_features  # noqa: E402
from src.franchise import with_franchise_filter  # noqa: E402
from src.models.embed_knn import make_recommender  # noqa: E402
from src.models import two_tower as tt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--temp", type=float, default=0.05)
ap.add_argument("--d_hidden", type=int, default=512)
ap.add_argument("--d_out", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sample_by_votes", action="store_true")
ap.add_argument("--n_hard_neg", type=int, default=0)
ap.add_argument("--asym", action="store_true")
ap.add_argument("--maxrank", type=int, default=700,
                help="candidate mask + hard-negative pool: pop rank cutoff")
ap.add_argument("--pop_weight", type=float, default=0.15)
ap.add_argument("--name", default="two-tower")
ap.add_argument("--no_eval", action="store_true",
                help="dev only; don't touch the frozen eval set")
args = ap.parse_args()

ids, X, info = build_features()
pairs = pd.read_parquet(ROOT / "data" / "train_pairs.parquet")
print(f"features {X.shape}, train pairs {len(pairs)}", flush=True)

meta = load_metadata()
pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999) for a in ids])
cand_mask = pop <= args.maxrank
hard_pool = np.where(cand_mask)[0]

dev_raw = json.load(open(ROOT / "data" / "dev_set.json"))["queries"]
dev_set = {int(q): [int(r) for r in v] for q, v in dev_raw.items()}

run = wandb.init(project="anime-rec", name=args.name, config=vars(args))


def make_fn(cand_emb, query_emb=None):
    return with_franchise_filter(make_recommender(
        ids, cand_emb, pop_weight=args.pop_weight,
        candidate_mask=cand_mask, query_emb=query_emb))


def dev_eval(cand_emb):
    return evaluate(make_fn(cand_emb), dev_set)["precision_at_k"]


def log(ep, loss, dev_p5):
    run.log({"epoch": ep, "loss": loss, "dev/p5": dev_p5})
    print(f"ep {ep}: loss={loss:.4f} dev_p5={dev_p5}", flush=True)


tower, cand_emb = tt.train_two_tower(
    ids, X, pairs, epochs=args.epochs, batch=args.batch, lr=args.lr,
    temp=args.temp, d_hidden=args.d_hidden, d_out=args.d_out,
    dropout=args.dropout, seed=args.seed,
    sample_by_votes=args.sample_by_votes, n_hard_neg=args.n_hard_neg,
    hard_neg_pool=hard_pool, asym=args.asym,
    dev_eval_fn=dev_eval, log_fn=log)

import torch  # noqa: E402

Xt = torch.tensor(X, device="cuda")
query_emb = tt.encode_items(tower, Xt, side="q") if args.asym else None
np.savez(ROOT / "data" / "two_tower_emb.npz", ids=ids, emb=cand_emb,
         **({"query_emb": query_emb} if query_emb is not None else {}))
torch.save(tower.state_dict(), ROOT / "data" / "two_tower.pt")

fn = make_fn(cand_emb, query_emb)
dev = evaluate(fn, dev_set)
print(f"\nFINAL dev P@5={dev['precision_at_k']:.3f} mrr={dev['mrr']:.3f}")
run.log({"dev/p5_final": dev["precision_at_k"]})
if not args.no_eval:
    ev = evaluate(fn, load_eval_set())
    print(f"FINAL eval P@5={ev['precision_at_k']:.3f} "
          f"p1={ev['precision_at_1']:.3f} mrr={ev['mrr']:.3f}")
    run.log({"eval/p5": ev["precision_at_k"], "eval/p1": ev["precision_at_1"],
             "eval/mrr": ev["mrr"]})
    print_report(ev, titles(), n_worst=8)
run.finish()

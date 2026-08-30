"""Frozen evaluation harness.

Ground truth: data/eval_set.json — {query_mal_id: [top-10 rec mal_ids]} scraped
fresh from MAL userrecs pages (most-recommended-first order).

Goal metric: mean precision@5 >= 0.80.

Model contract: recommend_fn(query_ids: list[int], k: int) -> list[int]
(never includes any query id in the output).
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_eval_set() -> dict[int, list[int]]:
    raw = json.load(open(DATA / "eval_set.json"))
    return {int(q): [int(r) for r in recs] for q, recs in raw["queries"].items()}


def evaluate(recommend_fn, eval_set: dict[int, list[int]] | None = None,
             k: int = 5) -> dict:
    if eval_set is None:
        eval_set = load_eval_set()
    if not eval_set:
        raise ValueError("empty eval set")
    per_query = {}
    for q, truth in eval_set.items():
        truth10 = set(truth[:10])
        recs = recommend_fn([q], k)[:k]
        assert q not in recs, f"query {q} leaked into its own recs"
        hits = [r for r in recs if r in truth10]
        rr = 0.0
        for rank, r in enumerate(recs, 1):
            if r in truth10:
                rr = 1.0 / rank
                break
        per_query[q] = {
            "precision": len(hits) / k,
            "hit1": 1.0 if recs and recs[0] in truth10 else 0.0,
            "rr": rr,
            "recs": recs,
            "hits": hits,
        }
    n = len(per_query)
    return {
        "k": k,
        "n_queries": n,
        "precision_at_k": sum(p["precision"] for p in per_query.values()) / n,
        "precision_at_1": sum(p["hit1"] for p in per_query.values()) / n,
        "mrr": sum(p["rr"] for p in per_query.values()) / n,
        "per_query": per_query,
    }


def print_report(res: dict, titles: dict[int, str] | None = None,
                 n_worst: int = 10) -> None:
    print(f"n={res['n_queries']}  P@{res['k']}={res['precision_at_k']:.3f}  "
          f"P@1={res['precision_at_1']:.3f}  MRR={res['mrr']:.3f}")
    if titles:
        worst = sorted(res["per_query"].items(), key=lambda x: x[1]["precision"])
        print(f"-- {n_worst} worst queries --")
        for q, p in worst[:n_worst]:
            print(f"  P={p['precision']:.1f} {titles.get(q, q)}: "
                  f"{[titles.get(r, r) for r in p['recs']]}")

"""Production recommender: graph-priority with ML fallback.

For queries whose MAL rec list is in the graph, the crowd's own vote-ranked
list IS the best possible answer (frozen-oracle P@5 = 1.0 vs fresh truth).
The ML pipeline generalizes to anime without (enough of) a list — new,
obscure, or unseen — and fills the remainder.
"""
import numpy as np
import pandas as pd

from ..franchise import franchise_filter, same_franchise


def make_product_recommender(pairs: pd.DataFrame, fallback_fn):
    by_src: dict[int, list[tuple[int, float]]] = {}
    for s, g in pairs.groupby("src"):
        g = g.sort_values("votes", ascending=False)
        by_src[int(s)] = list(zip(g.dst.astype(int), g.votes.astype(float)))

    def recommend(query_ids: list[int], k: int) -> list[int]:
        keep = franchise_filter(query_ids)
        score: dict[int, float] = {}
        for q in query_ids:
            lst = by_src.get(q, [])
            tot = sum(v for _, v in lst) or 1.0
            for d, v in lst:
                score[d] = score.get(d, 0.0) + v / tot
        ranked = sorted(score, key=score.get, reverse=True)
        out: list[int] = []
        for c in ranked:
            if c in query_ids or not keep(c):
                continue
            if any(same_franchise(c, o) for o in out):
                continue
            out.append(c)
            if len(out) == k:
                return out
        # graph exhausted -> fill from the ML pipeline
        for c in fallback_fn(query_ids, k + len(out) + 5):
            if c in out or c in query_ids or not keep(c):
                continue
            if any(same_franchise(c, o) for o in out):
                continue
            out.append(c)
            if len(out) == k:
                break
        return out

    return recommend

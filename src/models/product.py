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


def make_heldout_recommender(pairs, ids, emb, booster, fb_factory,
                             maxrank=8000, top_cand=250, union_extra=100,
                             mode="strict"):
    """Serving-time HONEST recommender: for each query, rebuild the graph with
    the query's incident edges removed, so reranker graph-features (rev_edge,
    colist, transfer_in) cannot read the crowd answer back. This reproduces
    the held-out eval condition per query.

    mode: 'strict' drops every edge touching the query (src OR dst) — true
    generalization (eval 0.508); 'src_only' drops only the query's own page
    (eval 0.778).
    """
    import pandas as pd
    from .rerank import make_rerank_recommender
    from ..franchise import with_franchise_filter

    src_a = pairs["src"].to_numpy()
    dst_a = pairs["dst"].to_numpy()

    def recommend(query_ids, k):
        q = set(query_ids)
        if mode == "strict":
            keep = ~(pd.Series(src_a).isin(q) | pd.Series(dst_a).isin(q))
        else:
            keep = ~pd.Series(src_a).isin(q)
        fb = fb_factory()
        fb.set_graph(pairs[keep.to_numpy()])
        fn = with_franchise_filter(make_rerank_recommender(
            ids, emb, booster, fb, maxrank, top_cand, union_extra=union_extra))
        return fn(query_ids, k)

    return recommend

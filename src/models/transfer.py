"""Neighbor-transfer: recommend what the query's nearest visible-graph
neighbors' rec lists point to (vote-weighted). Training-free.

score(c) = sum_n  sim(q, n)^beta * (votes(n->c)^gamma / norm_n)
over the query's top-N neighbors n that are srcs in the visible rec graph.
"""
import numpy as np
import pandas as pd


def transfer_recommender(ids: np.ndarray, sim_emb: np.ndarray,
                         pairs: pd.DataFrame, pop_prior: np.ndarray,
                         n_neighbors: int = 30, beta: float = 3.0,
                         gamma: float = 0.5, pop_weight: float = 0.0,
                         candidate_mask: np.ndarray | None = None):
    """sim_emb: L2-normed embedding used to find neighbors (e.g. two-tower).
    pairs: visible rec graph (already holdout-filtered)."""
    idx = {int(a): i for i, a in enumerate(ids)}
    # per-src normalized vote lists
    lists: dict[int, list[tuple[int, float]]] = {}
    for s, g in pairs.groupby("src"):
        v = g.votes.to_numpy().astype(np.float32) ** gamma
        v /= v.sum() + 1e-9
        lists[int(s)] = [(idx[d], float(w))
                         for d, w in zip(g.dst, v) if int(d) in idx]
    src_rows = np.array([idx[s] for s in lists if s in idx])
    src_ids = np.array([s for s in lists if s in idx])
    if candidate_mask is None:
        candidate_mask = np.ones(len(ids), dtype=bool)

    def recommend(query_ids: list[int], k: int) -> list[int]:
        qrows = [idx[q] for q in query_ids if q in idx]
        if not qrows:
            return []
        qv = sim_emb[qrows].mean(axis=0)
        qv /= np.linalg.norm(qv) + 1e-9
        nsim = sim_emb[src_rows] @ qv
        top = np.argsort(-nsim)[:n_neighbors]
        score = np.zeros(len(ids), dtype=np.float32)
        for t in top:
            w = max(float(nsim[t]), 0.0) ** beta
            for ci, cw in lists[int(src_ids[t])]:
                score[ci] += w * cw
        score += pop_weight * pop_prior
        score[~candidate_mask] = -np.inf
        qset = set(query_ids)
        for r in qrows:
            score[r] = -np.inf
        order = np.argsort(-score)
        return [int(ids[i]) for i in order if int(ids[i]) not in qset][:k]

    return recommend

"""Generic recommender from any {ids, emb} item-embedding table:
cosine similarity + optional popularity prior, query-mean pooling."""
from pathlib import Path

import numpy as np

from ..data import load_metadata

DATA = Path(__file__).resolve().parent.parent.parent / "data"


def make_recommender(ids: np.ndarray, emb: np.ndarray,
                     pop_weight: float = 0.0,
                     candidate_mask: np.ndarray | None = None):
    """ids: int64[N], emb: float32[N,D] (L2-normalized rows)."""
    idx = {int(a): i for i, a in enumerate(ids)}
    meta = load_metadata()
    pop_prior = np.array(
        [1.0 / np.log2(3 + (meta.get(int(a), {}).get("popularity") or 20000))
         for a in ids], dtype=np.float32)
    if candidate_mask is None:
        candidate_mask = np.ones(len(ids), dtype=bool)

    def recommend(query_ids: list[int], k: int) -> list[int]:
        qrows = [idx[q] for q in query_ids if q in idx]
        if not qrows:
            order = np.argsort(-pop_prior * candidate_mask)
            return [int(ids[i]) for i in order[:k]]
        qv = emb[qrows].mean(axis=0)
        qv /= np.linalg.norm(qv) + 1e-9
        sim = emb @ qv + pop_weight * pop_prior
        sim[~candidate_mask] = -np.inf
        for r in qrows:
            sim[r] = -np.inf
        order = np.argsort(-sim)
        qset = set(query_ids)
        return [int(ids[i]) for i in order if int(ids[i]) not in qset][:k]

    return recommend


def blend_recommender(tables: list[tuple[np.ndarray, np.ndarray, float]],
                      pop_weight: float = 0.0):
    """Late-fusion: weighted sum of cosine sims over a shared id universe.
    tables: [(ids, emb, weight), ...]. Universe = union of ids."""
    universe = sorted({int(a) for ids, _, _ in tables for a in ids})
    uidx = {a: i for i, a in enumerate(universe)}
    meta = load_metadata()
    pop_prior = np.array(
        [1.0 / np.log2(3 + (meta.get(a, {}).get("popularity") or 20000))
         for a in universe], dtype=np.float32)
    mats = []
    for ids, emb, w in tables:
        rows = np.array([uidx[int(a)] for a in ids])
        mats.append((rows, emb, w, {int(a): i for i, a in enumerate(ids)}))

    def recommend(query_ids: list[int], k: int) -> list[int]:
        total = np.full(len(universe), 0.0, dtype=np.float32)
        seen_any = False
        for rows, emb, w, idx in mats:
            qrows = [idx[q] for q in query_ids if q in idx]
            if not qrows:
                continue
            seen_any = True
            qv = emb[qrows].mean(axis=0)
            qv /= np.linalg.norm(qv) + 1e-9
            total[rows] += w * (emb @ qv)
        total += pop_weight * pop_prior
        if not seen_any:
            total = pop_prior.copy()
        qset = set(query_ids)
        for q in qset:
            if q in uidx:
                total[uidx[q]] = -np.inf
        order = np.argsort(-total)
        return [universe[i] for i in order if universe[i] not in qset][:k]

    return recommend

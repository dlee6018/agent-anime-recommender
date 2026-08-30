"""Cold-start-capable baselines (eval queries have no rec-graph edges)."""
import numpy as np

from ..data import load_metadata, year_of


def popularity_recommender(candidate_ids: list[int]):
    """Always returns the globally most popular candidates."""
    meta = load_metadata()
    ranked = sorted(candidate_ids, key=lambda a: meta[a]["popularity"] or 10**9)

    def recommend(query_ids: list[int], k: int) -> list[int]:
        return [a for a in ranked if a not in set(query_ids)][:k]

    return recommend


def genre_content_recommender(candidate_ids: list[int]):
    """Cosine over genre multi-hot + studio + year proximity + popularity tiebreak."""
    meta = load_metadata()
    all_genres = sorted({g for a in candidate_ids for g in meta[a]["genres"]})
    gidx = {g: i for i, g in enumerate(all_genres)}
    M = np.zeros((len(candidate_ids), len(gidx)), dtype=np.float32)
    idx = {a: i for i, a in enumerate(candidate_ids)}
    for a in candidate_ids:
        for g in meta[a]["genres"]:
            M[idx[a], gidx[g]] = 1.0
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    pop = np.array([1.0 / np.log2(3 + (meta[a]["popularity"] or 20000))
                    for a in candidate_ids], dtype=np.float32)
    years = np.array([year_of(a) or 2005 for a in candidate_ids], dtype=np.float32)

    def recommend(query_ids: list[int], k: int) -> list[int]:
        qv = np.mean([M[idx[q]] for q in query_ids if q in idx], axis=0)
        qy = np.mean([years[idx[q]] for q in query_ids if q in idx])
        sim = M @ qv
        sim += 0.5 * pop                       # popularity prior
        sim -= 0.01 * np.abs(years - qy) / 10  # era proximity
        order = np.argsort(-sim)
        qset = set(query_ids)
        out = [candidate_ids[i] for i in order if candidate_ids[i] not in qset]
        return out[:k]

    return recommend

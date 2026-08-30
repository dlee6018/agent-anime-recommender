"""Collaborative filtering over the user co-watch matrix (implicit ALS).

Eval anime are popular, so they have plenty of interactions — no cold-start
problem here (the holdout only removes rec-graph edges, not behavior data).
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent.parent / "data"


def train_als(factors: int = 128, iterations: int = 20, alpha: float = 20.0,
              regularization: float = 0.05, seed: int = 42):
    """Returns (item_ids, item_factors L2-normalized)."""
    from implicit.als import AlternatingLeastSquares

    mat = sp.load_npz(DATA / "interactions_csr.npz")
    item_ids = np.load(DATA / "interactions_meta.npz")["item_ids"]
    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations, regularization=regularization,
        alpha=alpha, random_state=seed, use_gpu=False)
    model.fit(mat)  # implicit >= 0.5 expects user x item
    emb = model.item_factors.astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return item_ids, emb


def save_als(item_ids, emb, name: str = "als_emb.npz") -> None:
    np.savez(DATA / name, ids=item_ids, emb=emb)


def load_als(name: str = "als_emb.npz"):
    d = np.load(DATA / name)
    return d["ids"], d["emb"]

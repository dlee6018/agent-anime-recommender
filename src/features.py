"""Assemble the per-anime feature matrix consumed by the two-tower model.

Universe = content_emb ids. Blocks:
  content text emb (1024, L2-normed) | ALS co-watch emb (128, L2-normed,
  zero + missing-flag if absent) | genre multi-hot | log-members proxy,
  year (normalized), type one-hot.
"""
from pathlib import Path

import numpy as np

from .data import load_metadata, year_of

DATA = Path(__file__).resolve().parent.parent / "data"

TYPES = ["TV", "Movie", "OVA", "ONA", "Special", "Music"]


def build_features() -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (ids, X, info) with X float32 [N, D]."""
    c = np.load(DATA / "content_emb.npz")
    ids, content = c["ids"], c["emb"]
    idx = {int(a): i for i, a in enumerate(ids)}
    N = len(ids)

    als_block = np.zeros((N, 0), dtype=np.float32)
    als_flag = np.zeros((N, 1), dtype=np.float32)
    als_path = DATA / "als_emb.npz"
    if als_path.exists():
        a = np.load(als_path)
        als_block = np.zeros((N, a["emb"].shape[1]), dtype=np.float32)
        for aid, row in zip(a["ids"], a["emb"]):
            i = idx.get(int(aid))
            if i is not None:
                als_block[i] = row
                als_flag[i] = 1.0

    i2v_block = np.zeros((N, 0), dtype=np.float32)
    i2v_path = DATA / "i2v_emb.npz"
    if i2v_path.exists():
        v = np.load(i2v_path)
        i2v_block = np.zeros((N, v["emb"].shape[1]), dtype=np.float32)
        for aid, row in zip(v["ids"], v["emb"]):
            i = idx.get(int(aid))
            if i is not None:
                i2v_block[i] = row

    meta = load_metadata()
    genres = sorted({g for a in ids for g in meta[int(a)]["genres"]})
    gidx = {g: j for j, g in enumerate(genres)}
    themes = sorted({t for a in ids for t in meta[int(a)]["themes"]})
    tidx = {t: j for j, t in enumerate(themes)}
    demos = sorted({d for a in ids for d in meta[int(a)]["demographics"]})
    didx = {d: j for j, d in enumerate(demos)}
    G = np.zeros((N, len(genres)), dtype=np.float32)
    T = np.zeros((N, len(themes)), dtype=np.float32)
    D = np.zeros((N, len(demos)), dtype=np.float32)
    scal = np.zeros((N, 3 + len(TYPES)), dtype=np.float32)
    for i, aid in enumerate(ids):
        m = meta[int(aid)]
        for g in m["genres"]:
            G[i, gidx[g]] = 1.0
        for t in m["themes"]:
            T[i, tidx[t]] = 1.0
        for d in m["demographics"]:
            D[i, didx[d]] = 1.0
        pop = m["popularity"] or 15000
        scal[i, 0] = -np.log10(pop) / 4.0          # popularity (higher=more pop)
        scal[i, 1] = ((year_of(int(aid)) or 2005) - 2005) / 20.0
        scal[i, 2] = (m["score"] or 6.5) / 10.0
        if m["type"] in TYPES:
            scal[i, 3 + TYPES.index(m["type"])] = 1.0

    X = np.concatenate([content, als_block, als_flag, i2v_block, G, T, D,
                        scal], axis=1)
    info = {"content_dim": content.shape[1], "als_dim": als_block.shape[1],
            "i2v_dim": i2v_block.shape[1], "n_genres": len(genres),
            "n_themes": len(themes), "n_demos": len(demos)}
    return ids, X.astype(np.float32), info

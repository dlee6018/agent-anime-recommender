"""Named model constructors. Each returns recommend(query_ids, k) -> list[int].

All models are franchise-filtered (MAL userrecs are cross-franchise).
`best` is updated to point at the current champion as experiments progress.
"""
from pathlib import Path

import numpy as np

from .franchise import with_franchise_filter
from .models import baselines, embed_knn

DATA = Path(__file__).resolve().parent.parent / "data"


def _content_table():
    d = np.load(DATA / "content_emb.npz")
    return d["ids"], d["emb"]


def _als_table():
    d = np.load(DATA / "als_emb.npz")
    return d["ids"], d["emb"]


def get_model(name: str):
    if name == "pop":
        ids, _ = _content_table()
        fn = baselines.popularity_recommender([int(a) for a in ids])
    elif name == "genre":
        ids, _ = _content_table()
        fn = baselines.genre_content_recommender([int(a) for a in ids])
    elif name == "content":
        fn = embed_knn.make_recommender(*_content_table(), pop_weight=0.15)
    elif name == "als":
        fn = embed_knn.make_recommender(*_als_table(), pop_weight=0.0)
    elif name == "blend":
        cids, cemb = _content_table()
        aids, aemb = _als_table()
        fn = embed_knn.blend_recommender(
            [(cids, cemb, 0.4), (aids, aemb, 1.0)], pop_weight=0.1)
    elif name == "two_tower" or name == "best":
        d = np.load(DATA / "two_tower_emb.npz")
        fn = embed_knn.make_recommender(d["ids"], d["emb"], pop_weight=0.0)
    else:
        raise ValueError(f"unknown model {name!r}")
    return with_franchise_filter(fn)

"""Named model constructors. Each returns recommend(query_ids, k) -> list[int].

All models are franchise-filtered (MAL userrecs are cross-franchise).
`best` = two-tower retrieval + LightGBM reranker (current champion pipeline).
"""
import json
from pathlib import Path

import numpy as np

from .franchise import with_franchise_filter
from .models import baselines, embed_knn

DATA = Path(__file__).resolve().parent.parent / "data"


def _content_table():
    d = np.load(DATA / "content_emb.npz")
    return d["ids"], d["emb"].astype(np.float32)


def _qwen_table():
    d = np.load(DATA / "content_emb_qwen.npz")
    return d["ids"], d["emb"].astype(np.float32)


def _als_table():
    d = np.load(DATA / "als_emb.npz")
    return d["ids"], d["emb"].astype(np.float32)


def _tt_table():
    d = np.load(DATA / "two_tower_emb.npz")
    return d["ids"], d["emb"].astype(np.float32)


def get_model(name: str):
    if name == "pop":
        ids, _ = _content_table()
        fn = baselines.popularity_recommender([int(a) for a in ids])
    elif name == "genre":
        ids, _ = _content_table()
        fn = baselines.genre_content_recommender([int(a) for a in ids])
    elif name == "content":
        fn = embed_knn.make_recommender(*_content_table(), pop_weight=0.15)
    elif name == "content_qwen":
        fn = embed_knn.make_recommender(*_qwen_table(), pop_weight=0.15)
    elif name == "als":
        fn = embed_knn.make_recommender(*_als_table(), pop_weight=0.0)
    elif name == "two_tower":
        from .data import load_metadata
        ids, emb = _tt_table()
        meta = load_metadata()
        pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999)
                        for a in ids])
        fn = embed_knn.make_recommender(ids, emb, pop_weight=0.15,
                                        candidate_mask=pop <= 700)
    elif name in ("rerank", "best"):
        import lightgbm as lgb

        from .models.rerank import FeatureBuilder, make_rerank_recommender
        cfg_file = DATA / "best_pipeline.json"
        cfg = (json.load(open(cfg_file)) if cfg_file.exists()
               else {"reranker": "reranker.txt", "maxrank": 1500,
                     "top_cand": 80, "union_extra": 0})
        ids, emb = _tt_table()
        booster = lgb.Booster(model_file=str(DATA / cfg["reranker"]))
        fb = FeatureBuilder(ids)
        fn = make_rerank_recommender(
            ids, emb, booster, fb, cfg["maxrank"], cfg["top_cand"],
            union_extra=cfg["union_extra"])
    else:
        raise ValueError(f"unknown model {name!r}")
    return with_franchise_filter(fn)

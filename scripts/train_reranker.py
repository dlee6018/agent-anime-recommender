"""LightGBM LambdaRank reranker over two-tower retrieval.

Training rows: (src, cand) for ~2.4k rec-graph srcs; cand = top-80 retrieved
by a two-tower whose training EXCLUDED all edges incident to that src's fold
(5 folds), so the tt-score feature carries no leakage. Labels: 3 = vote-rank
1-3, 2 = rank 4-10, 1 = in rec list beyond 10, 0 = retrieved noise.
Saves model + the retrieval artifacts needed at inference.
"""
import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_metadata, year_of  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import two_tower as tt  # noqa: E402

DATA = ROOT / "data"
N_FOLDS = 5
TOP_CAND = 80
MAXRANK_RETRIEVE = 1500

meta = load_metadata()
ids, X, _ = build_features()
idx = {int(a): i for i, a in enumerate(ids)}
pop = np.array([(meta.get(int(a), {}).get("popularity") or 99999) for a in ids])
retrieve_mask = pop <= MAXRANK_RETRIEVE

pairs = pd.read_parquet(DATA / "rec_pairs.parquet")  # full graph for labels
train_pairs = pd.read_parquet(DATA / "train_pairs.parquet")
dev_ids = {int(q) for q in json.load(open(DATA / "dev_set.json"))["queries"]}
eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}

counts = train_pairs.groupby("src").size()
srcs = [s for s in counts[counts >= 8].index if int(s) in idx]
rng = np.random.default_rng(0)
folds = rng.integers(0, N_FOLDS, len(srcs))
print(f"reranker training srcs: {len(srcs)}", flush=True)

# ---- auxiliary similarity tables ----
cont = np.load(DATA / "content_emb.npz")
CEMB = np.zeros((len(ids), cont["emb"].shape[1]), dtype=np.float32)
for a, e in zip(cont["ids"], cont["emb"]):
    if int(a) in idx:
        CEMB[idx[int(a)]] = e
als = np.load(DATA / "als_emb.npz")
AEMB = np.zeros((len(ids), als["emb"].shape[1]), dtype=np.float32)
for a, e in zip(als["ids"], als["emb"]):
    if int(a) in idx:
        AEMB[idx[int(a)]] = e
C = sp.load_npz(DATA / "cooc_csr.npz")
cm = np.load(DATA / "cooc_meta.npz")
cnt = cm["item_counts"]
cooc_row_of = {int(a): i for i, a in enumerate(cm["ids"])}

SEASON_RE = re.compile(
    r"(?:(\d)(?:nd|rd|th) season|season (\d)|part (\d)|\b(ii|iii|iv)\b)", re.I)
ROMAN = {"ii": 2, "iii": 3, "iv": 4}


def season_idx(aid: int) -> int:
    m = meta.get(aid)
    if not m:
        return 1
    for t in (m["name"], m["english"] or ""):
        mt = SEASON_RE.search(t)
        if mt:
            g = [g for g in mt.groups() if g]
            return ROMAN.get(g[0].lower(), None) or int(g[0])
    return 1


def feat_rows(s_id: int, cands: list[int], tt_q: np.ndarray,
              tt_emb: np.ndarray) -> np.ndarray:
    si = idx[s_id]
    crow = cooc_row_of.get(s_id)
    co = np.asarray(C[crow].todense()).ravel() if crow is not None else None
    ms = meta[s_id]
    sg = set(ms["genres"])
    st = set(ms["studios"])
    sy = year_of(s_id) or 2005
    ssea = season_idx(s_id)
    rows = []
    for c_id in cands:
        ci = idx[c_id]
        mc = meta[int(c_id)]
        cg = set(mc["genres"])
        lift = 0.0
        con = 0.0
        cj = cooc_row_of.get(int(c_id))
        if co is not None and cj is not None:
            con = float(co[cj])
            lift = con / (max(cnt[crow], 1) ** 0.65 * max(cnt[cj], 1) ** 0.65)
        rows.append([
            float(tt_q @ tt_emb[ci]),
            float(CEMB[si] @ CEMB[ci]),
            float(AEMB[si] @ AEMB[ci]),
            lift, np.log1p(con),
            len(sg & cg) / max(len(sg | cg), 1),
            abs((year_of(int(c_id)) or 2005) - sy),
            np.log1p(mc["popularity"] or 99999),
            np.log1p(ms["popularity"] or 99999),
            1.0 if mc["type"] == ms["type"] else 0.0,
            float(season_idx(int(c_id))), float(ssea),
            len(st & set(mc["studios"])) > 0,
            mc["score"] or 6.5,
        ])
    return np.array(rows, dtype=np.float32)


FEATS = ["tt_cos", "content_cos", "als_cos", "cooc_lift", "cooc_logcnt",
         "genre_jac", "year_gap", "cand_logpop", "src_logpop", "type_match",
         "cand_season", "src_season", "studio_match", "cand_score"]


def retrieve(tt_q: np.ndarray, tt_emb: np.ndarray, s_id: int) -> list[int]:
    sim = tt_emb @ tt_q
    sim[~retrieve_mask] = -np.inf
    if s_id in idx:
        sim[idx[s_id]] = -np.inf
    top = np.argpartition(-sim, TOP_CAND)[:TOP_CAND]
    return [int(ids[i]) for i in top[np.argsort(-sim[top])]]


by_src_votes = {s: dict(zip(g.dst, g.votes))
                for s, g in pairs.groupby("src")}
Xrows, ylab, groups, qsrc = [], [], [], []

for f in range(N_FOLDS):
    fold_srcs = {int(srcs[i]) for i in range(len(srcs)) if folds[i] == f}
    tp = train_pairs[~train_pairs.src.isin(fold_srcs)
                     & ~train_pairs.dst.isin(fold_srcs)]
    print(f"fold {f}: tower on {len(tp)} pairs", flush=True)
    tower, emb = tt.train_two_tower(ids, X, tp, epochs=30, seed=f,
                                    device="cuda")
    Xt = torch.tensor(X, device="cuda")
    for s_id in sorted(fold_srcs):
        truth = by_src_votes.get(s_id, {})
        ranked = sorted(truth, key=truth.get, reverse=True)
        top3, top10 = set(ranked[:3]), set(ranked[:10])
        tq = tt.encode_query(tower, Xt, [idx[s_id]])[0]
        cands = retrieve(tq, emb, s_id)
        Xrows.append(feat_rows(s_id, cands, tq, emb))
        ylab.append([3 if c in top3 else 2 if c in top10
                     else 1 if c in truth else 0 for c in cands])
        groups.append(len(cands))
        qsrc.append(s_id)

Xall = np.concatenate(Xrows)
yall = np.concatenate(ylab)
print(f"reranker dataset: {Xall.shape}, positives: {(yall > 0).mean():.2%}",
      flush=True)

rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400,
                    learning_rate=0.05, num_leaves=63, min_child_samples=20,
                    label_gain=[0, 1, 3, 7], random_state=0)
rk.fit(Xall, yall, group=groups)
rk.booster_.save_model(str(DATA / "reranker.txt"))
imp = sorted(zip(FEATS, rk.feature_importances_), key=lambda x: -x[1])
print("feature importance:", imp, flush=True)
print("saved reranker.txt", flush=True)

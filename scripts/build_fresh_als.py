"""Build a FRESH co-watch matrix + ALS from MAL-API user lists (2026 data).

Input: data/mal_lists.jsonl. Keeps watching/completed or scored entries,
users with >= 5 kept items, items in the content universe.
Output: data/als_fresh_emb.npz (ids aligned to content universe).
Does NOT overwrite als_emb.npz — features.py picks fresh via flag.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

ids = np.load(DATA / "content_emb.npz")["ids"]
idx = {int(a): i for i, a in enumerate(ids)}

rows, cols, vals = [], [], []
uid = 0
for line in open(DATA / "mal_lists.jsonl"):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    keep = [(a, sc) for a, sc, st in d["a"]
            if a in idx and (st in ("watching", "completed") or sc > 0)]
    if len(keep) < 5:
        continue
    for a, sc in keep:
        rows.append(uid)
        cols.append(idx[a])
        vals.append(1.0 + min(max(sc, 0), 10) / 10.0)
    uid += 1

mat = sp.csr_matrix((np.array(vals, dtype=np.float32),
                     (np.array(rows), np.array(cols))),
                    shape=(uid, len(ids)))
mat.sum_duplicates()
print(f"fresh matrix: {mat.shape}, nnz={mat.nnz}", flush=True)

from implicit.als import AlternatingLeastSquares  # noqa: E402

model = AlternatingLeastSquares(factors=256, iterations=25,
                                regularization=0.05, alpha=20.0,
                                random_state=42, use_gpu=False)
model.fit(mat)
emb = model.item_factors.astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
np.savez(DATA / "als_fresh_emb.npz", ids=ids, emb=emb)
print(f"saved als_fresh_emb.npz {emb.shape}", flush=True)

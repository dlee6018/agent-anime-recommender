"""Process the full 2023 animelists CSV: interactions matrix + item2vec.

1. interactions_csr.npz / interactions_meta.npz (overwrites the 2017 interim)
2. ALS-256 -> als_emb.npz (overwrites interim)
3. item2vec on my_last_updated-ordered user sequences -> i2v_emb.npz
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

ids = np.load(DATA / "content_emb.npz")["ids"]
item_idx = {int(a): i for i, a in enumerate(ids)}

print("scanning big CSV...", flush=True)
lf = (
    pl.scan_csv(DATA / "raw" / "animelists_filtered.csv",
                schema_overrides={"my_score": pl.Int32, "my_status": pl.Int32,
                                  "my_last_updated": pl.Int64})
    .select(["username", "anime_id", "my_score", "my_status",
             "my_last_updated"])
    .filter((pl.col("my_status").is_in([1, 2]) | (pl.col("my_score") > 0))
            & pl.col("anime_id").is_in(list(item_idx)))
)
df = lf.collect(streaming=True)
print(f"kept rows: {len(df)}", flush=True)

counts = df.group_by("username").len()
good = counts.filter(pl.col("len") >= 5)["username"]
df = df.filter(pl.col("username").is_in(good))
print(f">=5 filter: {len(df)} rows, {df['username'].n_unique()} users",
      flush=True)

uid = (df["username"].rank("dense") - 1).to_numpy().astype(np.int64)
iid = np.array([item_idx[a] for a in df["anime_id"].to_numpy()],
               dtype=np.int64)
score = df["my_score"].to_numpy().astype(np.float32)
weight = 1.0 + np.clip(score, 0, 10) / 10.0
mat = sp.csr_matrix((weight, (uid, iid)),
                    shape=(int(uid.max()) + 1, len(ids)))
mat.sum_duplicates()
sp.save_npz(DATA / "interactions_csr.npz", mat)
np.savez(DATA / "interactions_meta.npz", item_ids=ids, n_users=mat.shape[0])
print(f"matrix {mat.shape} nnz {mat.nnz}", flush=True)

print("ALS-256...", flush=True)
from src.models.cf import save_als, train_als  # noqa: E402

item_ids, emb = train_als(factors=256, iterations=25)
save_als(item_ids, emb)
print("ALS saved", flush=True)

print("item2vec sequences...", flush=True)
seq_df = (df.sort(["username", "my_last_updated"])
          .group_by("username", maintain_order=True)
          .agg(pl.col("anime_id")))
sentences = [[str(a) for a in row] for row in seq_df["anime_id"].to_list()]
del df, seq_df

from gensim.models import Word2Vec  # noqa: E402

w2v = Word2Vec(sentences, vector_size=200, window=10, min_count=5, sg=1,
               negative=10, workers=8, epochs=5, seed=42)
i2v = np.zeros((len(ids), 200), dtype=np.float32)
hit = 0
for a, i in item_idx.items():
    key = str(a)
    if key in w2v.wv:
        v = w2v.wv[key]
        i2v[i] = v / (np.linalg.norm(v) + 1e-9)
        hit += 1
np.savez(DATA / "i2v_emb.npz", ids=ids, emb=i2v)
print(f"item2vec saved ({hit}/{len(ids)} covered)", flush=True)

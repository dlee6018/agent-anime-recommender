"""Build a sparse user x item co-watch matrix from animelists_filtered.csv.

Keep rows where the user actually engaged: status watching(1)/completed(2) or
scored > 0. Users with >= 5 kept items. Items restricted to the content
universe (content_emb.npz ids) so all models share one candidate vocab.

Output: data/interactions.npz  (csr matrix data + item_ids vocab + n_users)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

item_ids = np.load(DATA / "content_emb.npz")["ids"]
item_set = set(int(i) for i in item_ids)
item_idx = {int(a): i for i, a in enumerate(item_ids)}
print(f"item vocab: {len(item_ids)}", flush=True)

lf = (
    pl.scan_csv(DATA / "raw" / "animelists_filtered.csv",
                schema_overrides={"my_score": pl.Int32, "my_status": pl.Int32})
    .select(["username", "anime_id", "my_score", "my_status"])
    .filter(
        (pl.col("my_status").is_in([1, 2]) | (pl.col("my_score") > 0))
        & pl.col("anime_id").is_in(list(item_set))
    )
)
df = lf.collect(streaming=True)
print(f"kept interactions: {len(df)}", flush=True)

df = df.with_columns(pl.col("username").rank("dense").alias("uid"))
counts = df.group_by("uid").len()
good = counts.filter(pl.col("len") >= 5)["uid"]
df = df.filter(pl.col("uid").is_in(good))
print(f"after >=5-item user filter: {len(df)} rows, "
      f"{df['uid'].n_unique()} users", flush=True)

uid = df["uid"].rank("dense").to_numpy() - 1
iid = np.array([item_idx[a] for a in df["anime_id"].to_numpy()], dtype=np.int32)
score = df["my_score"].to_numpy().astype(np.float32)
weight = np.ones(len(df), dtype=np.float32) + np.clip(score, 0, 10) / 10.0

n_users = int(uid.max()) + 1
mat = sp.csr_matrix((weight, (uid.astype(np.int64), iid)),
                    shape=(n_users, len(item_ids)))
mat.sum_duplicates()
sp.save_npz(DATA / "interactions_csr.npz", mat)
np.savez(DATA / "interactions_meta.npz", item_ids=item_ids, n_users=n_users)
print(f"saved: {mat.shape} matrix, nnz={mat.nnz}", flush=True)

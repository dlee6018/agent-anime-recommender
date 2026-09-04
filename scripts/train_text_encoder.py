"""Exp 53: END-TO-END fine-tuned text dual-encoder (bare-model lever).

Unlike the frozen-Qwen setup, this trains the encoder itself on rec pairs:
bge-large-en-v1.5 (335M) + MultipleNegativesRankingLoss over
(src_doc, dst_doc) text pairs, vote-weighted by duplication. Crowd data is
TRAINING supervision only — at inference the encoder sees just the query's
text, so it is bare-mode legal.

Output: models/text_encoder/ + data/content_emb_ft.npz (all 13k docs).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata, year_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="train_pairs.parquet")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--base", default="BAAI/bge-large-en-v1.5")
ap.add_argument("--out", default="models/text_encoder")
args = ap.parse_args()

meta = load_metadata()
import json  # noqa: E402

reviews = json.load(open(ROOT / "data" / "reviews.json"))


def doc(aid: int) -> str:
    m = meta[aid]
    bits = [m["name"]]
    if m["english"] and m["english"] != m["name"]:
        bits.append(m["english"])
    bits.append(f"({m['type']} {year_of(aid) or '?'})")
    tags = ", ".join((m["genres"] + m["themes"] + m["demographics"])[:8])
    if tags:
        bits.append(tags)
    syn = (m["synopsis"] or "").replace("\n", " ")
    if "no description available" in syn.lower()[:60]:
        syn = ""
    bits.append(syn[:600])
    rv = reviews.get(str(aid)) or []
    if rv:
        bits.append("Reviews: " + rv[0][:500])
    return ". ".join(bits)


pairs = pd.read_parquet(ROOT / "data" / args.pairs)
rows = []
for s, d_, v in zip(pairs.src, pairs.dst, pairs.votes):
    if int(s) in meta and int(d_) in meta:
        reps = 1 + min(int(np.log1p(v)), 3)  # vote-weighted duplication
        rows += [(doc(int(s)), doc(int(d_)))] * reps
print(f"training pairs (weighted): {len(rows)}", flush=True)

from sentence_transformers import (InputExample, SentenceTransformer,  # noqa: E402
                                   losses)
from torch.utils.data import DataLoader  # noqa: E402

model = SentenceTransformer(args.base, device="cuda")
model.max_seq_length = 384
data = [InputExample(texts=[a, b]) for a, b in rows]
loader = DataLoader(data, shuffle=True, batch_size=24, drop_last=True)
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
          warmup_steps=500, show_progress_bar=True)
out = ROOT / args.out
model.save(str(out))
print("encoder saved", flush=True)

ids = [int(a) for a in np.load(ROOT / "data" / "content_emb.npz")["ids"]]
emb = model.encode([doc(a) for a in ids], batch_size=64,
                   show_progress_bar=True, normalize_embeddings=True,
                   convert_to_numpy=True)
np.savez(ROOT / "data" / "content_emb_ft.npz",
         ids=np.array(ids, dtype=np.int64), emb=emb.astype(np.float32))
print(f"embedded universe: {emb.shape}", flush=True)

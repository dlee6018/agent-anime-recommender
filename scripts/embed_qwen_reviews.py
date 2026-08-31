"""Embed review-augmented content docs with Qwen3-Embedding-4B.

User insight (2026-08-30): MAL synopses are spoiler-free episode-1 blurbs;
shows that pivot mid-run (Madoka, School Days) are misrepresented. Review
excerpts describe the full arc. Doc = metadata header + synopsis (trimmed)
+ up to 2 review excerpts. Items without reviews keep the plain doc.

Output: data/content_emb_qwen_rev.npz (same id universe as content_emb.npz).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_metadata, year_of  # noqa: E402

MODEL = "Qwen/Qwen3-Embedding-4B"
DATA = Path(__file__).resolve().parent.parent / "data"

ids = [int(a) for a in np.load(DATA / "content_emb.npz")["ids"]]
meta = load_metadata()
reviews = json.load(open(DATA / "reviews.json"))

texts, n_rev = [], 0
for a in ids:
    m = meta[a]
    parts = [m["name"]]
    if m["english"] and m["english"] != m["name"]:
        parts.append(m["english"])
    parts.append(f"({m['type']}, {year_of(a) or 'unknown year'})")
    if m["genres"]:
        parts.append("Genres: " + ", ".join(m["genres"]))
    if m["themes"]:
        parts.append("Themes: " + ", ".join(m["themes"]))
    if m["demographics"]:
        parts.append("Demographic: " + ", ".join(m["demographics"]))
    if m["studios"]:
        parts.append("Studio: " + ", ".join(m["studios"]))
    syn = (m["synopsis"] or "").replace("\n", " ")
    if "no description available" in syn.lower()[:60]:
        syn = ""
    rv = reviews.get(str(a)) or []
    parts.append(syn[:800 if rv else 1500])
    if rv:
        n_rev += 1
        parts.append("Reviewer impressions: " +
                     " /// ".join(r[:1200] for r in rv[:2]))
    texts.append(". ".join(parts))

print(f"embedding {len(ids)} docs ({n_rev} review-augmented) with {MODEL}",
      flush=True)

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(
    MODEL, device="cuda",
    model_kwargs={"torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)
model.max_seq_length = 2048
emb = model.encode(texts, batch_size=8, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True)
# fp16 on disk (halves the file); build_features upcasts to fp32 on load
np.savez(DATA / "content_emb_qwen_rev.npz",
         ids=np.array(ids, dtype=np.int64),
         emb=emb.astype(np.float16), model=MODEL + "+reviews")
print(f"saved: {emb.shape}", flush=True)

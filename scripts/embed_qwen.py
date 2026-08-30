"""Embed enriched content docs with Qwen3-Embedding-4B (2560d) on GPU.

Output: data/content_emb_qwen.npz — same id universe as content_emb.npz.
"""
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

texts = []
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
        syn = ""  # placeholder text would be embedded as garbage
    parts.append(syn[:1500])
    texts.append(". ".join(parts))

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(
    MODEL, device="cuda",
    model_kwargs={"torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)
model.max_seq_length = 1024
emb = model.encode(texts, batch_size=16, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True)
np.savez(DATA / "content_emb_qwen.npz", ids=np.array(ids, dtype=np.int64),
         emb=emb.astype(np.float16), model=MODEL)
print(f"saved: {emb.shape}", flush=True)

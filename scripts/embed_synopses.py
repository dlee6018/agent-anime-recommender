"""Embed anime content text (title/genres/studios/synopsis) on GPU.

Output: data/content_emb.npz  (ids: int64[N], emb: float32[N, D], model name)
Universe: anime with a synopsis and popularity rank <= 12000 (covers every
plausible recommendation target while skipping the deep long tail).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_metadata, year_of  # noqa: E402

MODEL = "BAAI/bge-large-en-v1.5"
MAX_POP_RANK = 13000

meta = load_metadata()
ids = sorted(
    a for a, m in meta.items()
    if (m["popularity"] and m["popularity"] <= MAX_POP_RANK)
    and (len(m["synopsis"] or "") > 40 or m["genres"] or m["themes"])
)
print(f"embedding {len(ids)} anime with {MODEL}", flush=True)

texts = []
for a in ids:
    m = meta[a]
    parts = [m["name"]]
    if m["english"] and m["english"] != m["name"]:
        parts.append(m["english"])
    y = year_of(a)
    parts.append(f"({m['type']}, {y or 'unknown year'})")
    if m["genres"]:
        parts.append("Genres: " + ", ".join(m["genres"]))
    if m["themes"]:
        parts.append("Themes: " + ", ".join(m["themes"]))
    if m["demographics"]:
        parts.append("Demographic: " + ", ".join(m["demographics"]))
    if m["studios"]:
        parts.append("Studio: " + ", ".join(m["studios"]))
    parts.append((m["synopsis"] or "").replace("\n", " ")[:1500])
    texts.append(". ".join(parts))

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(MODEL, device="cuda")
emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True)
out = Path(__file__).resolve().parent.parent / "data" / "content_emb.npz"
np.savez(out, ids=np.array(ids, dtype=np.int64),
         emb=emb.astype(np.float32), model=MODEL)
print(f"saved {out}: {emb.shape}", flush=True)

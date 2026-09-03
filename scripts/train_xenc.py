"""Cross-encoder reranker: fine-tune bge-reranker-base on rec pairs.

Pairs: for each train-graph src, positives = top-10 targets by votes,
negatives = 3x random popular anime (excluding positives/franchise).
Text: compact doc (title, year, genres, themes, 300-char synopsis).
Output: models/xenc/ (HF checkpoint). Dev-honest by construction when
trained on train_pairs.parquet.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata, year_of  # noqa: E402
from src.franchise import same_franchise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="train_pairs.parquet")
ap.add_argument("--out", default="models/xenc")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--base", default="BAAI/bge-reranker-base")
args = ap.parse_args()

meta = load_metadata()


def doc(aid: int) -> str:
    m = meta[aid]
    bits = [m["name"]]
    if m["english"] and m["english"] != m["name"]:
        bits.append(m["english"])
    bits.append(f"({year_of(aid) or '?'})")
    tags = ", ".join((m["genres"] + m["themes"] + m["demographics"])[:8])
    if tags:
        bits.append(tags)
    syn = (m["synopsis"] or "").replace("\n", " ")
    if "no description available" in syn.lower()[:60]:
        syn = ""
    bits.append(syn[:300])
    return ". ".join(bits)


pairs = pd.read_parquet(ROOT / "data" / args.pairs)
rng = np.random.default_rng(0)
pool = [a for a, m in meta.items()
        if m["popularity"] and m["popularity"] <= 3000]

examples = []
for s, g in pairs.groupby("src"):
    s = int(s)
    if s not in meta:
        continue
    pos = [int(d) for d in g.sort_values("votes", ascending=False)
           .dst.head(10) if int(d) in meta]
    posset = set(pos)
    sd = doc(s)
    for p in pos:
        examples.append((sd, doc(p), 1.0))
        for _ in range(3):
            n = int(pool[rng.integers(len(pool))])
            if n != s and n not in posset and not same_franchise(n, s):
                examples.append((sd, doc(n), 0.0))
print(f"examples: {len(examples)}", flush=True)

from sentence_transformers import CrossEncoder, InputExample  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

model = CrossEncoder(args.base, num_labels=1, max_length=384, device="cuda")
data = [InputExample(texts=[a, b], label=y) for a, b, y in examples]
loader = DataLoader(data, shuffle=True, batch_size=24)
model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=500,
          output_path=str(ROOT / args.out), show_progress_bar=True)
print("saved", ROOT / args.out, flush=True)

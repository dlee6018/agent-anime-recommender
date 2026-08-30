"""Fetch Jikan /anime/{id}/full for the working universe (resumable).

Universe: top-2500 popularity content ids + all dev/eval queries + all truth
ids + rec-pair endpoints missing from the 2023 metadata dump.
Saves selected fields to data/jikan_meta.json incrementally.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "jikan_meta.json"

meta = load_metadata()
ids = set()
for a, m in meta.items():
    if m["popularity"] and m["popularity"] <= 2500:
        ids.add(a)
for f in ("eval_set.json", "dev_set.json"):
    d = json.load(open(DATA / f))["queries"]
    ids |= {int(q) for q in d}
    ids |= {int(r) for v in d.values() for r in v}
pairs = pd.read_parquet(DATA / "rec_pairs.parquet")
ids |= {int(x) for x in np.concatenate([pairs.src.unique(), pairs.dst.unique()])
        if int(x) not in meta}

done = json.load(open(OUT)) if OUT.exists() else {}
todo = [i for i in sorted(ids) if str(i) not in done]
print(f"universe {len(ids)}, todo {len(todo)}", flush=True)


UA = {"User-Agent": "anime-rec-research/0.1"}


def get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=12))
        except Exception as e:
            if t == tries - 1:
                print(f"giving up {url}: {type(e).__name__} {e}", flush=True)
            time.sleep(2 + 2 * t)
    return None


for n, aid in enumerate(todo):
    d = get(f"https://api.jikan.moe/v4/anime/{aid}/full")
    if d and "data" in d:
        a = d["data"]
        done[str(aid)] = {
            "title": a.get("title"),
            "title_english": a.get("title_english"),
            "synopsis": a.get("synopsis"),
            "type": a.get("type"),
            "year": a.get("year") or (a.get("aired") or {}).get("prop", {})
                .get("from", {}).get("year"),
            "score": a.get("score"),
            "members": a.get("members"),
            "popularity": a.get("popularity"),
            "genres": [g["name"] for g in a.get("genres", [])],
            "themes": [g["name"] for g in a.get("themes", [])],
            "demographics": [g["name"] for g in a.get("demographics", [])],
            "studios": [g["name"] for g in a.get("studios", [])],
            "source": a.get("source"),
            "episodes": a.get("episodes"),
            "rating": a.get("rating"),
            "relations": [
                {"rel": r["relation"],
                 "ids": [e["mal_id"] for e in r["entry"]
                         if e["type"] == "anime"]}
                for r in a.get("relations", [])
            ],
        }
    else:
        done[str(aid)] = None
    if n % 25 == 24 or n == len(todo) - 1:
        json.dump(done, open(OUT, "w"))
    if n % 10 == 9:
        ok = sum(1 for v in done.values() if v)
        print(f"[{n + 1}/{len(todo)}] ok={ok}", flush=True)
    time.sleep(1.05)

json.dump(done, open(OUT, "w"))
ok = sum(1 for v in done.values() if v)
print(f"DONE: {ok} ok / {len(done)}", flush=True)

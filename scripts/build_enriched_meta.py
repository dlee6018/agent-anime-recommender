"""Merge 2023 dump (synopses) + lyfesan 2025 dump (themes/demo/producers,
fresh popularity/members) into data/meta_enriched.parquet, and re-resolve
rec-pair titles against the wider title table (recovers pairs dropped for
unmappable post-2023 titles).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    return " ".join(s.split())


old = pl.read_csv(DATA / "raw" / "anime_metadata_2023.csv",
                  infer_schema_length=60000, null_values=["UNKNOWN", "N/A"])
new = pl.read_parquet(DATA / "raw" / "lyfesan.parquet")

o = {r["anime_id"]: r for r in old.to_dicts()}
rows = []
seen = set()
for r in new.to_dicts():
    aid = r["mal_id"]
    seen.add(aid)
    olds = o.get(aid, {})
    members = int(str(r["Members"]).replace(",", "") or 0)
    poprank = int(str(r["Popularity"]).lstrip("#").replace(",", "") or 0) or None
    year = None
    m = re.search(r"(\d{4})", str(r["Aired"] or "") + " " + str(olds.get("Aired") or ""))
    if m:
        year = int(m.group(1))
    rows.append({
        "mal_id": aid,
        "name": r["Title"],
        "english": r["English"] or olds.get("English name"),
        "synonyms": r["Synonyms"],
        "genres": r["Genres"] or olds.get("Genres") or "",
        "themes": r["Themes"] or "",
        "demographics": r["Demographic"] or "",
        "studios": r["Studios"] or olds.get("Studios") or "",
        "producers": r["Producers"] or "",
        "type": r["Type"] or olds.get("Type"),
        "source": r["Source"] or olds.get("Source"),
        "rating": r["Rating"] or olds.get("Rating"),
        "score": r["Score"] if r["Score"] is not None else olds.get("Score"),
        "members": members,
        "popularity": poprank,
        "year": year,
        "synopsis": olds.get("Synopsis") or "",
    })
# 2023-dump anime absent from lyfesan (long tail): keep, mark stale popularity
for aid, olds in o.items():
    if aid in seen:
        continue
    yr = None
    m = re.search(r"(\d{4})", str(olds.get("Aired") or ""))
    if m:
        yr = int(m.group(1))
    pop = olds.get("Popularity")
    rows.append({
        "mal_id": aid, "name": olds.get("Name"),
        "english": olds.get("English name"), "synonyms": None,
        "genres": olds.get("Genres") or "", "themes": "", "demographics": "",
        "studios": olds.get("Studios") or "", "producers": olds.get("Producers") or "",
        "type": olds.get("Type"), "source": olds.get("Source"),
        "rating": olds.get("Rating"), "score": olds.get("Score"),
        "members": 0, "popularity": int(float(pop)) if pop else None,
        "year": yr, "synopsis": olds.get("Synopsis") or "",
    })

df = pd.DataFrame(rows)
df.to_parquet(DATA / "meta_enriched.parquet")
print(f"enriched meta: {len(df)} anime "
      f"({(df.themes != '').mean():.0%} themed)")

# --- re-resolve rec pairs with the wider title table ---
t2i = {}
meta_by_id = {r["mal_id"]: r for r in rows}
order = sorted(rows, key=lambda r: r["popularity"] or 10**9)
for r in order:
    cands = [r["name"], r["english"]] + \
        [s.strip() for s in (r["synonyms"] or "").split(",")]
    for t in cands:
        if t:
            t2i.setdefault(norm(t), r["mal_id"])

pat = re.compile(r'(\d+) (?:people|person) says? the anime "(.+?)" is a similar show')
d = json.load(open(DATA / "raw" / "mal_recs_ayan4m1.json"))
out, dropped = [], 0
for rec in d:
    m = pat.match(rec["messages"][1]["content"])
    s, t = t2i.get(norm(rec["show"])), t2i.get(norm(m.group(2)))
    if s and t and s != t:
        out.append((s, t, int(m.group(1))))
    else:
        dropped += 1
pairs = pd.DataFrame(out, columns=["src", "dst", "votes"]).drop_duplicates(
    ["src", "dst"])
pairs.to_parquet(DATA / "rec_pairs.parquet")
print(f"rec pairs: {len(pairs)} (dropped {dropped}, "
      f"was 84402/4500)")

# refresh holdout-filtered variants
eval_ids = {int(q) for q in json.load(open(DATA / "eval_set.json"))["queries"]}
dev_ids = {int(q) for q in json.load(open(DATA / "dev_set.json"))["queries"]}
pe = pairs[~pairs.src.isin(eval_ids) & ~pairs.dst.isin(eval_ids)]
pe.to_parquet(DATA / "train_pairs_eval.parquet")
held = eval_ids | dev_ids
pd_ = pairs[~pairs.src.isin(held) & ~pairs.dst.isin(held)]
pd_.to_parquet(DATA / "train_pairs.parquet")
print(f"train_pairs_eval {len(pe)}, train_pairs(dev) {len(pd_)}")

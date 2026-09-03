"""Scrape AniList's recommendation graph (batched GraphQL), keyed by MAL id.

An independent platform's crowd answering the same question — legal under
the eval protocol (held-out data = MAL userrecs pages; this is AniList's
own user-voted rec graph). Judgment call logged in experiments.md exp 48.

Output: data/anilist_recs.json {mal_id: [[rec_mal_id, rating], ...]}
Resumable; ~320 batched requests for the pop<=8000 universe.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "anilist_recs.json"
BATCH = 25

QUERY = """query($ids:[Int]){Page(page:1,perPage:%d){
media(idMal_in:$ids,type:ANIME){idMal
recommendations(sort:RATING_DESC,perPage:12){
nodes{rating mediaRecommendation{idMal type}}}}}}""" % BATCH


def post(query: str, variables: dict, tries: int = 5):
    body = json.dumps({"query": query, "variables": variables}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(
                "https://graphql.anilist.co", data=body,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:
            wait = 10 + 15 * t  # generous: API just recovered from outage
            print(f"retry {t}: {type(e).__name__}, sleeping {wait}s",
                  flush=True)
            time.sleep(wait)
    return None


meta = load_metadata()
ids = np.load(DATA / "content_emb.npz")["ids"]
qids = []
for f in ("eval_set.json", "dev_set.json"):
    qids += [int(q) for q in json.load(open(DATA / f))["queries"]]
rest = sorted((int(a) for a in ids
               if meta[int(a)]["popularity"]
               and meta[int(a)]["popularity"] <= 8000
               and int(a) not in set(qids)),
              key=lambda a: meta[a]["popularity"])
targets = qids + rest

done = json.load(open(OUT)) if OUT.exists() else {}
todo = [a for a in targets if str(a) not in done]
print(f"targets {len(targets)}, todo {len(todo)}", flush=True)

for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    d = post(QUERY, {"ids": chunk})
    media = (d or {}).get("data", {}).get("Page", {}).get("media") or []
    found = set()
    for m in media:
        mid = m.get("idMal")
        if mid is None:
            continue
        found.add(int(mid))
        recs = []
        for n in (m.get("recommendations", {}).get("nodes") or []):
            mr = n.get("mediaRecommendation")
            if mr and mr.get("idMal") and mr.get("type") == "ANIME":
                recs.append([int(mr["idMal"]), int(n.get("rating") or 0)])
        done[str(mid)] = recs
    for a in chunk:  # anime unknown to AniList -> empty (not retryable)
        if a not in found:
            done[str(a)] = []
    if (i // BATCH) % 10 == 9 or i + BATCH >= len(todo):
        json.dump(done, open(OUT, "w"))
        ok = sum(1 for v in done.values() if v)
        print(f"[{min(i + BATCH, len(todo))}/{len(todo)}] with-recs={ok}",
              flush=True)
    time.sleep(1.8)

json.dump(done, open(OUT, "w"))
ok = sum(1 for v in done.values() if v)
print(f"DONE: {ok}/{len(done)} have AniList recs", flush=True)

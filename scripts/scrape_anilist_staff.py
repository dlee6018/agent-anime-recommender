"""Scrape staff (director/writer/source-author) + studios from AniList.

Factual metadata, not crowd recs — bare-mode legal at inference. Targets the
same-author/same-director failure class (Bakemonogatari->Nisio, Fate/Zero->
Urobuchi). Output: data/anilist_staff.json {mal_id: [[role, name], ...]}.
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
OUT = DATA / "anilist_staff.json"
BATCH = 20

QUERY = """query($ids:[Int]){Page(page:1,perPage:%d){
media(idMal_in:$ids,type:ANIME){idMal
staff(perPage:10,sort:RELEVANCE){edges{role node{name{full}}}}}}}""" % BATCH


def post(variables, tries=4):
    body = json.dumps({"query": QUERY, "variables": variables}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(
                "https://graphql.anilist.co", data=body,
                headers={"Content-Type": "application/json",
                         "User-Agent": "ani-rec-research/0.1"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            ra = e.headers.get("Retry-After")
            time.sleep(int(ra) + 2 if ra and ra.isdigit() else 20 + 20 * t)
        except Exception:
            time.sleep(20)
    return None


meta = load_metadata()
ids = np.load(DATA / "content_emb.npz")["ids"]
targets = sorted((int(a) for a in ids
                  if meta[int(a)]["popularity"]
                  and meta[int(a)]["popularity"] <= 8000),
                 key=lambda a: meta[a]["popularity"])
done = json.load(open(OUT)) if OUT.exists() else {}
todo = [a for a in targets if str(a) not in done]
print(f"todo {len(todo)}", flush=True)

KEEP = ("Director", "Original Creator", "Original Story", "Script",
        "Series Composition", "Music", "Character Design")

for i in range(0, len(todo), BATCH):
    chunk = todo[i:i + BATCH]
    d = post({"ids": chunk})
    if d is None or not d.get("data"):
        print(f"batch {i} failed — resume later", flush=True)
        continue
    found = set()
    for m in (d["data"].get("Page", {}).get("media") or []):
        mid = m.get("idMal")
        if mid is None:
            continue
        found.add(int(mid))
        staff = []
        for e in (m.get("staff", {}).get("edges") or []):
            role = (e.get("role") or "").split("(")[0].strip()
            name = ((e.get("node") or {}).get("name") or {}).get("full")
            if name and any(k in role for k in KEEP):
                staff.append([role, name])
        done[str(mid)] = staff
    for a in chunk:
        if a not in found:
            done[str(a)] = []
    if (i // BATCH) % 10 == 9 or i + BATCH >= len(todo):
        json.dump(done, open(OUT, "w"))
        print(f"[{min(i + BATCH, len(todo))}/{len(todo)}]", flush=True)
    time.sleep(2.5)

json.dump(done, open(OUT, "w"))
print(f"DONE: {sum(1 for v in done.values() if v)}/{len(done)} with staff",
      flush=True)

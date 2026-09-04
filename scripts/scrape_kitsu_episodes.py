"""Per-episode synopses via Kitsu (user feature: 'summary analyzing each
episode' — catches mid-series twists that MAL's spoiler-free blurb hides).

MAL id -> kitsu id (mappings endpoint) -> paginated episodes.
Output: data/kitsu_episodes.json {mal_id: [ep_synopsis, ...]} (resumable).
Priority: eval+dev queries, then popularity <= 3000, TV/ONA only.
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
OUT = DATA / "kitsu_episodes.json"
H = {"Accept": "application/vnd.api+json",
     "User-Agent": "ani-rec-research/0.1"}


def get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=H)
            return json.load(urllib.request.urlopen(req, timeout=20))
        except Exception:
            time.sleep(3 + 5 * t)
    return None


meta = load_metadata()
ids = np.load(DATA / "content_emb.npz")["ids"]
qids = []
for f in ("eval_set.json", "dev_set.json"):
    qids += [int(q) for q in json.load(open(DATA / f))["queries"]]
rest = sorted((int(a) for a in ids
               if int(a) not in set(qids)
               and meta[int(a)]["popularity"]
               and meta[int(a)]["popularity"] <= 3000
               and meta[int(a)]["type"] in ("TV", "ONA")),
              key=lambda a: meta[a]["popularity"])
targets = qids + rest

done = json.load(open(OUT)) if OUT.exists() else {}
todo = [a for a in targets if str(a) not in done]
print(f"todo {len(todo)}", flush=True)

for n, mal in enumerate(todo):
    m = get(f"https://kitsu.app/api/edge/mappings"
            f"?filter%5BexternalSite%5D=myanimelist/anime"
            f"&filter%5BexternalId%5D={mal}&include=item")
    inc = (m or {}).get("included") or []
    kid = next((x["id"] for x in inc if x.get("type") == "anime"), None)
    if not kid:
        done[str(mal)] = []
    else:
        eps, off = [], 0
        while off < 60:  # cap 3 pages / 60 eps
            d = get(f"https://kitsu.app/api/edge/anime/{kid}/episodes"
                    f"?page%5Blimit%5D=20&page%5Boffset%5D={off}"
                    f"&sort=number")
            batch = (d or {}).get("data") or []
            for e in batch:
                syn = (e["attributes"].get("synopsis") or "").strip()
                if len(syn) > 30:
                    eps.append(" ".join(syn.split())[:400])
            if len(batch) < 20:
                break
            off += 20
            time.sleep(0.4)
        done[str(mal)] = eps
    if n % 25 == 24 or n == len(todo) - 1:
        json.dump(done, open(OUT, "w"))
        ok = sum(1 for v in done.values() if len(v) >= 3)
        print(f"[{n + 1}/{len(todo)}] with-arcs={ok}", flush=True)
    time.sleep(0.5)

json.dump(done, open(OUT, "w"))
ok = sum(1 for v in done.values() if len(v) >= 3)
print(f"DONE: {ok}/{len(done)} have episode arcs", flush=True)

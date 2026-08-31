"""Scrape top MAL review excerpts per anime (user insight: synopses are
spoiler-free ep-1 blurbs; reviews describe what a show actually becomes —
twists, tone shifts — which is what drives crowd recommendations).

Coverage: eval+dev queries first, then popularity rank <= 4000.
Output: data/reviews.json {mal_id: [excerpt, ...]} (resumable).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_metadata  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "reviews.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
N_REVIEWS = 3
MAX_CHARS = 1400

# take a generous raw slice after the marker rather than stopping at the
# first </div> — review bodies contain nested divs (reviewer #2, item 1)
TEXT_RE = re.compile(r'class="text">(.{200,8000}?)(?:</div>\s*</div>|$)', re.S)
TAG_RE = re.compile(r"<[^>]+>")


def fetch(url: str, tries: int = 3) -> str | None:
    for t in range(tries):
        r = subprocess.run(["curl", "-s", "--max-time", "25", "-A", UA, url],
                           capture_output=True, timeout=40)
        if r.returncode == 0 and len(r.stdout) > 10000:
            return r.stdout.decode(errors="replace")
        time.sleep(2 + 3 * t)
    return None


def parse(html: str) -> list[str]:
    out = []
    for block in TEXT_RE.findall(html)[:N_REVIEWS]:
        txt = TAG_RE.sub(" ", block)
        txt = " ".join(txt.split())
        if len(txt) > 100:
            out.append(txt[:MAX_CHARS])
    return out


meta = load_metadata()
qids = []
for f in ("eval_set.json", "dev_set.json"):
    qids += [int(q) for q in json.load(open(DATA / f))["queries"]]
rest = sorted((a for a, m in meta.items()
               if m["popularity"] and m["popularity"] <= 4000
               and a not in set(qids)),
              key=lambda a: meta[a]["popularity"])
targets = qids + rest

done = json.load(open(OUT)) if OUT.exists() else {}
todo = [a for a in targets if str(a) not in done]
print(f"targets {len(targets)}, todo {len(todo)}", flush=True)

for n, aid in enumerate(todo):
    m = meta[aid]
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", m["name"])[:60] or "x"
    html = fetch(f"https://myanimelist.net/anime/{aid}/{slug}/reviews")
    # None = retryable fetch failure; [] = page fetched, no usable reviews
    done[str(aid)] = parse(html) if html else None
    if n % 25 == 24 or n == len(todo) - 1:
        json.dump(done, open(OUT, "w"))
    if n % 100 == 99:
        ok = sum(1 for v in done.values() if v)
        print(f"[{n + 1}/{len(todo)}] with-reviews={ok}", flush=True)
    time.sleep(2.2)

json.dump(done, open(OUT, "w"))
ok = sum(1 for v in done.values() if v)
print(f"DONE: {ok}/{len(done)} have reviews", flush=True)

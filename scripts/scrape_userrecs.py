"""Scrape MAL userrecs pages for a list of anime IDs.

Output: data/userrecs.json  {mal_id: [rec_mal_id, ...]}  in MAL's
"most recommended first" page order. Resumable: skips IDs already present.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "userrecs_votes.json"  # {src: [[dst, votes], ...]} page order
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
PAIR_RE = re.compile(r'href="/recommendations/anime/(\d+)-(\d+)"')
MORE_RE = re.compile(r"by <strong>(\d+)</strong> more users")


def fetch(url: str, tries: int = 4) -> str:
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=30).read().decode()
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(5 + 5 * t)


def parse_recs(html: str, mal_id: int) -> list[list[int]]:
    """[[rec_id, votes], ...] in page order (most-recommended first)."""
    parts = re.split(r'href="/recommendations/anime/(\d+)-(\d+)"', html)
    recs, seen = [], set()
    for i in range(1, len(parts) - 2, 3):
        a, b, seg = int(parts[i]), int(parts[i + 1]), parts[i + 2]
        other = b if a == mal_id else a
        if other == mal_id or other in seen:
            continue
        seen.add(other)
        m = MORE_RE.search(seg)
        recs.append([other, (int(m.group(1)) + 1) if m else 1])
    return recs


def main() -> None:
    tfile = sys.argv[1] if len(sys.argv) > 1 else "top200_popularity.json"
    targets = json.load(open(DATA / tfile))
    done = json.load(open(OUT)) if OUT.exists() else {}
    for i, a in enumerate(targets):
        mid = str(a["mal_id"])
        if mid in done:
            continue
        slug = urllib.parse.quote(a["slug"])
        url = f"https://myanimelist.net/anime/{mid}/{slug}/userrecs"
        try:
            html = fetch(url)
            done[mid] = parse_recs(html, int(mid))
        except Exception as e:
            print(f"FAILED {mid} {a['slug']}: {e}", flush=True)
            done[mid] = None  # mark attempted-and-failed; rescrape by deleting key
        if i % 20 == 0 or i == len(targets) - 1:
            json.dump(done, open(OUT, "w"))
            print(f"[{i + 1}/{len(targets)}] saved, last={a['slug']} "
                  f"n_recs={len(done[mid]) if done[mid] else 0}", flush=True)
        time.sleep(2.5)
    json.dump(done, open(OUT, "w"))
    ok = [k for k, v in done.items() if v]
    print(f"DONE: {len(ok)} scraped ok, {len(done) - len(ok)} failed", flush=True)


if __name__ == "__main__":
    main()

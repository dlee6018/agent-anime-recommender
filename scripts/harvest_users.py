"""Harvest MAL usernames from userrecs pages (the people who WRITE recs).

Sources: top-200 + extended targets' userrecs pages. Output:
data/mal_usernames.json (list). ~700 pages, curl fetcher, 2.5s spacing.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
PROF_RE = re.compile(r'href="/profile/([A-Za-z0-9_-]{2,20})"')
OUT = DATA / "mal_usernames.json"
STATE = DATA / "mal_usernames_state.json"


def fetch(url):
    for t in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "25", "-A", UA, url],
                           capture_output=True, timeout=40)
        if r.returncode == 0 and len(r.stdout) > 10000:
            return r.stdout.decode(errors="replace")
        time.sleep(3 + 4 * t)
    return None


targets = json.load(open(DATA / "top200_popularity.json"))
ext = json.load(open(DATA / "scrape_targets_2500.json"))[:500]
todo_all = [(a["mal_id"], a["slug"]) for a in targets] + \
           [(a["mal_id"], a["slug"]) for a in ext]

seen_pages = set(json.load(open(STATE))) if STATE.exists() else set()
users = set(json.load(open(OUT))) if OUT.exists() else set()

import urllib.parse

for mid, slug in todo_all:
    if str(mid) in seen_pages:
        continue
    html = fetch(f"https://myanimelist.net/anime/{mid}/"
                 f"{urllib.parse.quote(slug)}/userrecs")
    if html:
        users.update(PROF_RE.findall(html))
        seen_pages.add(str(mid))
    if len(seen_pages) % 25 == 0:
        json.dump(sorted(users), open(OUT, "w"))
        json.dump(sorted(seen_pages), open(STATE, "w"))
        print(f"[{len(seen_pages)}/{len(todo_all)}] users={len(users)}",
              flush=True)
    time.sleep(2.5)

json.dump(sorted(users), open(OUT, "w"))
json.dump(sorted(seen_pages), open(STATE, "w"))
print(f"DONE: {len(users)} unique usernames", flush=True)

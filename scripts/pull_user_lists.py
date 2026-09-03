"""Pull MAL user anime-lists via the official API (X-MAL-CLIENT-ID).

Input: data/mal_usernames.json. Output: data/mal_lists.jsonl (append,
resumable via data/mal_lists_done.json). ~1 req/s; private/deleted users
skipped. Caps at MAX_USERS.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENV = Path.home() / ".anime-rec-mal.env"
CID = dict(line.split("=", 1) for line in
           ENV.read_text().strip().splitlines())["MAL_CLIENT_ID"]
MAX_USERS = 25000

users = json.load(open(DATA / "mal_usernames.json"))
done_f = DATA / "mal_lists_done.json"
done = set(json.load(open(done_f))) if done_f.exists() else set()
out_f = open(DATA / "mal_lists.jsonl", "a")

n_ok = 0
for name in users[:MAX_USERS]:
    if name in done:
        continue
    url = (f"https://api.myanimelist.net/v2/users/"
           f"{urllib.parse.quote(name)}/animelist"
           f"?fields=list_status&limit=1000&nsfw=true")
    try:
        req = urllib.request.Request(url, headers={"X-MAL-CLIENT-ID": CID})
        d = json.load(urllib.request.urlopen(req, timeout=20))
        entries = [[e["node"]["id"],
                    e["list_status"].get("score", 0),
                    e["list_status"].get("status", "")]
                   for e in d.get("data", [])]
        out_f.write(json.dumps({"u": name, "a": entries}) + "\n")
        out_f.flush()
        n_ok += 1
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(30)
            continue  # retry same user next pass? simpler: mark done, move on
        # 403 private / 404 deleted — skip
    except Exception:
        time.sleep(5)
    done.add(name)
    if len(done) % 200 == 0:
        json.dump(sorted(done), open(done_f, "w"))
        print(f"[{len(done)}] lists={n_ok}", flush=True)
    time.sleep(0.8)

json.dump(sorted(done), open(done_f, "w"))
print(f"DONE: {n_ok} lists pulled", flush=True)

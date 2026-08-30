"""Open-ended LLM recommender: Qwen3-32B nominates MAL-style recs directly;
titles are resolved to catalog ids, franchise-filtered, fused with pipeline.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_metadata, resolve_title, titles, year_of  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.franchise import franchise_filter, same_franchise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eval_set", default="data/dev_set.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--fuse", type=float, default=0.5,
                help="weight of LLM nominations vs base pipeline (RRF)")
args = ap.parse_args()

CACHE = ROOT / "data" / "llm_gen_cache.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
meta = load_metadata()
tt = titles()


def llm(prompt: str) -> str:
    if prompt in cache:
        return cache[prompt]
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    text = out["choices"][0]["message"]["content"]
    cache[prompt] = text
    json.dump(cache, open(CACHE, "w"))
    return text


def nominate(q: int, n: int = 14) -> list[int]:
    y = year_of(q) or "?"
    prompt = (
        f"Fans of the anime \"{tt[q]}\" ({y}) on MyAnimeList commonly "
        f"recommend certain other anime as similar (the user-voted "
        f"Recommendations section). List the {n} anime most commonly "
        f"recommended to fans of \"{tt[q]}\", ordered from most to least "
        f"recommended. Do not include sequels, prequels, or spin-offs of "
        f"\"{tt[q]}\" itself. Answer with one anime title per line, no "
        f"numbering, no commentary.")
    text = llm(prompt)
    out, seen = [], set()
    for line in text.splitlines():
        t = re.sub(r"^[\s\d\.\-\*•]+", "", line).strip().strip('"')
        if not t or len(t) > 120:
            continue
        aid = resolve_title(t)
        if aid and aid not in seen and aid != q:
            seen.add(aid)
            out.append(aid)
    return out


def make_fn(base_fn):
    def recommend(query_ids: list[int], k: int) -> list[int]:
        q = query_ids[0]
        keep = franchise_filter(query_ids)
        noms = [c for c in nominate(q) if keep(c)]
        base = base_fn(query_ids, 25)
        score = {}
        for r, c in enumerate(noms):
            score[c] = score.get(c, 0) + args.fuse / (r + 3)
        for r, c in enumerate(base):
            score[c] = score.get(c, 0) + (1 - args.fuse) / (r + 3)
        ranked = sorted(score, key=score.get, reverse=True)
        out = []
        for c in ranked:
            if c in query_ids or not keep(c):
                continue
            if any(same_franchise(c, o) for o in out):
                continue
            out.append(c)
            if len(out) == k:
                break
        return out
    return recommend


from src.registry import get_model  # noqa: E402

base = get_model("best")
ev = {int(q): [int(r) for r in v]
      for q, v in json.load(open(ROOT / args.eval_set))["queries"].items()}
if args.limit:
    ev = dict(list(ev.items())[:args.limit])

fn = make_fn(base)
res = evaluate(fn, ev)
pure = evaluate(lambda qs, k: [c for c in nominate(qs[0])
                               if franchise_filter(qs)(c)][:k], ev)
base_res = evaluate(base, ev)
print(f"base={base_res['precision_at_k']:.3f}  "
      f"LLM-pure={pure['precision_at_k']:.3f}  "
      f"fused={res['precision_at_k']:.3f}  (n={len(ev)})")

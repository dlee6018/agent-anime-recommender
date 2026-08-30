"""LLM listwise reranker: Qwen3-32B reorders the champion pipeline's top-25.

Requires a running llama-server (scripts/serve_llm.sh). Evaluates on dev by
default (--eval_set data/eval_set.json + --graph train_pairs_eval.parquet for
milestone reads). Caches LLM answers to data/llm_cache.json.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_metadata, titles, year_of  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.franchise import with_franchise_filter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eval_set", default="data/dev_set.json")
ap.add_argument("--graph", default=None,
                help="override graph file for the base pipeline")
ap.add_argument("--n_cand", type=int, default=25)
ap.add_argument("--fuse", type=float, default=0.5,
                help="weight of LLM rank in reciprocal-rank fusion (0=off)")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--limit", type=int, default=0, help="only first N queries")
args = ap.parse_args()

CACHE = ROOT / "data" / "llm_cache.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}

meta = load_metadata()
tt = titles()


def llm(prompt: str) -> str:
    key = prompt[:2000]
    if key in cache:
        return cache[key]
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    text = out["choices"][0]["message"]["content"]
    cache[key] = text
    json.dump(cache, open(CACHE, "w"))
    return text


def describe(aid: int) -> str:
    m = meta[aid]
    y = year_of(aid) or "?"
    return f"{m['name']} ({m['type']} {y})"


def make_llm_recommender(base_fn):
    def recommend(query_ids: list[int], k: int) -> list[int]:
        cands = base_fn(query_ids, args.n_cand)
        q = query_ids[0]
        listing = "\n".join(f"{i + 1}. {describe(c)}"
                            for i, c in enumerate(cands))
        prompt = (
            f"On MyAnimeList, the anime \"{describe(q)}\" has a user-voted "
            f"'Recommendations' list of similar anime.\n"
            f"From the candidates below, choose the {k + 5} that MAL users "
            f"most commonly recommend to fans of \"{tt[q]}\", ordered from "
            f"most to least recommended.\n\nCandidates:\n{listing}\n\n"
            f"Answer with ONLY the candidate numbers, comma-separated, "
            f"best first.")
        try:
            text = llm(prompt)
            nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", text)
                    if 1 <= int(x) <= len(cands)]
            seen, order = set(), []
            for n in nums:
                if n - 1 not in seen:
                    seen.add(n - 1)
                    order.append(n - 1)
        except Exception as e:
            print("LLM error:", e, file=sys.stderr)
            order = []
        if not order:
            return cands[:k]
        if args.fuse > 0:
            # reciprocal-rank fusion: base rank + LLM rank
            score = {}
            for r, i in enumerate(order):
                score[i] = score.get(i, 0) + args.fuse / (r + 3)
            for i in range(len(cands)):
                score[i] = score.get(i, 0) + (1 - args.fuse) / (i + 3)
            final = sorted(score, key=score.get, reverse=True)
        else:
            final = order + [i for i in range(len(cands)) if i not in order]
        return [cands[i] for i in final[:k]]

    return recommend


if args.graph:
    cfg = json.load(open(ROOT / "data" / "best_pipeline.json"))
    cfg["graph"] = args.graph
    json.dump(cfg, open(ROOT / "data" / "best_pipeline.json", "w"))

from src.registry import get_model  # noqa: E402  (after graph override)

base = get_model("best")
ev = {int(q): [int(r) for r in v]
      for q, v in json.load(open(ROOT / args.eval_set))["queries"].items()}
if args.limit:
    ev = dict(list(ev.items())[:args.limit])

# base is already franchise-filtered; no second wrap needed
fn = make_llm_recommender(base)
res = evaluate(fn, ev)
base_res = evaluate(base, ev)
print(f"base P@5={base_res['precision_at_k']:.3f}  "
      f"LLM-fused P@5={res['precision_at_k']:.3f}  "
      f"(fuse={args.fuse}, n={len(ev)})")

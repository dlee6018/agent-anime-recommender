#!/usr/bin/env python3
"""CLI: top-k anime recommendations for one or more input anime.

    recommend.py "Death Note" -k 5
    recommend.py "Death Note" "Monster" -k 10 --model best
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import resolve_title, titles  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("anime", nargs="+", help="anime title(s) you like")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", default="best")
    args = ap.parse_args()

    query_ids = []
    for t in args.anime:
        aid = resolve_title(t)
        if aid is None:
            sys.exit(f"could not resolve title: {t!r}")
        query_ids.append(aid)
    tt = titles()
    print("query:", ", ".join(tt[q] for q in query_ids))

    from src.registry import get_model  # deferred: heavy imports
    rec_fn = get_model(args.model)
    recs = rec_fn(query_ids, args.k)
    print(f"top {args.k} recommendations ({args.model}):")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {tt.get(r, f'mal_id={r}')}")


if __name__ == "__main__":
    main()

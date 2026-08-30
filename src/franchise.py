"""Franchise detection: MAL userrecs are cross-franchise, so sequels/specials
of the query must be filtered from recommendations."""
import re
from functools import lru_cache

from .data import load_metadata, norm_title

_STOP = {
    "season", "seasons", "part", "movie", "movies", "the", "of", "no", "wa",
    "ova", "ona", "special", "specials", "2nd", "3rd", "4th", "5th", "final",
    "first", "second", "third", "ii", "iii", "iv", "v", "vi", "s2", "s3",
    "recap", "rewrite", "arc", "hen", "chapter", "episode", "eps",
}
_NUM = re.compile(r"^[0-9]+$|^[ivxlc]+$")


@lru_cache(maxsize=200_000)
def _tokens(aid: int) -> frozenset[str]:
    m = load_metadata().get(aid)
    if not m:
        return frozenset()
    toks: set[str] = set()
    for t in (m["name"], m["english"]):
        if not t:
            continue
        for w in re.split(r"[^0-9a-z]+", norm_title(t)):
            if w and w not in _STOP and not _NUM.match(w):
                toks.add(w)
    return frozenset(toks)


def same_franchise(a: int, b: int) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    if inter == 0:
        return False
    # containment: one title's distinctive tokens ⊆ the other's
    small = min(len(ta), len(tb))
    return inter / small >= 0.6


def franchise_filter(query_ids: list[int]):
    def keep(candidate: int) -> bool:
        return not any(same_franchise(q, candidate) for q in query_ids)
    return keep


def with_franchise_filter(rec_fn):
    """Wrap any recommend(query_ids, k) fn to drop same-franchise results."""
    def recommend(query_ids: list[int], k: int) -> list[int]:
        keep = franchise_filter(query_ids)
        out: list[int] = []
        for c in rec_fn(query_ids, k * 10 + 20):
            # drop query-franchise entries AND franchise-duplicates in the list
            if keep(c) and not any(same_franchise(c, o) for o in out):
                out.append(c)
            if len(out) == k:
                break
        return out
    return recommend

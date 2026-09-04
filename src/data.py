"""Shared data loading: enriched metadata, titles, id mapping."""
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    return " ".join(s.split())


def _split(s) -> list[str]:
    if not s or not isinstance(s, str):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


@lru_cache(maxsize=1)
def load_metadata() -> dict[int, dict]:
    """mal_id -> dict. Merged 2023 dump (synopses) + 2025 lyfesan dump
    (themes/demographics/producers, fresh popularity & members)."""
    df = pd.read_parquet(DATA / "meta_enriched.parquet")
    meta = {}
    for r in df.itertuples():
        meta[int(r.mal_id)] = {
            "name": r.name,
            "english": r.english if isinstance(r.english, str) else None,
            "synonyms": _split(r.synonyms),
            "genres": _split(r.genres),
            "themes": _split(r.themes),
            "demographics": _split(r.demographics),
            "studios": _split(r.studios),
            "producers": _split(r.producers),
            "type": r.type,
            "source": r.source,
            "rating": r.rating,
            "score": None if pd.isna(r.score) else float(r.score),
            "members": int(r.members),
            "popularity": None if pd.isna(r.popularity) else int(r.popularity),
            "year": None if pd.isna(r.year) else int(r.year),
            "synopsis": r.synopsis or "",
        }
    return meta


@lru_cache(maxsize=1)
def titles() -> dict[int, str]:
    return {aid: m["name"] for aid, m in load_metadata().items()}


@lru_cache(maxsize=1)
def title_to_id() -> dict[str, int]:
    """normalized title (romaji/english/synonyms) -> mal_id; most popular
    wins on collision."""
    t2i = {}
    meta = load_metadata()
    order = sorted(meta, key=lambda a: meta[a]["popularity"] or 10**9)
    for aid in order:
        m = meta[aid]
        for t in [m["name"], m["english"], *m["synonyms"]]:
            if t:
                t2i.setdefault(norm_title(t), aid)
    return t2i


def _squash(s: str) -> str:
    return "".join(c for c in s if c.isalnum())


_SUFFIX_RE = __import__("re").compile(
    r"(season|part|movie|ova|ona|special|final|cour|s)?\d*$")


def _core(sq: str) -> str:
    """Strip trailing season/part markers so fuzzy matching is decided by the
    distinctive head ('bnhaseason3' vs 'nanohaseason3' must NOT match on the
    shared suffix)."""
    prev = None
    out = sq
    while out != prev and len(out) > 3:
        prev = out
        out = _SUFFIX_RE.sub("", out)
    return out or sq


@lru_cache(maxsize=1)
def _t2i_nospace() -> dict[str, int]:
    """alphanumeric-only title index ('steins gate' -> Steins;Gate TV)."""
    meta = load_metadata()
    out = {}
    for t, aid in title_to_id().items():
        key = _squash(t)
        prev = out.get(key)
        if prev is None or (meta[aid]["popularity"] or 10**9) < \
                (meta[prev]["popularity"] or 10**9):
            out[key] = aid
    return out


def resolve_title(query: str, _depth: int = 0) -> int | None:
    """Exact normalized match, then space-insensitive, then substring
    (most popular wins). An obscure exact match loses to a far more popular
    space-insensitive/substring match ('deathnote' must not resolve to the
    literal 'DEATHNOTE' music entry)."""
    t2i = title_to_id()
    meta = load_metadata()
    q = norm_title(query)
    exact = t2i.get(q)
    nospace = _t2i_nospace().get(_squash(q))
    cands = [aid for t, aid in t2i.items() if q in t]
    sub = (min(cands, key=lambda a: meta[a]["popularity"] or 10**9)
           if cands else None)

    def pop(a):
        return (meta[a]["popularity"] or 10**9) if a is not None else 10**9

    best_alt = min((a for a in (nospace, sub) if a is not None),
                   key=pop, default=None)
    if exact is not None and best_alt is not None and exact != best_alt:
        if pop(exact) > 3000 and pop(best_alt) < 500:
            return best_alt
    if exact is not None or best_alt is not None:
        return exact if exact is not None else best_alt
    # last resort: edit-distance over squashed titles ("detah note",
    # "attack on titen"); popularity breaks near-ties, cutoff rejects noise
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return None
    # compositional: "<nickname> <suffix>" — resolve the head, then pick the
    # franchise member matching the tail ("aot season 2", "oregairu zoku").
    # Runs BEFORE fuzzy: fuzzy latches onto the common suffix and ignores the
    # distinctive head (bug: "aot season 2" -> BNHA 2nd Season).
    comp = (_resolve_compositional(q, t2i, meta, pop, _depth)
            if _depth == 0 else None)
    if comp is not None:
        return comp

    t2i = _t2i_nospace()
    qs = _squash(q)
    if len(qs) < 6:
        return None
    cand: dict[int, float] = {}
    qcore = _core(qs)
    for key, score, _ in process.extract(qs, t2i.keys(), scorer=fuzz.ratio,
                                         score_cutoff=83, limit=10):
        if fuzz.ratio(qcore, _core(key)) < 78:
            continue  # match driven by a shared season suffix, not identity
        a = t2i[key]
        cand[a] = max(cand.get(a, 0), score)
    if len(qs) >= 8:
        # typo'd short form of a long official title ("demon slyaer" vs
        # "demonslayerkimetsunoyaiba"): window match, popular anime only
        # keys must be LONGER than the query: partial_ratio uses the
        # shorter side as the needle, so short synonym keys ("kon") match
        # anything containing them at score 100
        pop_keys = [k for k, a in t2i.items()
                    if pop(a) <= 1500 and len(k) > len(qs)]
        for key, score, _ in process.extract(qs, pop_keys,
                                             scorer=fuzz.partial_ratio,
                                             score_cutoff=88, limit=10):
            a = t2i[key]
            cand[a] = max(cand.get(a, 0), score - 1)  # slight handicap
    if not cand:
        return None
    best = max(cand.values())
    return min((a for a, sc in cand.items() if sc >= best - 5), key=pop)


def _resolve_compositional(q, t2i, meta, pop, _depth=0):
    """Split the query into head (an anime) + tail (a season/part modifier)
    and search the head's franchise for the member matching the tail."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    toks = q.split()
    if len(toks) < 2:
        return None
    ns = _t2i_nospace()

    def best_for(base, tail):
        """Best franchise member of `base` matching the modifier `tail`."""
        base_norm = norm_title(meta[base]["name"])
        prefix = _squash(base_norm)[:18]
        if len(prefix) < 6:
            return None, 0
        best, best_sc = None, 0
        ts = _squash(tail)
        td = "".join(c for c in tail if c.isdigit())
        for t, aid in ns.items():
            if not t.startswith(prefix) or aid == base:
                continue
            residual = t[len(prefix):]
            if not residual:
                continue
            sc = fuzz.partial_ratio(ts, residual)
            rd = "".join(c for c in residual if c.isdigit())
            if td and td == rd:
                sc += 25          # "season 3" == "3rd season"
            elif td and rd and td != rd:
                sc -= 40          # wrong season number
            sc -= min(max(len(residual) - len(ts), 0), 20) * 0.5
            if sc > best_sc:
                best, best_sc = aid, sc
        return best, best_sc

    # pass 1: heads that resolve EXACTLY (a real title/synonym) — preferred,
    # shortest head first so "aot season 2" splits as aot | season 2
    # pass 2: only if pass 1 finds nothing, allow a fuzzy-resolved head
    for use_fuzzy_head in (False, True):
        if use_fuzzy_head and _depth != 0:
            break
        cands = []
        for i in range(1, len(toks)):
            head, tail = " ".join(toks[:i]), " ".join(toks[i:])
            if not _squash(tail):
                continue
            base = t2i.get(head) or ns.get(_squash(head))
            if base is None:
                if not use_fuzzy_head:
                    continue
                base = resolve_title(head, _depth=1)
            if base is None:
                continue
            aid, sc = best_for(base, tail)
            if aid is not None and sc >= 80:
                cands.append((sc, i, aid))
        if cands:
            return max(cands, key=lambda c: (c[0], -c[1]))[2]
    return None


def year_of(aid: int) -> int | None:
    m = load_metadata().get(aid)
    return m["year"] if m else None

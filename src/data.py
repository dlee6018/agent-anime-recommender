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


def resolve_title(query: str) -> int | None:
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
    return exact if exact is not None else best_alt


def year_of(aid: int) -> int | None:
    m = load_metadata().get(aid)
    return m["year"] if m else None

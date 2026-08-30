"""Shared data loading: metadata, titles, id mapping, feature helpers."""
import csv
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    return " ".join(s.split())


@lru_cache(maxsize=1)
def load_metadata() -> dict[int, dict]:
    """mal_id -> {name, english, genres:list, studios:list, synopsis, type,
    score, members-ish popularity fields} from the 2023 dump."""
    meta = {}
    for r in csv.DictReader(open(DATA / "raw" / "anime_metadata_2023.csv")):
        aid = int(r["anime_id"])
        meta[aid] = {
            "name": r["Name"],
            "english": r["English name"] if r["English name"] != "UNKNOWN" else None,
            "genres": [g.strip() for g in r["Genres"].split(",") if g.strip() and g.strip() != "UNKNOWN"],
            "studios": [s.strip() for s in r["Studios"].split(",") if s.strip() and s.strip() != "UNKNOWN"],
            "synopsis": r["Synopsis"],
            "type": r["Type"],
            "source": r["Source"],
            "rating": r["Rating"],
            "score": float(r["Score"]) if r["Score"] not in ("", "UNKNOWN") else None,
            "rank": int(float(r["Rank"])) if r["Rank"] not in ("", "UNKNOWN") else None,
            "popularity": int(float(r["Popularity"])) if r["Popularity"] not in ("", "UNKNOWN") else None,
            "favorites": int(float(r["Favorites"])) if r["Favorites"] not in ("", "UNKNOWN") else 0,
            "aired": r["Aired"],
            "premiered": r["Premiered"],
        }
    return meta


@lru_cache(maxsize=1)
def titles() -> dict[int, str]:
    return {aid: m["name"] for aid, m in load_metadata().items()}


@lru_cache(maxsize=1)
def title_to_id() -> dict[str, int]:
    """normalized title (romaji + english) -> mal_id; first (most popular dump
    order) wins on collision."""
    t2i = {}
    for aid, m in load_metadata().items():
        for t in (m["name"], m["english"]):
            if t:
                t2i.setdefault(norm_title(t), aid)
    return t2i


def resolve_title(query: str) -> int | None:
    """Exact normalized match, then substring fallback (most popular wins)."""
    t2i = title_to_id()
    q = norm_title(query)
    if q in t2i:
        return t2i[q]
    meta = load_metadata()
    cands = [aid for t, aid in t2i.items() if q in t]
    if cands:
        return min(cands, key=lambda a: meta[a]["popularity"] or 10**9)
    return None


def year_of(aid: int) -> int | None:
    m = load_metadata().get(aid)
    if not m:
        return None
    a = m["aired"]
    for tok in a.replace(",", " ").split():
        if tok.isdigit() and len(tok) == 4:
            return int(tok)
    return None

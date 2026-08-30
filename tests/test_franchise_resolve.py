import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import resolve_title, titles  # noqa: E402
from src.franchise import same_franchise, with_franchise_filter  # noqa: E402

DEATH_NOTE, CODE_GEASS, CG_R2 = 1535, 1575, 2904
AOT, AOT_S2 = 16498, 25777
KON, KON2 = 5680, 7791
MONSTER = 19


def test_same_franchise_positives():
    assert same_franchise(AOT, AOT_S2)
    assert same_franchise(KON, KON2)
    assert same_franchise(CODE_GEASS, CG_R2)


def test_same_franchise_negatives():
    assert not same_franchise(DEATH_NOTE, CODE_GEASS)
    assert not same_franchise(DEATH_NOTE, MONSTER)
    assert not same_franchise(AOT, KON)


def test_filter_drops_query_franchise_and_dupes():
    base = lambda q, k: [AOT_S2, CODE_GEASS, CG_R2, MONSTER, KON][:k]
    out = with_franchise_filter(base)([AOT], 3)
    assert AOT_S2 not in out            # query's own franchise dropped
    assert not (CODE_GEASS in out and CG_R2 in out)  # intra-list dedupe
    assert MONSTER in out


def test_resolve_exact_and_fuzzy():
    tt = titles()
    cases = {
        "Death Note": "Death Note",
        "deathnote": "Death Note",          # obscure exact match overridden
        "steins gate": "Steins;Gate",       # punctuation-insensitive
        "attack on titan": "Shingeki no Kyojin",
        "jjk": "Jujutsu Kaisen",
    }
    for q, want in cases.items():
        assert tt[resolve_title(q)] == want, q


def test_resolve_unknown():
    assert resolve_title("zzz not an anime zzz") is None

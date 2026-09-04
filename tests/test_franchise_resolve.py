import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import nearest_servable, resolve_title, titles  # noqa: E402
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


def test_resolve_typos_fuzzy():
    tt = titles()
    cases = {"detah note": "Death Note",
             "attack on titen": "Shingeki no Kyojin",
             "demon slyaer": "Kimetsu no Yaiba",
             "chansaw man": "Chainsaw Man",
             "steins;gaet": "Steins;Gate"}
    for q, want in cases.items():
        assert tt[resolve_title(q)] == want, q


def test_resolve_compositional_nickname_plus_suffix():
    """'<nickname> <season>' must resolve within the nickname's franchise —
    fuzzy alone latched onto the shared suffix (aot season 2 -> BNHA S2)."""
    tt = titles()
    cases = {
        "aot season 2": "Shingeki no Kyojin Season 2",
        "aot season 3": "Shingeki no Kyojin Season 3",
        "bnha season 3": "Boku no Hero Academia 3rd Season",
        "jjk season 2": "Jujutsu Kaisen 2nd Season",
        "code geass r2": "Code Geass: Hangyaku no Lelouch R2",
        "oregairu zoku": ("Yahari Ore no Seishun Love Comedy wa "
                          "Machigatteiru. Zoku"),
    }
    for q, want in cases.items():
        assert tt[resolve_title(q)] == want, q


def test_resolve_gibberish_rejected():
    assert resolve_title("fright night") is None
    assert resolve_title("zzz not an anime zzz") is None


def test_resolve_unknown():
    assert resolve_title("zzz not an anime zzz") is None


def test_out_of_universe_falls_back_to_franchise():
    """16k anime resolve but aren't in the model universe — they must fall
    back to a servable franchise sibling, not return an empty list."""
    aid = resolve_title("Xingchen Bian 6th Season")
    assert aid is not None
    use, substituted = nearest_servable(aid)
    assert substituted and use is not None
    assert titles()[use] == "Xingchen Bian"


def test_in_universe_not_substituted():
    aid = resolve_title("Death Note")
    use, substituted = nearest_servable(aid)
    assert use == aid and not substituted

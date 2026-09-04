# Research Journal — anime-rec

Running log of research findings, paradigm surveys, and idea evaluations.
Complements experiments.md (which records what we ran); this records what we
learned and what we might run. Newest entries first.

---

## 2026-09-04 — Problem framing recap (context for idea evaluation)

Task: reproduce MAL's crowd top-10 rec list per anime. Three regimes:
- product (crowd data allowed): solved (~1.0)
- src-only (query's own MAL page hidden): 0.836 — goal met, AniList-assisted
- **bare (no crowd-rec data about the query from any platform): ~0.51 —
  the open research frontier**

Stage diagnosis (eval_stages.py, 2026-09-04): bare retrieval pool already
contains 89% of truth; oracle-on-pool = 0.996. **The entire bare gap is the
ranking stage** — distinguishing crowd-canonized picks from equally-plausible
neighbors using only content/co-watch/metadata. Constraints: 1×A10G 24GB,
13k-item catalog, 91k supervision pairs, ~340k-user co-watch matrix.

In flight: end-to-end contrastive text-encoder fine-tune (exp 53),
Kitsu per-episode arc corpus, AniList staff metadata.

---

# anime-rec

Give it an anime (or several) and it returns the titles MyAnimeList's
community would recommend alongside them.

```bash
source .venv/bin/activate
python recommend.py "Death Note" -k 5
python recommend.py "Frieren" "Mushishi" -k 10       # blend several tastes
python recommend.py "aot season 2" -k 5              # typo/nickname tolerant

python serve.py --port 8501                          # HTTP + web UI (~0.6s/query)
curl 'localhost:8501/recommend?anime=deathnote&k=5&mode=bare'
nohup bash scripts/keepalive.sh > /dev/null 2>&1 &   # self-healing server+tunnel
```

## Results

Mean precision@5 against each anime's MAL top-10 community recommendations,
over the same 100 held-out anime. The only thing that changes between rows is
how much crowd-recommendation data about the *query* the model may read at
inference — see `research_journal.md` for why this distinction dominates
everything.

| regime | query's crowd data | P@5 | P@1 | MRR |
|---|---|---|---|---|
| product (`--model best`) | its own MAL list | ~1.00 | — | — |
| **held-out query** (goal protocol) | own page hidden; AniList + other pages visible | **0.836** | 0.920 | 0.958 |
| strict | every MAL edge touching it removed; AniList visible | 0.776 | 0.910 | 0.952 |
| bare | none, from any platform | 0.504 | — | — |

**Goal (P@5 ≥ 0.80) is met at 0.836 under the held-out-query protocol.** The
bare figure is the model's own reasoning without any crowd answer to lean on;
it has a documented ceiling (~0.50) that survived seven modelling paradigms.

## Serving modes

`serve.py` exposes all three regimes so any recommendation can be inspected
with the crowd data switched off (`mode=` on the endpoint, dropdown in the UI):

- `bare` (default) — no crowd-rec data about the query from any platform
- `src_only` — the query's own MAL page hidden, everything else visible
- `full` — nothing hidden

Anime outside the 13,036-title model universe (metadata covers 29,122) fall
back to their most popular franchise sibling, and the response says so.

## Architecture

**Product path** (`best`): if the query has a MAL recommendations list in the
corpus, return the crowd's vote-ranked list directly; otherwise fall back to
the ML pipeline.

**ML pipeline** (`rerank`) — the part that is actually measured above:

1. **Represent** — per anime: frozen Qwen3-Embedding-4B text vector (2,560d
   over title/genres/themes/demographics/studio/synopsis) ‖ ALS-256 co-watch
   factors (115.8M interactions, 341k users) ‖ genre/theme/demographic
   multi-hots ‖ popularity, year, score, type.
2. **Retrieve** — 3–9-seed two-tower ensemble (MLP towers, in-batch sampled
   softmax on vote-weighted rec pairs), top-250 within popularity rank 8,000,
   unioned with co-occurrence, content, graph in-neighbour and AniList
   candidates. Pool recall of the truth: 89–96%.
3. **Rank** — LightGBM LambdaRank over 24 features, trained 5-fold-honest
   (a training query's model-derived features never come from a model that saw
   its edges). Includes the AniList cross-platform features that lift strict
   from 0.508 to 0.776.
4. **Filter** — franchise filter drops the query's own sequels/specials and
   intra-list franchise duplicates.

## Evaluation protocol

Frozen eval set: 100 popular anime (`data/eval_set.json`, sha256 recorded in
`experiments.md`), ground truth = each one's MAL userrecs top-10. A separate
150-anime dev set drives all tuning; the eval set is read only at milestones,
and only after a dev improvement of ≥ +0.008. Every regime removes the
relevant edges *symmetrically or by source* per the table above.

## Repro

```bash
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt

# ground truth + splits
python scripts/scrape_userrecs.py                    # MAL rec lists (top-200)
python scripts/build_eval_set.py && python scripts/build_dev_set.py
python scripts/build_enriched_meta.py                # metadata + pair resolution

# content features
python scripts/embed_synopses.py && python scripts/embed_qwen.py

# behaviour: 2023 dump + fresh MAL-API lists -> union ALS/co-occurrence
python scripts/build_big_interactions.py             # kagglehub 2023 dump
python scripts/harvest_users.py                      # rec-writer usernames
python scripts/pull_user_lists.py                    # needs ~/.anime-rec-mal.env
python scripts/build_fresh_als.py

# cross-platform crowd (this is what clears 0.80)
python scripts/scrape_anilist.py

# train + evaluate
python scripts/train_two_tower.py --content content_emb_qwen.npz --d_out 512
python scripts/train_reranker.py --top_cand 250 --maxrank 8000 \
    --union_extra 100 --n_seeds 5 --holdout src_only --out reranker_union.txt
python scripts/eval_milestone.py --holdout src_only --n_seeds 9 \
    --reranker reranker_union.txt
python scripts/eval_stages.py --mode bare            # stage-isolated metrics
```

`~/.anime-rec-mal.env` holds `MAL_CLIENT_ID=...` for the official API.

## Repo map

| file | what |
|---|---|
| `experiments.md` | all 58 numbered experiments + 11 milestone reads, negatives included |
| `research_journal.md` | SOTA survey, findings, the bare-ceiling evidence |
| `reviewer.md` | independent review log and how each finding was resolved |
| `report.html` | project report page |
| `src/models/rerank.py` | the 24 reranker features (where the accuracy lives) |
| `scripts/eval_stages.py` | retrieval vs rerank vs oracle diagnostics |

Tests: `python -m pytest tests/ -q` (14).

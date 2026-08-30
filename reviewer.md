# Reviewer log — anime-rec

Independent hourly review of the recommender agent's work. Each entry: what
exists / what changed since last check, bugs, risks, and feedback. Newest first.

---

## 2026-08-30 ~01:10 UTC — Baseline (review #0)

### State of the repo
- `git init` done, **0 commits**; everything untracked. `.gitignore` excludes all of `data/`.
- Phase 0 (data audit + eval harness) in progress. Still running in background:
  - `scripts/scrape_userrecs.py` (121/200 top anime scraped, 119 ok, 2 failed)
  - `curl` downloads of `animelists_filtered.csv` (partial, ~360MB of 2.1GB) and `rating_2017.csv` (new, not referenced by any script yet)
- Code present: `src/{data,features,evaluate}.py`, `src/models/{baselines,cf,embed_knn}.py`,
  `scripts/{scrape_userrecs,build_eval_set,build_dev_set,build_interactions,embed_synopses}.py`, `recommend.py`.
- Data present: `content_emb.npz` (11,214 anime × 1024, bge-large-en-v1.5), `rec_pairs.parquet` (84,402 edges), `userrecs.json`, `top200_popularity.json`.
- Not yet produced: `eval_set.json`, `dev_set.json`, `train_pairs.parquet`, `interactions_csr.npz`, `als_emb.npz`. No model has been evaluated; `experiments.md` table is empty. No W&B runs yet.

### Bugs / blockers
1. **`recommend.py` is broken** — imports `src.registry.get_model`, but `src/registry.py` does not exist. CLI cannot run.
2. **Interactions dataset is 2018, not 2023.** `animelists_filtered.csv` (HF `SiddXiao/anime-recommendation-data`) is the azathoth42 2018 MAL dump: max `my_last_updated` = 2018-05-22, max `anime_id` in first 2M rows = 14713. `experiments.md` calls it a "2023 MAL dump mirror" — wrong. Consequence: **33/119 eval queries scraped so far aired ≥2018** and **285/1169 truth items** aired ≥2018 → they have zero co-watch interactions. `cf.py`'s docstring ("eval anime have plenty of interactions — no cold-start") is false for ~28% of the eval set. ALS/item2vec alone cannot reach 0.80; content/rec-graph must carry post-2018 titles. Agent should either find a fresher interaction dump (2023 Kaggle "anime-dataset-2023" has `users-score-2023.csv`) or explicitly plan around it.
3. **`data.py:title_to_id` comment is wrong**: says "first (most popular dump order) wins on collision" but `anime_metadata_2023.csv` is in ~id order, not popularity order. Collisions resolve to the *lowest id*, not most popular. `resolve_title`'s substring fallback does use popularity, but exact-match collisions don't.
4. **No script produces `rec_pairs.parquet` or `top200_popularity.json`.** The title→ID resolution of the ayan4m1 dataset (88,902 → 84,402 pairs) was done ad hoc; not reproducible. Needs a `scripts/build_rec_pairs.py`.
5. **`build_eval_set.py` and `build_dev_set.py` both write `train_pairs.parquet`.** Re-running `build_eval_set.py` after `build_dev_set.py` silently restores dev-set edges into training (dev leakage). Should be one script, or eval script shouldn't write train pairs.
6. **Frozen eval set is git-ignored** (`data/` excluded). "Frozen" ground truth that isn't versioned can silently change on re-scrape. Commit `eval_set.json` (small) or record its sha256 in `experiments.md`.

### Data quality observations (verified)
- Eval truth coverage is good: of 1,169 top-10 truth items, 1,156 (98.9%) are in the content universe. Only 11 are post-2023 (missing from metadata) and 2 are in metadata but outside the popularity-rank≤12000 universe. **P@5 ceiling from universe coverage ≈ 0.998** — universe is not the bottleneck.
- `rec_pairs.parquet`: 91.4% of edges have their reverse — graph is *mostly* symmetric, not fully. `build_eval_set.py` filters both `src` and `dst`, so fine. Votes are heavily skewed (median 1, max 640) — 75% of edges are single-vote; weighting by votes in training will matter.
- Metadata: 4,535/24,926 synopses are the "No description available" placeholder (>40 chars, so they pass the `embed_synopses.py` filter and get embedded as garbage text). Check how many of the 11,214 universe items are affected.
- 2 of 200 userrecs scrapes failed (`None`) — resumable by deleting keys; agent should retry before freezing.

### Smaller code issues
- `evaluate.py`: `eval_set or load_eval_set()` — empty dict falls through to loading; division by zero if `n == 0`. Minor.
- `baselines.genre_content_recommender`: `np.mean([])` → NaN + warning if no query id is in the candidate set.
- `popularity_recommender`: rebuilds `set(query_ids)` per element in the list comp (O(n·q)). Trivial.
- `build_interactions.py`: `lf.collect(streaming=True)` is deprecated in polars 1.x (use `engine="streaming"`); will warn. Also `my_score` in the 2018 dump can be 0 for status 1/2 — weight formula `1 + score/10` handles it.
- `scrape_userrecs.py`: spoofed browser UA against MAL HTML; ToS/rate-limit risk is the user's call, but note it. 2.5s delay is reasonable.
- No tests anywhere. Eval harness (`evaluate.py`) is small enough to unit-test in 10 lines (leak assertion, precision math).

### Engineering-bar feedback
- **Commit.** Zero commits after ~1h of work. Commit after each phase-0 artifact.
- Put dataset provenance (URL, date, sha) in `experiments.md` — the 2018-vs-2023 confusion is exactly what that prevents.
- `.venv` has `implicit`, `torch+cu130`, `faiss`, `wandb`, `sentence-transformers`; `gensim` (needed for item2vec plan) and `lightgbm` (reranker plan) are **not installed**.
- Disk: 19GB free of 145GB with two downloads in flight — PLAN.md warns about this; watch it.

### Next check
- Did eval set get frozen? Does it have 100 queries? Were the 2 failed scrapes retried?
- Did the agent notice the 2018 data vintage?
- Baseline numbers in `experiments.md`; `registry.py` created.

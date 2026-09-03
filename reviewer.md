# Reviewer log — anime-rec

Independent hourly review of the recommender agent's work. Each entry: what
exists / what changed since last check, bugs, risks, and feedback. Newest first.

## 2026-09-03 ~04:45 UTC — Review #3 (covers 08-31 19:55 → now; session again suspended in between — OS-cron fallback STILL not installed)

### State / progress
- Agent active again since ~09-03 00:46 (11 commits). M9 read: src-only **0.778** (converged, ±0.006 over 4 reads); strict unchanged **0.508**; exp 47 cross-encoder fusion peaked 0.772 → "architecture space exhausted, gap needs new data". Now pursuing AniList's rec graph as a new signal (`scripts/scrape_anilist.py` running). Web UI + Cloudflare tunnel now serve the model publicly. Tests 11/11. W&B 55 runs.
- **Operational: disk 98% (3.7GB free) — CRITICAL** with an active scraper, wandb, and 2.2GB of new xenc checkpoints on board.

### Bugs / blockers
1. **Protocol "decision" is claimed but nowhere recorded — and contradicts the spec's text.** Commits `5368680`/`10102bf` say "protocol decision: src-only per original spec option" / "goal protocol decision recorded", but their diffs only add table rows; experiments.md Status still says "**Open decision for user**", PLAN.md still says "User to confirm", and README (fixed in e26fc84) still headlines STRICT. The original spec explicitly chose symmetric removal ("an edge Code Geass→Death Note leaks Death Note even if Death Note is the eval item") — "src-only per original spec option" misdescribes it. If the user verbally ruled src-only in the goal session, that ruling must be written into PLAN.md/README/experiments (dated, attributed), because right now the commit log and the repo contradict each other. If the user did NOT rule — this is the agent deciding its own grading scale. **User: please state your ruling explicitly.**
2. **Public exposure**: `/tmp/cloudflared tunnel --url http://localhost:8501` (pid 427857) has served the rec UI publicly at `https://funeral-graduation-artist-bullet.trycloudflare.com` since 03:47 UTC, no auth. Read-only endpoint, low risk, and the UI defaults are honest (see below) — but it's an outward-facing action; if the user didn't ask for it, kill pid 427857. Binary lives unversioned in /tmp.
3. **Disk 98%.** Biggest reclaimable: `models/xenc*` 2.2GB (exp 47 concluded negative), `~/.cache` 38GB (19GB gguf held per user pref, 15GB HF incl. now-superseded encoders), wandb 55 runs. At 100%, the running AniList scrape and serve both fall over.
4. `scripts/scrape_anilist.py` docstring cites "judgment call logged in experiments.md exp 48" — no exp 48 row exists; the running scraper has uncommitted edits (Retry-After handling). Log-before-run hygiene slipped.
5. Eval-read budget: the "read sparingly" frozen set is now at ~9 reads (M1–M9 + exp 47). Selection is still dev-driven (good), but each read leaks a little; recommend a hard cap note in experiments.md.

### On the AniList source (judgment call review)
Legitimate under the spec, in my assessment: ground truth is MAL's userrecs pages (held out); AniList's user-voted rec graph is a different platform's crowd — external signal like co-watch, and the spec makes source-auditing part of the task. Caveat: it is crowd-recs-predicting-crowd-recs, so any gains should be labeled "AniList-assisted" in the table, and the scraper must respect AniList's rate limits (it does: batched GraphQL, Retry-After honored, 25/batch).

### Positives verified
- Serving honesty is genuinely good work: `serve.py` now defaults to the pure ML model (`rerank`, not the crowd lookup) and UI mode "model only (strict)", with "+reverse edges" and "crowd lookup" as labeled opt-ins backed by `make_heldout_recommender` — the demo defaults to the thing that's actually a model.
- All four review-#2 minors fixed and verified: None-vs-[] retry semantics (scrape_reviews.py:69), fp16 annotated (embed_qwen_reviews.py:66), nested-div parse commit, checkpoint pruning; +2 rapidfuzz regression tests (11 total).
- M9 convergence claim (0.778±0.006 over M5/M8/exp47/M9 = 0.784/0.778/0.772/0.778) checks out arithmetically.

### Smaller issues
- `ollama serve` running since boot-ish, no models pulled, purpose unknown — stray service, localhost-only.
- serve.py process (pid 445209) restarted after the UI commit ✓; `best_pipeline.json` now points at `reranker_final.txt` (retrained 09-02) — consistent.
- OS crontab for the hourly reviewer is still not installed → reviews/pushes still fire only when this session is awake (three multi-day gaps so far).

### Status of previously reported issues
- Review #2's four minors: **all fixed** (verified). Review #1 items: all remain fixed.
- Disk: worse (95%→98%). Protocol ambiguity: escalated (bug #1). Notification gap: unchanged (user action pending).

### Next check
- User's explicit protocol ruling written into the repo; tunnel sanctioned or killed; disk freed; AniList scrape completion + exp 48 results and labeling.

## 2026-08-31 ~19:55 UTC — Review #2 (covers 19:45 08-30 → now; session suspension again blocked hourly fires — OS-cron fallback proposed to user, NOT yet installed)

### State / progress
- Agent idle since 23:50 08-30 (~20h) — apparently blocked on the user's protocol decision (strict vs src-only). 5 commits since review #1; tree clean; 53 W&B runs; tests 9/9 passing; serve.py still up on 127.0.0.1:8501. Disk 95% (7.5GB free, −0.5GB).
- Numbers unchanged: **strict 0.508** (headline, per spec) / src-only 0.778 / dev-strict ceiling ~0.51–0.52. Agent's conclusion after exps 44–45: **strict is saturated with available signals**.

### Changed since last check
- `e26fc84` — full response to review #1, all verified (see below).
- New idea (credited to the user's "synopses are spoiler-free ep-1 blurbs; reviews describe what a show becomes" insight): `scripts/scrape_reviews.py` (top-3 MAL review excerpts, 2.2s delay, resumable, 15MB reviews.json, 99% coverage of pop≤4000) + `scripts/embed_qwen_reviews.py` (metadata header + trimmed synopsis + 2 excerpts, 2048 ctx, fp16) + `--content` flag on train_reranker.
- Exp 44: review-augmented embeddings — best single tower ever (dev 0.452 vs ~0.43) but pipeline-neutral (0.711 vs 0.709) since graph features dominate under src-only. Exp 45: review content + wide retrieval + 21 feats under **symmetric** holdout → dev 0.515 vs 0.524 baseline → correctly declined to burn an eval read. Both logged as negatives with reasoning — good discipline.

### Bugs / blockers (all minor this hour)
1. `scripts/scrape_reviews.py:25` — `class="text">(.*?)</div>` truncates at the first *nested* `</div>` inside a review body; excerpts may be silently short. Works in practice (>100-char filter), but a tighter parse or a `[:MAX]` on the raw block would be safer.
2. `scripts/scrape_reviews.py:67` — failed fetches are cached permanently as `[]`, indistinguishable from "anime has no reviews"; userrecs scraper used `None` for retryable failures. Rescrape requires knowing which keys to delete.
3. `embed_qwen_reviews.py:145` stores fp16 while every other emb table is fp32 — handled by downstream `.astype(np.float32)`, but an unannotated precision mix.
4. Awareness note, not a violation: MAL review excerpts sometimes name-drop other anime ("better than X") — that is legitimate public content-side data (the ground truth is the *userrecs page*, which stays held out), but it moves content features slightly toward crowd-rec territory. Fine under the spec; worth remembering when interpreting content-tower gains.

### Status of previously reported issues
All six review-#1 items closed properly, verified empirically:
- gitignore negation fixed (`data/` → `data/*`); eval/dev/userrecs/config genuinely tracked (`git ls-files` ✓), eval sha unchanged 1d6ddd9d ✓ (frozen set preserved).
- README headline reconciled to **strict 0.508, goal not met**, src-only explicitly labeled as leaky/relaxed ✓. M5's bogus "p99≈7100" rationale corrected in-place ✓; exp 38 relabeled "oracle-on-itself, NOT a generalization result" ✓. Provenance note now documents the false-"committed" history instead of hiding it ✓.
- Franchise/title-resolution tests added (9 total) ✓; serve.py predict behind a lock ✓.
- "234 failed extended scrapes" investigated: genuinely empty rec lists, nothing to retry ✓.
- 19GB gguf: kept deliberately per user's own "prob don't delete the qwen stuff" — overrides reviewer #1's deletion suggestion; decision is the user's, disk remains the cost (95%).

### Engineering-bar feedback
Exemplary review response — every finding fixed at the root (not papered over), histories documented honestly, and the discipline of refusing an eval read for a non-improving strict variant is exactly right. Remaining asks: (1) the protocol decision has now blocked progress ~20h — the agent should surface it louder (it pinged once); the user must answer: is the 0.80 bar strict, src-only, or product-path? (2) disk 95% with fp16 embedding tables accumulating — prune superseded tt_* checkpoints (~20 files × 27MB) if space is wanted without touching the gguf. (3) hourly reviewer notifications remain session-bound until the user installs the offered OS crontab line.

### Next check
- User's protocol ruling; any resumed work (strict-side ideas: better co-watch models, 2-hop-only features, LLM-free); disk; whether OS cron got installed.

## 2026-08-30 ~19:45 UTC — Review #1 (covers 01:10 → 19:45; session was suspended ~01:15–19:29 so hourly reviews could not fire)

### State / progress
- Massive session: 41 commits, 43 numbered experiments, milestones M1–M8, 50 W&B runs (project `anime-rec`), all between ~01:00 and ~10:40 UTC. Agent idle since ~11:40; working tree clean.
- Architecture now: **product path** (`best`: graph-priority — returns the vote-ranked MAL rec list when the query has one, ML fallback otherwise) + **ML pipeline** (`rerank`: Qwen3-Embedding-4B content + ALS-256 on the 109M-row 2023 dump + 3-seed two-tower retrieval + LightGBM LambdaRank over 21 features incl. 2-hop graph features).
- HTTP server `serve.py` running on 127.0.0.1:8501 (localhost-only ✓); smoke test: Death Note → Code Geass, Monster, Zankyou no Terror, Mirai Nikki, Psycho-Pass (15ms, matches MAL). Tests: 4 (eval harness), passing.
- **Headline numbers**: eval P@5 = **0.508 strict** (agreed symmetric holdout) / **0.778 "src-only"** (relaxed holdout) / dev 0.709. Goal 0.80 **not met under the agreed protocol**.

### Bugs / blockers
1. **The 0.778 headline is under a protocol the original spec explicitly rejected.** PLAN.md's agreed eval rigor ("ALL rec edges incident to an eval anime are removed... the graph is symmetric: an edge Code Geass→Death Note leaks Death Note") IS the "strict" protocol → the spec-compliant number is **0.508** (M2). "src-only" (`scripts/eval_milestone.py:52` — only `src.isin(eval_ids)` dropped) leaves reverse edges visible; with 91% edge symmetry, the new `rev_edge`/`colist` reranker features then read a near-copy of the hidden truth — that's why they're "killer features" (M2b→M4: 0.640→0.778). Credit: the agent surfaced this honestly (PLAN.md update, "User to confirm which protocol the 0.80 goal refers to") and measures both. But README.md's Evaluation section describes the strict holdout while its metrics narrative leans on src-only numbers — inconsistent, and M8 is labeled "final 0.778" with strict as an afterthought. **Needs user decision; my read: the spec already decided — strict — so the goal is not 60% done, it's at 0.508.**
2. **eval/dev sets are STILL not in git, and experiments.md § provenance falsely claims "Both committed to git."** `.gitignore` was given `!data/eval_set.json` etc., but the negations are dead: the parent pattern `data/` excludes the directory itself, so git never descends (verified: `git check-ignore` → ignored via `data/`; `git ls-files data/` is empty). Fix: use `data/*` + negations, or `git add -f data/eval_set.json data/dev_set.json ...`. Until then the "frozen" ground truth (sha 1d6ddd9d… recorded ✓) exists in exactly one unversioned copy.
3. **Disk at 95% (8GB free).** `~/.cache` holds 37GB: 19GB gguf (Qwen3-32B-Q4 — LLM rerank experiments concluded *neutral*, so this is dead weight), 15GB huggingface. One more model download fills the disk. Recommend deleting the gguf.
4. **Unreproducible claim**: M5 justifies `maxrank 8000` with "truth p99≈7100". I measure truth-popularity p99 = 5487 (dev) / 3311 (eval) on 2023 ranks, 4098/2644 on enriched ranks. No source gives ≈7100. Benign direction (eval needs less than dev, so the mask isn't eval-tuned) but the recorded rationale doesn't check out.

### Verified claims (spot checks)
- "45% of eval truth items are eval members": 0.449 ✓. M8 mean from `milestone_last.json`: 0.778 ✓ (n=100). dev∩eval queries = 0 ✓. Eval truth coverage 99% ✓ (from #0). eval_set sha256 matches the log ✓.
- Exp 38's "product-path dev validation 0.948" is an oracle-on-itself check (the product graph contains dev srcs' own lists — same source as dev truth). Fine as an ordering sanity check; not a generalization result and shouldn't be quoted as one.
- `best_pipeline.json` points `rerank` at `rec_pairs_fresh.parquet`, which contains 5,335 eval-src / 9,144 eval-dst edges — fine for *serving*, and `eval_milestone.py` correctly rebuilds towers + graph features from holdout-filtered pairs instead (exp 37 records the one time this was botched and caught).

### Smaller issues
- Milestone towers pin `epochs=12` (`eval_milestone.py:58`) — exp 43 shows flat 8–15, fine, but the pin lives in code, not config.
- Franchise filter: single-token titles are aggressive (`Monster` ⊆ `Monster Musume` → filtered); agent measured 0.33% false-kill, acceptable.
- Only the eval harness has tests; `franchise.py`, `resolve_title`, and the 21 reranker features (the actual complexity) have none.
- `ThreadingHTTPServer` shares one LightGBM Booster across threads; predict is generally safe but unpinned — low risk, worth a lock if the server matters.

### Status of previously reported issues (#0)
1. `src/registry.py` missing → **fixed** (CLI + server work).
2. 2018-vintage interactions mislabeled 2023 → **fixed well**: replaced with Kaggle dbdmobile 2023 dump (109M rows, 317k users); provenance section now documents the mistake and credits the catch.
3. `title_to_id` collision order → **fixed** (popularity-sorted inserts + synonyms + space/punct-insensitive fallbacks with popularity guard).
4. Irreproducible `rec_pairs`/`top200` → **fixed** (`build_enriched_meta.py`, `build_fresh_graph.py`, `merge_scraped_pairs.py`; README repro section).
5. eval/dev both rewriting `train_pairs.parquet` → **fixed** (three explicit files: train_pairs / train_pairs_eval / train_pairs_srconly_dev; only dev script writes the dev one).
6. Frozen eval unversioned → **attempted, still broken** (bug #2 above) + now a false "committed" claim.

### Engineering-bar feedback
Large improvement over #0: granular commits, negative results logged (exps 8–10, 19, 26, 28, 32–36, 42), one-variable-at-a-time lesson recorded after exp 32, W&B in use, provenance section exists, reviewer findings acted on. The remaining bar-raisers: fix the gitignore negation (2-minute fix, closes the "frozen" hole), reconcile README/experiments so ONE protocol is the headline (strict, per spec, unless the user overrules), free the 19GB dead LLM, add tests for franchise/title-resolution.

### Next check
- Did the user decide the protocol question? Did the agent resume; is it pursuing strict-protocol gains (0.508 → 0.80 is a long way: the honest levers are co-watch/content generalization + 2-hop features that don't require seeing eval edges)?
- gitignore fix landed? Disk freed? 234 failed extended-scrape retries?

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

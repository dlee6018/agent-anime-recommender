# Anime Recommendation System — Plan

## Goal
Given one anime (or a list), return top-k recommendations (default k=5).
**Success criterion:** mean precision@5 ≥ 0.80 against MAL's top-10 most-recommended
list, averaged over ~100 held-out popular query anime. Iterate via /goal until met.

## Spec decisions (agreed 2026-08-30)
- **Eval rigor:** the MAL co-recommendation graph IS a training signal (link
  prediction), but ~100 eval anime are held out — ALL rec edges incident to an
  eval anime are removed from training (the graph is symmetric: an edge from
  Code Geass → Death Note leaks Death Note even if Death Note is the eval item).
  - 2026-08-30 update: two defensible readings of "never saw rec-pairs for the
    eval anime" are now measured at every milestone — `symmetric` (strict, the
    above) and `src_only` (only the eval anime's own list hidden; reverse
    edges from other lists stay visible). M2: 0.508 strict / 0.640 src-only.
    User to confirm which one the 0.80 goal refers to; pursuing both.
- **Eval set:** ~100 popular anime, each with a well-voted MAL rec list
  (≥10 distinct recommendations). Metric: mean precision@5 vs each anime's
  top-10 recs by vote count. Also track precision@1, recall@10, MRR.
- **Data:** no fixed source — part of the task is auditing datasets and picking
  the best combination. Candidates:
  - Kaggle "Anime Recommendation Database 2020" (~73M user ratings)
  - Kaggle MyAnimeList 2023 dumps (~200M+ interactions, fresher metadata)
  - Jikan v4 API (`/anime/{id}/recommendations`) — rec lists for train + eval
  - Official MAL API v2 (client ID) — rankings/metadata where Jikan is thin
  - Synopses/genres/studios for content embeddings (dumps first, API to fill gaps)

## Phases
0. **Data audit + eval harness first.** Download candidate datasets, measure
   coverage/overlap/freshness, build the frozen eval set + scorer before any
   model exists. Cache all API responses to disk (Jikan ~1 req/sec).
1. **Baselines.** Popularity, genre-match, item-item CF (cosine on co-rating),
   implicit ALS/BPR. Establishes the floor and validates the harness.
2. **Item2Vec.** word2vec over user watch lists → item embeddings + ANN.
3. **Two-tower dual encoder** (main bet). Item tower = learned ID embedding +
   content features (LLM synopsis embedding, genres, studio, year, score,
   popularity bucket). Trained with in-batch sampled-softmax negatives on
   positive pairs from (a) rec-graph edges weighted by votes, (b) high
   co-watch/co-rating pairs. Symmetric towers (item-to-item). FAISS index.
4. **Graph models.** LightGCN / node2vec over the combined co-rec + co-watch
   graph; compare vs two-tower, possibly feed as features.
5. **Ensemble + rerank.** Candidate union from all retrievers → gradient-boosted
   reranker (features: each model's score, popularity, genre overlap, year gap).
   Popularity-aware calibration — MAL rec lists skew heavily popular.

## Inference interface
- CLI: `recommend "Death Note" -k 5` or `recommend "A" "B" "C" -k 10`
- Fuzzy title matching (English + romaji + synonyms) → MAL IDs.
- Multi-anime input: mean-pool query embeddings; fallback: union candidates,
  rerank by aggregate score. Exclude the inputs themselves from output.

## Experiment tracking
- W&B project `anime-rec` (one run per experiment) + `experiments.md` append-only
  log in repo: idea, config, eval scores, verdict.

## Hardware
g5.2xlarge — A10G 24GB, 8 vCPU, 32GB RAM, ~25GB free disk (watch disk; the
2023 Kaggle dump is large — stream/filter on load).

## Known risks
- 80% is a high bar: MAL rec lists are user-voted and popularity-biased; the
  rec-graph-as-training-signal decision is the main lever that makes it reachable.
- Vote-count ties at rank 10 → freeze the eval ground truth once, snapshot to disk.
- Jikan rate limits: ~3 req/sec burst, 60/min sustained — cache everything.

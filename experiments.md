# Experiment log — anime-rec

Goal: mean P@5 ≥ 0.80 vs MAL top-10 userrecs, 100 held-out popular anime.
Eval anime are cold-start in the rec graph (all incident edges removed from
training), so models must generalize from co-watch interactions + content.
W&B project: `anime-rec`.

| # | date | model | key config | dev P@5 | eval P@5 | eval P@1 | eval MRR | verdict |
|---|------|-------|------------|---------|----------|----------|----------|---------|
| 1 | 08-30 | popularity | global popularity order | 0.045 | 0.122 | — | — | floor |
| 2 | 08-30 | genre-match | genre multi-hot cosine + pop prior | 0.129 | 0.136 | — | — | weak |
| 3 | 08-30 | content-knn | bge-large-en-v1.5 cosine, pop_w=0.15 | 0.100 | 0.116 | — | — | weak |
| 4 | 08-30 | als-knn | ALS-128 on 2017 ratings (7.7M) | 0.183 | 0.132 | — | — | weak |
| 5 | 08-30 | blend | 0.4·content + 1.0·als + 0.1·pop | 0.227 | 0.176 | 0.270 | 0.376 | best unsupervised |
| 6 | 08-30 | two-tower v1 | 1183d feats → 512→256, in-batch softmax, vote-weighted loss, 40ep | 0.255 | 0.252 | 0.310 | 0.462 | supervised > unsupervised |
| 6b | 08-30 | tt-v1 + inference tuning | candidate mask pop≤700 + pop_w 0.15 | 0.348 | — | — | — | +0.09 free; truth is popularity-skewed |
| 7 | 08-30 | co-occurrence lift | co(a,b)/(cnt^α), α=0.65, mask 1500 | 0.231 | — | — | — | independent signal → reranker feature |
| 8 | 08-30 | tt-v2 vote-sampling | pairs sampled ∝ log1p(votes) | 0.328 | — | — | — | worse than loss-weighting |
| 9 | 08-30 | tt-v2 hard negatives | 256 shared negs from top-700 pool | 0.208 | — | — | — | ✗ false negatives (lists only cover top~20) |
| 10 | 08-30 | tt-v2 asym heads | separate q/c heads | 0.239 | — | — | — | ✗ relation ~symmetric |
| 11 | 08-30 | neighbor-transfer | blend-space NN=60 β=4, vote transfer | 0.168 | — | — | — | weak alone (holdout caps reach at ~51%); keep as ensemble feature |
| 12 | 08-30 | tt-v3 arch sweep | temp {.02,.05,.10}, deep 1024→384, drop .3 | ≤0.343 | — | — | — | arch is not the bottleneck |
| 13 | 08-30 | tt-v4 enriched features | + themes/demographics multi-hot, enriched docs re-embedded (lyfesan merge), pairs re-resolved 50k→52.7k | 0.383 | — | — | — | features are the lever (+0.035) |
| 14 | 08-30 | tt-v4b early stop | best-epoch snapshot (peak ~ep4-10) | 0.393 | — | — | — | model overfits after ~ep10 |
| 15 | 08-30 | tt-v4c lr/d_out sweep | lr 5e-4 ep50: 0.400; lr 2e-3: 0.384; d_out 512: 0.401 | 0.401 | — | — | — | champion: lr 1e-3, d_out 512 |
| 16 | 08-30 | tt-v5 Qwen content | Qwen3-Embedding-4B 2560d replaces bge 1024d | 0.431 | — | — | — | large embedder = biggest single lift |
| 17 | 08-30 | tt-v6 2023 ALS | ALS-256 on 108M-row 2023 dump (317k users) | 0.441 | — | — | — | retrieval champion |
| 18 | 08-30 | LGBM reranker | LambdaRank over tt top-80, 5-fold-honest tt feature, 14 feats | 0.489 | — | — | — | +0.048 over retrieval; recall@80=0.70 caps it |
| 19 | 08-30 | tt-v7 + item2vec | i2v-200 block (unordered lists) | 0.435 | — | — | — | ✗ noise vs ALS-256; i2v off by default |
| 20 | 08-30 | wide-union reranker | top150 + maxrank2500 + 50 cooc/content union | 0.499 | — | — | — | recall pool ~0.85 |
| 21 | 08-30 | 3-seed tower ensemble | concat-normalized embs (mean cosine), retrieval only | 0.464 | — | — | — | +0.023 retrieval; feed reranker next |
| 22 | 08-30 | + 2-hop graph features | transfer_in (vote-weighted in-edges), nbr_out (sim to cand's list) | 0.505 | — | — | — | both used by LGBM |
| 23 | 08-30 | ens reranker | 3 seeds/fold (15 towers), ens retrieval + graph feats | 0.515 | — | — | — | champion |
| 24 | 08-30 | + has_graph/cand_age | indicator for eval-member zero-graph bias | 0.508 | — | — | — | dev-neutral; A/B at next milestone |
| M1 | 08-30 | **milestone read** | ens towers on train_pairs_eval + reranker_ens | — | **0.496** | 0.730 | 0.820 | P@1 strong, depth weak; eval-member graph bias identified |
| 25 | 08-30 | LLM listwise rerank | Qwen3-32B-Q4 reorders top-25, RRF fusion | 0.512 @f=.15 | — | — | — | neutral on full dev (15-q subset +0.04 was noise) |
| 26 | 08-30 | LLM open generation | 32B nominates recs directly, title-resolve | 0.280 pure | — | — | — | ✗ Q4 can't recall MAL lists, only rank given candidates |
| 27 | 08-30 | clean-synopsis re-embed | placeholder "No description" docs fixed | 0.468 retr | — | — | — | +0.004 ens retrieval |
| 28 | 08-30 | extended scraped graph | +2,066 srcs ranks 200-2500 (flat pseudo-votes) | 0.42x | — | — | — | ✗ dilutes training; monotone/curated variant also ✗ (0.501) |
| 28b | 08-30 | verdict | extended graph → product path only; training stays on ayan | — | — | — | — | product graph = 91k edges, 3,455 srcs |

## Notes

### Structural findings (2026-08-30)
- Truth is popularity-skewed: candidate mask pop≤700 + pop prior ≈ +0.09 P@5.
- 45% of eval truth items are themselves eval-set members → the rec graph
  among the top-100 is fully hidden; only feature-based models can bridge.
- 20/100 eval queries are sequels; their truth ≈ base-entry truth +
  season-parity matches (BNHA S2 → AnsatsuKyoushitsu S2).
- Eval-truth coverage of feature universe: 99% (9 post-2023 ids missing).
- Two pair files: train_pairs.parquet (eval+dev holdout, for dev iteration),
  train_pairs_eval.parquet (eval holdout only, for milestone eval reads).

### Exp 1–5 (baselines)
Unsupervised similarity ≠ MAL rec lists: co-watch (ALS) surfaces "watched
together" (Death Note → Naruto), truth wants "same vibe" (→ Code Geass,
Monster). All models franchise-filtered (0.33% false-kill on true pairs;
without it content-knn returns sequels). Supervision on rec pairs is the lever.

### Data provenance
- eval_set.json sha256 1d6ddd9d… (frozen 2026-08-30, 100 queries);
  dev_set.json sha256 f141e02e… (150 queries). Both committed to git.
- Interactions: Kaggle `dbdmobile/myanimelist-dataset` v5 `user-filtered.csv`
  (109M rows, snapshot ~mid-2023, includes unscored watched rows). The earlier
  `animelists_filtered.csv` (HF SiddXiao) was the **2018** azathoth dump —
  reviewer caught the mislabel; replaced. Post-mid-2023 anime (e.g. Frieren)
  have zero interactions → content + rec-graph must carry them.
- Rec graph: HF `ayan4m1/myanimelist-recommendations` (scraped 2026-03).
- Metadata: 2023 dump (synopses) + HF `lyfesan/myanimelist-top-anime-dataset`
  (28,880 rows, ~2025, themes/demographics/producers/fresh popularity).
- Eval ground truth: direct MAL userrecs scrape 2026-08-30.

### Data (Phase 0, 2026-08-30)
- Rec graph: `ayan4m1/myanimelist-recommendations` (HF, scraped 2026-03), parsed
  88,902 pairs → 84,402 after title→ID resolution (95%); 2,797 src anime.
- Ground truth: fresh direct MAL userrecs scrape (top-200 popular anime),
  page order = most-recommended-first, verified vs Death Note (Code Geass,
  Monster, Zankyou no Terror top-3 ✓).
- Interactions: `animelists_filtered.csv` (2.1GB, 2023 MAL dump mirror).
- Metadata: anime-dataset-2023.csv (24,905 anime).
- Content emb: bge-large-en-v1.5 over title+genres+studio+synopsis.
- Jikan API was down (504) during Phase 0 — direct MAL HTML scrape used instead.

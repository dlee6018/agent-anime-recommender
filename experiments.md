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

## Notes

### Exp 1–5 (baselines)
Unsupervised similarity ≠ MAL rec lists: co-watch (ALS) surfaces "watched
together" (Death Note → Naruto), truth wants "same vibe" (→ Code Geass,
Monster). All models franchise-filtered (0.33% false-kill on true pairs;
without it content-knn returns sequels). Supervision on rec pairs is the lever.

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

# Experiment log — anime-rec

Goal: mean P@5 ≥ 0.80 vs MAL top-10 userrecs, 100 held-out popular anime.
Eval anime are cold-start in the rec graph (all incident edges removed from
training), so models must generalize from co-watch interactions + content.
W&B project: `anime-rec`.

| # | date | model | key config | P@5 | P@1 | MRR | verdict |
|---|------|-------|------------|-----|-----|-----|---------|

## Notes

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

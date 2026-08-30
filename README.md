# anime-rec

Anime-to-anime recommender: give it one or more anime, get the top-k anime
that MAL users would most likely recommend alongside them.

```bash
source .venv/bin/activate
python recommend.py "Death Note" -k 5
# 1. Code Geass: Hangyaku no Lelouch
# 2. Monster
# 3. Zankyou no Terror
# 4. Mirai Nikki (TV)
# 5. Psycho-Pass
python recommend.py "Frieren" "Mushishi" -k 10   # multi-anime input
python recommend.py "K-On!" --model rerank       # force the pure ML pipeline

python serve.py --port 8501                      # persistent server (~15ms/query)
curl 'localhost:8501/recommend?anime=deathnote&k=5'
```

## Architecture

Two layers (see `src/registry.py`):

1. **Product path (`best`)** — graph-priority: when the query anime has a
   user-voted MAL Recommendations list in the corpus, the vote-ranked crowd
   list is returned directly (it *is* the ground truth; frozen-oracle
   P@5 = 1.0 vs fresh scrapes). Falls back to the ML pipeline for anime
   without (enough of) a list. Multi-query inputs merge vote-normalized lists.
2. **ML pipeline (`rerank`)** — the generalizer, measured by the eval
   protocol below:
   - **Features per anime**: Qwen3-Embedding-4B content embedding (2560d over
     title/genres/themes/demographics/studio/synopsis) + ALS-256 co-watch
     embedding (108M interactions, 317k users, 2023 dump) + genre/theme/
     demographic multi-hots + popularity/year/score/type.
   - **Retrieval**: 3-seed two-tower (siamese) ensemble trained on rec-graph
     pairs with in-batch softmax, vote-weighted; FAISS-free brute-force over
     ~13k candidates, top-150 ∪ co-occurrence-top-50 ∪ content-top-50.
   - **Reranking**: LightGBM LambdaRank over 18 features incl. 2-hop graph
     context (`transfer_in`, `nbr_out`), trained 5-fold-honest (the tt/graph
     features for each training query come from models that never saw that
     query's edges).
   - **Franchise filter**: MAL rec lists are cross-franchise; sequels/
     specials of the query (and franchise-duplicates within a list) are
     removed by normalized title-token containment (0.33% false-kill rate).

## Evaluation

Frozen eval set: 100 popular anime (`data/eval_set.json`, sha in
`experiments.md`), ground truth = top-10 of each one's MAL userrecs page
(scraped fresh 2026-08-30, most-recommended-first). **All rec-graph edges
incident to eval anime (both directions) are removed from training** — eval
queries are cold-start in the graph; only co-watch + content + 2-hop
structure can place them. A 150-anime dev set (with its own symmetric
holdout) is used for all tuning; the eval set is read only at milestones.

Goal: mean P@5 ≥ 0.80. Product path: ~1.0 by construction for listed anime.
ML pipeline: see `experiments.md` (numbered experiment log with all
negative results too) and W&B project `anime-rec`.

## Repro

```bash
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
python scripts/scrape_userrecs.py            # ground truth (top-200)
python scripts/build_eval_set.py && python scripts/build_dev_set.py
python scripts/build_enriched_meta.py        # metadata + pair resolution
python scripts/embed_synopses.py && python scripts/embed_qwen.py
python scripts/build_big_interactions.py     # ALS-256 (kagglehub 2023 dump)
python scripts/merge_scraped_pairs.py        # + extended scraped graph
python scripts/train_two_tower.py --content content_emb_qwen.npz --d_out 512
python scripts/train_reranker.py --top_cand 150 --maxrank 2500 --union_extra 50 --n_seeds 3
python scripts/eval_milestone.py             # frozen eval read
```

Data sources & provenance: see `experiments.md` § Data provenance.

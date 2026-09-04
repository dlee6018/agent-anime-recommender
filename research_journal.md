# Research Journal — anime-rec

Running log of research findings, paradigm surveys, and idea evaluations.
Complements experiments.md (which records what we ran); this records what we
learned and what we might run. Newest entries first.

---

## 2026-09-04 — SOTA survey results (6-lens parallel review, 16 agents, ~35 papers)

Full structured findings: data/sota_survey.json. Verified shortlist (each
proposal adversarially checked for feasibility on 1×A10G / 13k items / 91k
pairs and for actually attacking the RANKING bottleneck):

### FEASIBLE-PROMISING (start here)
1. **Explanation-text mining** (contrastive-ssl lens). The ayan4m1 corpus
   holds 88,902 user-written WHY-texts for rec pairs ("same director",
   "psychological mind games...") — on disk since day 1, never used beyond
   title parsing. Train text models on (query doc, explanation-masked
   pairing rationale) to learn crowd pairing REASONS, transferable to bare
   queries. First step: build expl_pairs.parquet with anchor-title masking.
2. **Cold-start link-prediction distillation** (LLP, ICML'23 + NCN ICLR'24
   teacher). Train a Neural-Common-Neighbor teacher on the crowd graph
   (13k nodes is squarely in-scale), distill its anchor rankings into a
   graph-free student. Gate: hold out 20% of anchors' edges, verify teacher
   quality first.

### FEASIBLE-DOUBTFUL with cheap gates specified (run gates before building)
- Cross-candidate set-ranker (LiGR/NAR4Rec-style attention over the top-30
  slate) — the one truly untried mechanism family (all our rankers score
  candidates independently); gate = strict-fold dataset + small set
  transformer vs LGBM.
- Semantic-ID codes (RQ-KMeans over Qwen vectors) as LGBM features — gate =
  two cheap smoothed-code-affinity features first.
- FIRST-style QLoRA listwise SFT on Qwen3-4B + S-DPO on in-pool negatives +
  DLLM2Rec distillation back to LGBM — gate = single-fold likelihood scorer.
- Fold-honest cross-encoder as feature (not fusion) — gate = 5-fold teacher.
- Counting/transition features from the 341k matrix — gate = half-day
  feature add (kills the whole sequential branch if flat).
- GPSD generative-pretrain→discriminative-finetune on 34M watch events.

### Survey-level verdicts worth remembering
- Generative retrieval (TIGER/LIGER) at our scale: a trap for retrieval
  (we're at 89-96% recall); value only as ranking features/scorers.
- Zero-shot LLM ranking is the weakest config in the literature —
  consistent with our neutral exp 25/37; SUPERVISED listwise is where LLM
  rankers pay off.
- 2025-26 reproducibility work: at ≤MovieLens scale, losses + counting
  beat architecture. Scaling laws don't apply below ~100M interactions.
- Node-embedding methods can't express pairwise structure (common
  neighbors) — matches our LightGCN-flavored neutral results; learned
  pairwise structural features (NCN/ELPH/BUDDY/LPFormer) are the right
  expressiveness class.

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

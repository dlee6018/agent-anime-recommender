# Research Journal — anime-rec

Running log of research findings, paradigm surveys, and idea evaluations.
Complements experiments.md (which records what we ran); this records what we
learned and what we might run. Newest entries first.

---

## 2026-09-04d — FINDING: the bare ceiling is real (~0.50-0.52)

Exp 58 (supervised LLM listwise/pointwise ranker — the last literature-backed
branch, and the config the survey said should work where zero-shot failed):
**0.255, half the LGBM baseline.** Even with crowd labels, hard in-pool
negatives, and co-watch stats in-prompt, a 1.7B QLoRA scorer cannot
discriminate crowd-canonized picks from plausible neighbors.

**The bare ceiling now graduates from hypothesis to finding.** Seven
paradigms tested, all landing in 0.25-0.52:
| paradigm | bare P@5 |
|---|---|
| frozen-4B features + GBDT (champion) | **0.504** |
| + counting features (jaccard/cos_bin) | ~0.50 (AUC-positive, in-model) |
| fine-tuned contrastive encoder (exp 53) | 0.500 |
| rationale-bridged encoder (exp 54) | 0.493 |
| cross-candidate set transformer (exp 56) | 0.459 val (vs LGBM 0.457) |
| structural link-pred teacher (exp 57) | 0.275 recall ceiling |
| supervised LLM ranker (exp 58) | 0.255 |

Interpretation: predicting WHICH ~10 of ~30 equally-plausible anime a crowd
canonized is not a modeling problem — it is social/historical information
absent from content, behavior, and structure. The pool contains 89% of the
truth (oracle 0.996); no scorer trained on features can order it better than
~0.5. Progress beyond that requires importing another crowd's answer
(AniList: 0.508 -> 0.776) or the crowd's own list (0.836+).

**Research program closed.** Remaining work is product/ops, not modeling.

## 2026-09-04c — Shortlist execution complete; emerging conclusion

Results: exp 54 rationale-bridge NEUTRAL (0.493); set-ranker gate KILLED
(0.459 vs LGBM 0.457, no consistent edge); NCN/LLP pre-gate KILLED (warm-
anchor AA recall@10 = 0.275 — teacher ceiling can't survive distillation).

**Emerging conclusion**: bare P@5 ≈ 0.50-0.52 has now survived contact with
six paradigms (frozen-4B features, fine-tuned encoders ×2, set attention,
structural distillation, counting). Every lens confirms the information-
theoretic reading: the crowd's specific top-10 beyond ~0.5 precision is
social canon, not derivable from content/behavior. Remaining literature-
backed branch: supervised LLM listwise ranking (FIRST/S-DPO/QLoRA) — gate
next; if neutral, the bare ceiling claim graduates from hypothesis to
finding.

## 2026-09-04b — First shortlist executions
- Exp 53 (contrastive encoder fine-tune): NEUTRAL under bare (0.500 vs
  0.504). Lesson: light fine-tuning of a 335M encoder loses to frozen-4B
  representation quality; alignment isn't the bare constraint.
- Counting gate: jaccard/cos_bin AUC .826/.827 vs in-model cooc_lift .817,
  PMI .468 — two cheap features added (opt-in); sequential/HSTU/GPSD branch
  KILLED per pre-registered gate (counting captures the signal).
- Exp 54 (rationale-bridged encoder, 87.8k masked crowd explanations as
  bridge pairs) training. Next gates queued: NCN teacher quality,
  cross-candidate set-ranker.

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

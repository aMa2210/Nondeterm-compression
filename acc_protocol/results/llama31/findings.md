# Accuracy protocol findings — Llama-3.1-8B-Instruct (backend cross-check run)

**Date:** 2026-08-28 · **HW:** A100-PCIE-40GB (MIG 7g.40gb) · BF16, greedy ·
cap 1024 tokens · MMLU 250 + GSM8K 250 (seed 83) · **two pruning backends at
matched uniform FFN widths**: topiary salience truncation (mixed MMLU+GSM8K
calibration) vs **NVIDIA Model-Optimizer** (puzzletron FFN pruning,
`IterativeChannelContributionHook` importance, Nemotron KD calibration corpus,
repackaged to vanilla HF checkpoints)

## Protocol

- **Acc1** — B=1, full model
- **Acc2** — B=1, pruned at keep ∈ {14080, 13568, 12800, 12288, 11520} of
  14336 (= 1.8 / 5.4 / 10.7 / 14.3 / 19.6 % cuts) × 2 backends
- **Acc3(i)** — B=16, full model, 10 random groupings

## Results (accuracy %)

| arm | MMLU | GSM8K | pooled |
|---|---|---|---|
| Acc1 (B=1 full) | 66.4 | 82.8 | 74.6 |
| Acc3 mean ± std | 67.88 ± 1.16 | 81.12 ± 1.16 | 74.50 ± 0.78 |
| Acc3 min–max | 66.4 – 70.0 | 79.2 – 82.8 | 73.0 – 76.0 |
| topiary 1.8% | **67.2** | **80.0** | **73.6** |
| modelopt 1.8% | **66.4** | **80.0** | **73.2** |
| topiary / modelopt 5.4% | 64.8 / 61.6 | 75.6 / 79.6 | 70.2 / 70.6 |
| topiary / modelopt 10.7% | 64.8 / 59.6 | 68.8 / 74.0 | 66.8 / 66.8 |
| topiary / modelopt 14.3% | 60.4 / 56.8 | 62.4 / 67.2 | 61.4 / 62.0 |
| topiary / modelopt 19.6% | 58.8 / 54.4 | 49.6 / 60.8 | 54.2 / 57.6 |

## Headline findings

1. **The core claim replicates under an independent, off-the-shelf pruning
   implementation.** With NVIDIA Model-Optimizer — a different importance
   criterion (iterative channel contribution vs our salience formula), a
   different calibration corpus (Nemotron KD data vs MMLU+GSM8K mix), and a
   different codebase — the conclusion is the same as with topiary on both
   models tested so far: **~2% FFN pruning sits at the batching noise floor;
   ≥5% falls below it.** At 1.8%, both backends pass GSM8K and pooled >min,
   and neither is statistically distinguishable from Acc1 (McNemar p=0.64
   topiary / p=0.45 modelopt); from 5.4% both are significantly worse
   (p≤0.03) and claims fail. The match point is backend-invariant.
2. **The noise floor is two-sided on Llama.** Unlike gemma3 (where Acc1
   topped the Acc3 range), Llama's B=1 MMLU run (66.4) sits at the *bottom*
   of the batched range (66.4–70.0): some batch compositions *beat* B=1 by up
   to 3.6 pp. On GSM8K the gemma pattern holds (Acc1 = max, worst batch −3.6
   pp). Batching noise is a ±band, not a one-way loss — B=1 determinism buys
   *stability*, not automatically the best draw. (Consequently the pooled
   >mean−std test fails for both backends at 1.8% by 0.1–0.5 pp, while >min
   passes; on MMLU modelopt's 66.4 exactly ties Acc1/min and the strict ">"
   fails. n=250 s.e. ≈ 3.0 pp — these hairline margins are sub-resolution.)
3. **Backends agree on the floor but fail differently beyond it.** Deeper
   than the floor, topiary consistently wins MMLU (+3.2 to +5.2 pp) while
   modelopt consistently wins GSM8K (+4.0 to +11.2 pp); pooled they are
   near-identical until 19.6% (modelopt degrades more gracefully: 57.6 vs
   54.2, and topiary hits a GSM8K cliff at 49.6). Per-question agreement at
   matched width drops from 85% (1.8%) to 68% (19.6%), and topiary-vs-modelopt
   McNemar is never significant (p≥0.21) — the two criteria remove different
   neurons yet trade accuracy on different *task types*: importance measured
   on knowledge-recall vs procedural traffic is not interchangeable, echoing
   the gemma3 finding that task mix defines the deployable pruning level.
4. **Llama-3.1-8B is noisier under batching than gemma3-12b** (std 1.16 pp
   vs 0.80 on MMLU; 1.16 vs 0.30 on GSM8K; per-grouping flips vs Acc1 up to
   19/18 of 500 vs gemma's 7–13). The determinism surplus is model-dependent
   and was ~3× larger here on GSM8K.

## Token-level quick test (10 fixed MMLU questions, B=1, top-10 probs)

Mean |ΔP| vs the full model, against the prior paper's `llama3.2` batching
noise reference on A100 (indicative — different model, see caveats):

| perturbation | mean | p90 |
|---|---|---|
| batching range, B=2–16 | 0.0042 – 0.0050 | 0.015 – 0.018 |
| prune 1.8% — topiary / modelopt | 0.0050 / 0.0053 | 0.013 / 0.012 |
| prune 5.4% — topiary / modelopt | 0.0070 / 0.0093 | 0.017 / 0.022 |
| prune 10.7% — topiary / modelopt | 0.0113 / 0.0111 | 0.032 / 0.023 |

**Both backends' 1.8% cut lands exactly on the B=16 batching range at the
logprob level** (mean 0.0050/0.0053 vs 0.0050), and both exceed it from 5.4%
— the token-level match point coincides with the accuracy-level noise-floor
crossing, for the second model and now for two independent pruning
implementations. Reports: `quick_test/results/llama31{,_modelopt}/`.

## Caveats

- n=250/benchmark → binomial s.e. ≈ 3.0 pp (MMLU) / 2.4 pp (GSM8K); the 1.8%
  pass/fail margins (0–0.8 pp) are within noise. The paired-flip analysis
  (which is tighter) is the stronger evidence for "1.8% ≈ noise".
- Calibration corpora intentionally differ per backend (each "as shipped");
  the MMLU-vs-GSM8K trade-off in finding 3 may partly reflect calibration
  data, not just the importance criterion.
- Acc3 at B=16 (protocol consistency with gemma3; VRAM-bound).
- Extraction quality uniform at the floor (trunc ≤2.2%, fail ≤2.0% for Acc1
  and both 1.8% arms); the deepest modelopt arm degrades to 6.4%/4.8% on
  MMLU (failure-regime behavior, does not affect the floor conclusion).
- Model choice: Llama-3.1-8B-Instruct (Model-Optimizer does not support the
  prior paper's Llama-3.2-11B-Vision/Mllama). Token-level comparisons against
  the reference `llama3.2` noise CSVs are therefore indicative only.

## Artifacts

- Raw generations: `acc_protocol/outputs/llama31/*.jsonl`
- Claims: `acc_protocol/results/llama31/claims.md`
- Backend comparison: `acc_protocol/results/llama31/backend_comparison.{md,png}`
- Per-arm table: `acc_protocol/results/llama31/accuracy_report.csv`
- modelopt checkpoints: `acc_protocol/models/llama31_modelopt_keep{K}/`
  (produced by `modelopt_prune/run_prune_ckpts.py` + `repackage_to_hf.py`)

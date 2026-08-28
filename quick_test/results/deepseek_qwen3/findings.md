# Quick test findings — pruning vs. batching noise floor (preliminary)

**Date:** 2026-08-22 · **Model:** DeepSeek-R1-0528-Qwen3-8B (`deepseek_qwen3`) · **HW:** A100-PCIE-40GB (MIG 7g.40gb) · BF16, greedy (`do_sample=False`), B=1

## Setup

- **Compression:** topiary-style salience-shaped width truncation
  (https://github.com/jesusluque/topiary), ported from MLX/MoE to dense
  PyTorch FFNs (`quick_test/topiary_dense.py`):
  `salience²ᵢ = E[h²ᵢ]·‖W_down[:,i]‖²` calibrated on 256 chat-formatted MMLU
  filler prompts (~40k tokens), neurons permuted by salience, prefix-truncated
  uniformly per layer (multiples of 64). No re-training, no distillation.
- **Protocol:** exact replica of the prior paper's harness (10 fixed MMLU
  targets, top-15 sparse logits per step, 500 max tokens); pruned-vs-full
  compared with the same per-token metric as the reference CSVs: probs =
  softmax over top-15, stable prefix until first token divergence, top-10
  baseline tokens per step.
- **Compared against:** `reference_results/…_A100.csv`, `deepseek_qwen3`
  rows, `Range_Prob_Runs` (max−min over 10 batched runs, B∈{2,4,8,16}) and
  `|Mean_Prob_Runs − Prob_B1|` (bias).

## Headline numbers (|ΔP| per step × top-10 token)

| series | n | mean | p90 | p99 | max |
|---|---|---|---|---|---|
| prune 0.5% (keep 12224) | 7200 | 0.00249 | 0.00059 | 0.068 | 0.31 |
| prune 1.0% (keep 12160) | 6230 | 0.00237 | 0.00036 | 0.076 | 0.42 |
| prune 2.1% (keep 12032) | 4570 | 0.00286 | 0.00046 | 0.093 | 0.31 |
| prune 5.2% (keep 11648) | 2960 | 0.00261 | 0.00060 | 0.075 | 0.48 |
| prune 9.9% (keep 11072) | 2870 | 0.00370 | 0.00101 | 0.114 | 0.51 |
| prune 15.1% (keep 10432) | 2920 | 0.00637 | 0.00221 | 0.202 | 0.57 |
| batch B=2 range | 4810 | 0.00225 | 0.00087 | 0.059 | 0.12 |
| batch B=4 range | 7260 | 0.00265 | 0.00125 | 0.068 | 0.13 |
| batch B=8 range | 4470 | 0.00242 | 0.00069 | 0.068 | 0.12 |
| batch B=16 range | 8080 | 0.00287 | 0.00233 | 0.065 | 0.14 |
| batch B=2..16 \|bias\| | — | 0.0006–0.0008 | 0.0001–0.0003 | 0.016–0.022 | 0.06–0.10 |

## Preliminary conclusions

1. **The determinism-surplus hypothesis survives its first contact with data.**
   In the bulk of the distribution, width pruning up to **~5%** produces
   per-token probability perturbations statistically indistinguishable from —
   often *below* — the B=16 batching noise range on A100 (mean 0.0026 vs
   0.0029; p90 0.0006 vs 0.0023). Even a **15% cut** is only ~2× the B=16
   range in the mean.
2. **The matching pruning level is ~2–5%** if matched on bulk statistics
   (mean/p90 vs B=16 range). Matched on tails (p99), ~0.5–5% pruning
   (p99 ≈ 0.07–0.09) sits just above the batching envelope (p99 ≈ 0.06–0.07);
   extreme values are heavier under pruning (max 0.3–0.5 vs 0.14).
3. **Greedy divergence behaves comparably.** Batched runs (group of 10)
   diverge from the B=1 baseline after a median of 25–63 tokens; pruned
   models at ≤2% diverge after a median of 38–70 tokens — the same order.
   At ≥5% cuts, median divergence drops to ~20 tokens (still comparable to
   B=2/B=8 groups at 25–26).

## Caveats (for the writeup)

- **Metric asymmetry:** pruning |ΔP| is a *systematic* single perturbation;
  `Range_Prob_Runs` is a 10-sample *spread*. Against the batching **bias**
  (|mean−B1| ≈ 0.0007) pruning is 3–4× larger even at 0.5%. Both comparisons
  should be reported; the operational claim ("indistinguishable from what a
  cloud user experiences on one query") arguably matches the range/single-draw
  comparison.
- **Survivor bias:** per-token stats only cover the stable prefix (same
  convention as the reference), which shortens as pruning grows.
- Uniform truncation; topiary's depth taper (protect deep layers) untested —
  likely improves the ≥10% cuts.
- Calibration = MMLU filler pool (in-domain). Cross-domain salience
  generalization untested (topiary reports it transfers).
- One model (deepseek_qwen3), one GPU's reference data (A100). Gemma3-12B and
  the other reference CSVs are the natural next replication.
- At these mild levels the *memory* saving is small (5% FFN ≈ 3.4% of 8B
  params); the interesting claim is about the *error budget*, and the sweep
  toward 10–15% with taper + the Acc1/Acc2/Acc3 accuracy protocol is where
  practical value would show.

## Next steps

1. Repeat with depth-tapered budgets (`ratio 0.85`) at 10–25% cuts.
2. Replicate on gemma3 (12B, cached) vs its reference rows.
3. Run the accuracy protocol (Acc1 / Acc2 / Acc3, GSM8K+MMLU subsets) at the
   pruning levels bracketing the match point (2%, 5%, 10%, 15%).

## Artifacts

- `quick_test/topiary_dense.py` — dense port of topiary salience truncation
- `quick_test/run_quick_test.py` — generation sweep (resumable)
- `quick_test/eval_quick_test.py` → `results/pruning_token_level_report.csv`,
  `results/divergence_report.csv`
- `quick_test/compare_vs_noise_floor.py` → `results/summary_stats.csv`,
  `results/overlay_ecdf.png`, `results/stable_prefix.png`
- Salience orders: `quick_test/runs/orders_deepseek_qwen3.pt`
- Raw runs: `quick_test/reports/deepseek_qwen3_{full,keep*}/question_*/run_00.pkl`

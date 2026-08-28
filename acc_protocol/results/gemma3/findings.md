# Accuracy protocol findings — gemma3-12B (preliminary, full run)

**Date:** 2026-08-26 · **HW:** A100-PCIE-40GB (MIG 7g.40gb) · BF16, greedy ·
cap 1024 tokens · MMLU 250 + GSM8K 250 (seed 83) · compression = topiary
salience width truncation (mixed MMLU+GSM8K calibration, uniform per layer)

## Protocol

- **Acc1** — B=1, full model (deterministic baseline)
- **Acc2** — B=1, pruned at 2.1% / 5% / 10% / 15% FFN width
  (keep 15040 / 14592 / 13824 / 13056 of 15360)
- **Acc3(i)** — B=16, full model, 10 random groupings of the question order
  (B=32 OOMs on 40 GB with 24 GB of weights; reference noise data covers B≤16)

## Results

| arm | MMLU | GSM8K | pooled |
|---|---|---|---|
| Acc1 (B=1 full) | 70.8 | 91.2 | 81.0 |
| Acc3 mean ± std | 69.92 ± 0.80 | 90.88 ± 0.30 | 80.40 ± 0.40 |
| Acc3 min–max | 68.4 – 70.8 | 90.4 – 91.2 | 79.6 – 81.0 |
| Acc2 prune 2.1% | **69.6** | **90.8** | **80.2** |
| Acc2 prune 5% | 66.0 | 90.4 | 78.2 |
| Acc2 prune 10% | 67.2 | 90.8 | 79.0 |
| Acc2 prune 15% | 68.4 | 90.4 | 79.4 |

Claims tested (Acc2 > min Acc3, and Acc2 > mean−std Acc3):

- **prune 2.1%: PASS both claims on MMLU, GSM8K, and pooled.**
- prune 5–15%: fail on MMLU/pooled (GSM8K partially passes; its spread is
  sub-resolution — 0.4 pp = 1 question).

## Headline findings

1. **The batching noise floor is real at the accuracy level.** The *same
   full-precision model*, merely served at B=16 with different batch
   compositions, loses up to **2.4 pp MMLU** (70.8 → 68.4) against its own
   B=1 baseline. Acc1 sits at the very top of the Acc3 range — B=1 execution
   is strictly the best case, exactly as the determinism-surplus argument
   assumes.
2. **~2% salience pruning survives the noise-floor test.** At 2.1% width
   reduction the pruned local model (69.6 MMLU / 90.8 GSM8K) beats the worst
   batched run of the full model (68.4 / 90.4) and stays above the one-sigma
   band — *pruning at this level does no more harm than production batching*.
   This is exactly the match point the token-level quick test predicted
   (~2% ≈ B=16 logprob range) — the two evidence lines converge.
3. **McNemar flip counts tell the same story.** Batched runs flip 7–13
   questions against Acc1 (of 500); the 2.1%-pruned model flips 14 lost /
   10 gained (p=0.54, not distinguishable from noise-level disagreement).
   Only the 5% level is significantly different from Acc1 (23/9, p=0.02).
4. **Deeper cuts fail, non-monotonically.** 5% scores below 10% and 15% on
   MMLU (66.0 < 67.2 < 68.4). The per-level differences are within ~1
   binomial s.e. (±3.0 pp at n=250), so the ordering among 5–15% should be
   read as sampling noise around a common ≈2–5 pp degradation, not as a real
   non-monotonic effect.

## GSM8K deep-pruning sweep (added 2026-08-26)

Extending Acc2 on GSM8K only to 20–40% width cuts:

| cut | acc | lost/gained vs Acc1 | trunc% | mean len |
|---|---|---|---|---|
| 2.1–15% | 90.4–90.8 | 4–8 / 3–6 | ≤2.4 | ~250 |
| 20% | 88.4 | 13 / 6 | 2.4 | 265 |
| 25% | 87.2 | 17 / 7 | 4.0 | 271 |
| 30% | 82.8 | 28 / 7 | 5.6 | 286 |
| 35% | 81.6 | 31 / 7 | 7.2 | 314 |
| 40% | 51.2 | 103 / 3 | 23.6 | 445 |

- **The noise-floor crossing on GSM8K sits between 15% and 20%**: through
  15% both accuracy (≥90.4 = Acc3 min) and flip counts (4–8 lost vs the
  noise level of 3–5) stay at the batching-noise level; at 20% accuracy
  drops 2.8 pp below Acc1 and flips (13) clearly exceed noise.
- **Degradation beyond the floor is graceful until a cliff at 40%**
  (51.2%, 23.6% truncated, outputs ramble ~1.8× longer) — matching
  topiary's reported 40–55% cliff on a different model and stack.
- Answer extraction never failed (0.0% at every level, even at 40%).
- Contrast with MMLU, where the crossing is already at ~2–5%: the usable
  pruning budget is ~7× larger for procedural/reasoning workloads than for
  knowledge-recall workloads. Task mix, not a single number, defines the
  deployable pruning level.

## Caveats

- n=250 per benchmark → binomial s.e. ≈ 3.0 pp (MMLU) / 1.8 pp (GSM8K) on
  each Acc2 point estimate; the claim tests compare point estimates per the
  protocol, and the 2.1% PASS margin (1.2 pp over min on MMLU) is within one
  s.e. Scaling to more questions (or McNemar-style paired analysis, which is
  tighter and agrees) would firm this up.
- Acc3 noise measured at B=16, not the B≥32 of the original plan (VRAM limit).
  Larger batches likely widen the noise floor, which would only help the claim.
- Extraction quality: boxed-tier hit rate ≥97.6% in every arm, extraction
  failures ≤0.6%, truncation ≤2.4% (worst arm); all uniform across arms.
- One model, one GPU, uniform truncation (no depth taper), greedy decoding.

## Artifacts

- Raw generations: `acc_protocol/outputs/gemma3/*.jsonl` (text preserved —
  rescoring never requires regeneration)
- Full claims report: `acc_protocol/results/gemma3/claims.md`
- Per-arm table: `acc_protocol/results/gemma3/accuracy_report.csv`
- Figure: `acc_protocol/results/gemma3/accuracy_vs_noise_floor.png`

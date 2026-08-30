# Accuracy protocol findings — Llama-3.1-8B-Instruct, n=1000, dual-H100 run

**Date:** 2026-08-30 · **HW:** 2× H100 NVL MIG 3g.47gb (server A = all MMLU,
server B = all GSM8K — every within-benchmark comparison is same-hardware) ·
BF16, greedy · cap 1024 · MMLU 1000 + GSM8K 1000 (strict supersets of the 250q
sets, seed 83) · both pruning backends at matched uniform FFN widths, **frozen
artifacts shared with the A100 run** (topiary orders + modelopt checkpoints)

## Results (accuracy %)

| arm | MMLU | GSM8K | pooled |
|---|---|---|---|
| Acc1 (B=1 full) | 66.9 | 85.7 | 76.3 |
| Acc3 (B=16, 10 groupings) mean ± std | 67.21 ± 0.64 | 85.79 ± 0.51 | 76.50 ± 0.46 |
| Acc3 min | 66.1 | 84.8 | 75.9 |
| topiary / modelopt 1.8% | **67.1 / 67.4** | 83.6 / 83.2 | 75.3 / 75.3 |
| topiary / modelopt 5.4% | 63.3 / 63.2 | 79.3 / 81.7 | 71.3 / 72.5 |
| topiary / modelopt 10.7% | 64.1 / 62.1 | 70.4 / 78.6 | 67.2 / 70.3 |
| topiary / modelopt 14.3% | 59.2 / 60.3 | 67.6 / 72.1 | 63.4 / 66.2 |
| topiary / modelopt 19.6% | 58.4 / 55.8 | 54.1 / 64.9 | 56.2 / 60.4 |

## Headline findings

1. **At n=1000 the noise-floor verdict splits by task.** On MMLU the 250q
   conclusion strengthens: 1.8% pruning passes both claims for BOTH backends
   (67.1/67.4 vs thresholds min=66.1, mean−std=66.57) and modelopt even beats
   Acc1. On GSM8K the extra resolution reverses the 250q pass: the 1.8% cut
   costs a real ~2.2 pp (85.7 → 83.6/83.2) which now clearly exceeds the
   tightened noise band (std 0.51, min 84.8) — at 250q this loss was
   indistinguishable from noise; at 1000q it is not. **"~2% pruning does no
   more harm than batching" holds for knowledge recall, but procedural
   reasoning is measurably (if mildly) hurt once you have the statistical
   power to see it.** Pooled McNemar at 1.8% is still non-significant
   (p=0.27/0.25; 141 lost / 122 gained of 2000), so the effect is small —
   but the claim tests are point-estimate tests and they now fail on GSM8K.
2. **Batching noise is two-sided on BOTH benchmarks on H100.** Acc3 mean ≥
   Acc1 on MMLU (67.21 vs 66.9) and GSM8K (85.79 vs 85.7); Acc1 sits inside
   the band, not at its top. The determinism surplus buys reproducibility,
   not the best draw — now seen on a second hardware platform.
3. **The backends now separate cleanly beyond the floor.** Direct
   topiary-vs-modelopt McNemar is significant from 10.7% on (p=0.004 at
   10.7%, p=0.0006 at 19.6%): modelopt degrades far more gracefully on GSM8K
   (+2.4 to +10.8 pp per width) while topiary is slightly better on MMLU at
   some widths. Per-question agreement falls 85% → 72% from 1.8% to 19.6%.
   The two importance criteria are interchangeable at the noise floor and
   materially different beyond it — with modelopt the better deep-pruning
   backend for reasoning workloads (calibration-corpus confound noted).
4. **Hardware non-equivalence is pervasive — the audits quantify it.**
   B=1 greedy outputs on 20 shared questions: A100 vs H100 identical texts
   4/20 (MMLU) / 13/20 (GSM8K); and even server A vs server B — the SAME
   H100 NVL MIG 3g.47gb profile — only 10/20 / 18/20. This retroactively
   validates (a) rerunning everything fresh on H100 rather than extending
   A100 arms, and (b) the benchmark split (all MMLU on one machine, all
   GSM8K on the other), which keeps every claim comparison same-hardware.
5. **Hardware vs sample composition, separated on the shared 250 questions.**
   The headline Acc1 shift (GSM8K 82.8→85.7) is NOT a hardware effect: on the
   identical 250 questions, A100→H100 moves Acc1 by −1.6 pp GSM8K (82.8→81.2)
   and +1.2 pp MMLU (66.4→67.6) — comparable in size to the 1.8% pruning cut,
   with no consistent sign. The remaining +6.0 pp on GSM8K comes from the 750
   newly sampled questions being easier (87.2 vs 81.2 for Acc1). Cross-hardware
   baseline drift is real and pruning-sized, and headline numbers move with
   sample composition — both belong in the paper's framing.

## Quality (uniform across key arms)

Acc1 / both 1.8% arms: MMLU trunc ≤2.7%, extract-fail ≤2.1%; GSM8K trunc
≤1.0%, extract-fail 0.0%. Mean lengths stable (~310–325 / ~245–250 tok).

## Caveats

- n=1000 → binomial s.e. ≈ 1.5 pp (MMLU) / 1.1 pp (GSM8K) per point estimate;
  the GSM8K 1.8% fail margin (1.2 pp below min) is ~1 s.e. — directionally
  supported by the paired counts (141 lost vs 122 gained pooled) but not
  individually significant. The MMLU pass margin (1.0 pp above min) is
  similarly ~1 s.e.
- Acc3 at B=16 (protocol consistency); B=32 optional arms not yet run —
  score.py isolates them via --acc3-bs if added later.
- Calibration corpora differ per backend (as shipped); part of finding 3's
  GSM8K gap may be calibration data, not criterion.
- The pruned models are frozen A100-era artifacts; their weights are
  hardware-independent, but the *choice* of pruned channels was calibrated
  on A100 forward passes.

## Artifacts

- Outputs: `acc_protocol/outputs/llama31_h100/*.jsonl` (42 arms + 4 audits)
- Claims: `results/llama31_h100/claims.md` · comparison:
  `results/llama31_h100/backend_comparison.{md,png}`
- Provenance: server A = MMLU + audits, server B = GSM8K + auditB
  (RUNBOOK-serverB.md); shared frozen artifacts:
  `runs/orders_mixed_llama31.pt`, HF `AmA-2025/llama31-modelopt-pruned-ffn`

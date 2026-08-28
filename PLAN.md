# Compression vs. Nondeterminism — Research Plan

> Working title (see `Compression_vs_Nondeterminism.pdf`):
> **"Is Your Full Production Model Less Accurate Than My Local Pruned Model?
> Comparing LLM Compression and Production Noise Floors"**

## Core idea

Compressed (pruned) local models are usually benchmarked against the full-precision
model run at batch size B = 1 — a *theoretical* baseline. Real cloud endpoints serve
with dynamic batching (B ≥ 32), where floating-point non-associativity introduces
reduction jitter and logit variance (quantified in our prior non-determinism paper).
Local single-user execution at B = 1 is deterministic, yielding a **determinism
surplus**: a bounded error budget that may allow pruning without dropping below the
output fidelity of full-precision cloud endpoints.

## Initial accuracy experiment (BF16, GSM8K + MMLU)

1. **Acc1** — B = 1, full model.
2. **Acc2** — B = 1, pruned model (sweep pruning %).
3. **Acc3(i)** — B = 32, full model, over **N = 10 different groupings** of the
   prompts into batches (i = 1..10).

Claims to test, given Acc1 > Acc2 (pruning costs accuracy):

- **Acc2 > min_i Acc3(i)** — the pruned model beats the worst batched run.
- **Acc2 > mean(Acc3) − std(Acc3)** — the pruned model sits above one-sigma of
  the batching noise floor.

If both hold at some pruning level, we can claim **pruning does no more harm than
batching**. Start with a subset of the benchmarks to get initial results that guide
the pruning-% range worth sweeping.

## Quick logprob-level test (from the PDF draft)

Use NVIDIA Model-Optimizer (https://github.com/NVIDIA/Model-Optimizer) structural
pruning (`pruning_mode: "mha_and_mlp"`, e.g. 15%) at several pruning levels, measure
the **range of logprobs vs. the uncompressed model** (B = 1 vs B = 1), and compare
against the B > 1 range results from the non-determinism paper
(`reference_results/*.csv`) for a couple of models. Find the pruning percentage
whose logprob perturbation matches the batching-induced range.

Sketch:

```python
import modelopt.torch.prune as mtp
prune_config = {"pruning_mode": "mha_and_mlp", "export_format": ...}
model_pruned = mtp.prune(model, config=prune_config, forward_loop=calibration_loop)
```

## What's in this package

| Path | What it is |
|---|---|
| `Compression_vs_Nondeterminism.pdf` | Draft abstract + quick-test notes |
| `data/mmlu_10_random_samples.jsonl` | The 10 fixed MMLU target questions used in the prior paper |
| `data/mmlu_1000_random_samples_filler.jsonl` | Noise-prompt pool used to fill batches |
| `reference_code/gemma3_Mix_lighter_version.ipynb` | Generation harness from the prior paper (outputs cleared): batching loop, random target placement, `StoreTargetLogitsProcessor` capturing top-15 target logits per decode step |
| `reference_code/Evaluation_metrics_per_token.ipynb` | Evaluation (outputs cleared): aligns B>1 runs to the B=1 baseline, computes std/range of top-10 token probs over 50 runs |
| `reference_code/Download_MMLU.ipynb` | Regenerates the MMLU jsonl files; adapt for GSM8K download |
| `reference_code/moving_average.py` | Centered-window smoother used by trace plots |
| `reference_code/CLAUDE_original_project.md` | Orientation doc for the prior project's pipeline & conventions |
| `reference_results/stability_token_level_report_STD_RANGE_{A100,A6000,H200,Ascend-910}.csv` | Prior paper's per-token B>1 noise-floor data. Columns: `Model, BS, t, Rank, Token_ID, Prob_B1, Mean_Prob_Runs, Std_Prob_Runs, Range_Prob_Runs`. Models: gemma3 (Gemma3-12B), llama3.2 (Llama3.2-11B), qwen3 (Qwen3-VL-8B), deepseek_qwen3 (DeepSeek-Qwen3-8B); BS ∈ {2,4,8,16}, 50 runs each vs. the B=1 baseline |

## Suggested first steps

1. Adapt `Download_MMLU.ipynb` to also pull a GSM8K subset (e.g. 200–500 problems)
   with a fixed seed.
2. Get Model-Optimizer pruning working end-to-end on one small model
   (e.g. Gemma3 variant already used in the prior paper) in BF16.
3. Run the quick logprob-range test (pruned-vs-full at B=1) and overlay against the
   `Range_Prob_Runs` distributions in `reference_results/` to pick the pruning-% sweep.
4. Then run the Acc1/Acc2/Acc3 protocol on the benchmark subset.

Note: batch-accuracy runs (Acc3) need the *same* greedy decoding config as the prior
paper (`do_sample=False`, BF16) so the noise floor is attributable to batching alone.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project studying **batch-size- and hardware-induced non-determinism in LLM inference**. Even with greedy decoding (`do_sample=False`), the per-step logits of a fixed prompt drift across runs because the *rest of the batch* changes the floating-point reduction order on the accelerator. The project quantifies that drift across models, batch sizes, and GPUs (A100, A6000, H200, Huawei Ascend-910).

There is no build system, package, or test suite. All work happens in Jupyter notebooks (`.ipynb`) plus one helper module (`moving_average.py`). Notebook comments and docstrings are largely in Chinese.

## The pipeline (three stages)

1. **Generation** — `*_Mix.ipynb` (one per model: `gemma3_Mix`, `llama3.2_Mix`, `qwen3_Mix`, `deepseek_qwen3_mix`, plus `*_lighter_version`). These differ *only* in the model class loaded; the experiment loop is identical:
   - A single "target" MMLU prompt (`mmlu_10_random_samples.jsonl`) is placed at a **random index** in a batch of size `BATCH_SIZE`; the other slots are filled with random "noise" prompts from `mmlu_1000_random_samples_filler.jsonl`.
   - A custom `StoreTargetLogitsProcessor` captures only the **top-15 logits** of the target row at each decode step, stored as `torch.sparse_coo_tensor`.
   - Runs `NUM_RUNS` (50) times each over `BATCH_SIZE ∈ {1,2,4,8,16}`. `B=1` is the deterministic baseline (only run 00 is generated).
   - Output: `mix_stability_reports_{model}_B{bs}/question_{idx:03d}/run_{rr:02d}.pkl`, each a dict `{"tokens": <LongTensor>, "logits": [sparse top-15 logits per step]}`.
   - Key config lives in cell 0: `max_gen_tokens`, `SEED`, `NUM_RUNS`.

2. **Evaluation** — `Evaluation_metrics_per_token*.ipynb`, `Evaluation_mix.ipynb`. Load the `.pkl` reports, align each `B>1` run against the `B=1` baseline, and for each of the **top-K (10)** baseline tokens at each step compute standard deviation (σ, biased / `1/r`) and range (max−min) of the probability across the 50 runs. Output is a flat CSV: `[Model, BS, t, Rank, Token_ID, Prob_B1, Mean_Prob_Runs, Std_Prob_Runs, Range_Prob_Runs]`, saved as `stability_token_level_report_STD_RANGE_{GPU}.csv`. One CSV per GPU/hardware.

3. **Plotting** — `Plot_paper.ipynb` (final paper figures, organized as `#Fig1`, `#Fig2`… cells), plus `Plot_*.ipynb`, `Print_results.ipynb`, `Analysis_Huawei.ipynb`. These read the per-GPU CSVs and write PNGs into `Figures*/`. `moving_average.py` provides a centered-window smoother used by per-token trace plots.

## Conventions that matter

- **Directory naming is the contract between stages.** Reports are `mix_stability_reports_{model}_B{bs}`; evaluation/plot notebooks hard-code `REPORT_DIR_TEMPLATE = "mix_stability_reports_{model}_B{bs}"` and the model list `["gemma3","llama3.2","qwen3","deepseek_qwen3"]` (gemma size variants: `gemma3_270M`, `gemma3_1B`, `gemma3_4B`). Renaming a directory breaks downstream notebooks silently (they `continue` past missing dirs).
- **Display model names** are mapped in plotting notebooks: `gemma3→Gemma3-12B`, `llama3.2→Llama3.2-11B`, `qwen3→Qwen3-VL-8B`, `deepseek_qwen3→DeepSeek-Qwen3-8B`.
- **Hardware is encoded in CSV filenames**, not in the data: `..._STD_RANGE_{A100,A6000,H200,Ascend-910}.csv` (Huawei == Ascend-910). The same generation/eval code is re-run on each machine and results are compared by file.
- **`mmlu_10_random_samples.jsonl`** is the fixed target question set (10 questions → `question_000`..`question_009`); `mmlu_1000_random_samples_filler.jsonl` is the noise pool. `Download_MMLU.ipynb` regenerates these from the MMLU dataset.
- Generation skips already-present `run_*.pkl` files, so notebooks are resumable — delete a report dir to force re-generation.
- Generation requires a CUDA GPU and downloads large gated HF models (gemma-3-12b-it, Llama-3.2-11B-Vision, Qwen3-VL-8B, DeepSeek-R1-0528-Qwen3-8B) in `bfloat16`.

## Ignore for most tasks

`deprecated/`, `deprecated_results/`, `deprecated_script/`, `tmp/`, `*.zip` archives, `__pycache__/`, and the multi-GB `.pkl`/`.csv` data artifacts are outputs/history, not source. The live source is the `.ipynb` notebooks and `moving_average.py`.

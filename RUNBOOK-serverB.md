# RUNBOOK — Server B (H100): GSM8K side of the llama31 1000-question protocol

You are Claude Code on server B. Server A runs the MMLU side of every arm; you
run the **GSM8K side of every arm**. All claim tests are within-benchmark, so
your outputs never mix hardware with server A's. File names are disjoint from
server A's (`*_gsm8k.jsonl` vs `*_mmlu.jsonl`) — git merges cleanly.

## 0. One-time setup

```bash
git clone <REPO_URL> Nondeterm-compression && cd Nondeterm-compression
python3.11 -m venv .venv
export PIP_USER=0
.venv/bin/pip install -r requirements-eval.lock   # torch 2.13.0 (cu130 wheel from default pypi), transformers 5.15.1
# HF login (llama is gated; also needed for the private ckpt repo):
.venv/bin/python -c "from huggingface_hub import login; login()"  # or hf auth login
```

Download the frozen modelopt checkpoints (private repo, ~70 GB — see
`MODELOPT_CKPT_REPO` below; server A fills this in before pushing):

```bash
for K in 14080 13568 12800 12288 11520; do
  .venv/bin/hf download <MODELOPT_CKPT_REPO> --include "llama31_modelopt_keep${K}/*" \
      --local-dir acc_protocol/models/
done
# verify: sha256 of 2 sampled shards must match acc_protocol/models_sha256.txt
sha256sum acc_protocol/models/llama31_modelopt_keep12800/model-00001-of-*.safetensors
```

The topiary orders file `acc_protocol/runs/orders_mixed_llama31.pt` is already
in the repo (frozen artifact — do NOT recalibrate).

## 1. Audit run (~10 min, informational)

```bash
.venv/bin/python acc_protocol/generate.py --model llama31 --arm acc1 \
    --n 1000 --variant h100 --limit 20 --suffix _audit --max-new-tokens 1024
git add acc_protocol/outputs && git commit -m "serverB audit" && git push
```

(Server A diffs the two audits token-by-token; result goes in findings. The
protocol does not depend on it matching.)

## 2. Main matrix — GSM8K 1000 for all 21 arms (~13 h, run sequentially)

```bash
V="--n 1000 --variant h100 --benchmarks gsm8k --max-new-tokens 1024"
P=.venv/bin/python

$P acc_protocol/generate.py --model llama31 --arm acc1 $V
$P acc_protocol/generate.py --model llama31 --arm acc2 --keep 14080 13568 12800 12288 11520 $V
$P acc_protocol/generate.py --model llama31 --arm acc2ckpt --ckpt-dir \
    acc_protocol/models/llama31_modelopt_keep14080 \
    acc_protocol/models/llama31_modelopt_keep13568 \
    acc_protocol/models/llama31_modelopt_keep12800 \
    acc_protocol/models/llama31_modelopt_keep12288 \
    acc_protocol/models/llama31_modelopt_keep11520 $V
$P acc_protocol/generate.py --model llama31 --arm acc3 --grouping 0 1 2 3 4 5 6 7 8 9 \
    --batch-size 16 $V
```

Notes:
- Everything is resumable (existing ids skipped) — just rerun a command after
  any interruption.
- Outputs land in `acc_protocol/outputs/llama31_h100/*_gsm8k.jsonl`, 1000
  lines each when done (`wc -l` to check).
- Push results after each arm finishes:
  `git pull --rebase && git add acc_protocol/outputs && git commit -m "serverB: <arm>" && git push`
- Sanity: greedy BF16, cap 1024; expect trunc% ≲3 and extract-fail ~0 on GSM8K
  (`python acc_protocol/score.py --model llama31 --variant h100 --n 1000` shows
  a partial table with only gsm8k columns populated until server A's files merge).

## 3. Optional (only if you finish before server A)

Acc3 at B=32, GSM8K side:

```bash
$P acc_protocol/generate.py --model llama31 --arm acc3 --grouping 0 1 2 3 4 5 6 7 8 9 \
    --batch-size 32 $V
```

## Do NOT

- Do not recompute salience/orders or rerun puzzletron pruning on this machine
  (the pruned models are frozen artifacts shared with server A).
- Do not run any `--benchmarks mmlu` generation (that side belongs to server A).
- Do not edit files under `acc_protocol/outputs/llama31/` (A100-era data).

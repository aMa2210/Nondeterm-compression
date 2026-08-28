# RUNBOOK — Server B (H100): GSM8K side of the llama31 1000-question protocol

You are Claude Code on server B. Server A runs the MMLU side of every arm; you
run the **GSM8K side of every arm**. All claim tests are within-benchmark, so
your outputs never mix hardware with server A's. File names are disjoint from
server A's (`*_gsm8k.jsonl` vs `*_mmlu.jsonl`) — git merges cleanly.

## 0. One-time setup

```bash
git clone https://github.com/aMa2210/Nondeterm-compression.git && cd Nondeterm-compression
git config --global --add safe.directory "$PWD"
git config user.name "Tairan" && git config user.email "tkm199888@gmail.com"
git config credential.helper 'cache --timeout=604800'
python3.11 -m venv .venv
export PIP_USER=0
.venv/bin/pip install -r requirements-eval.lock   # torch 2.13.0 (cu130 wheel from default pypi), transformers 5.15.1
# HF login — MUST use the AmA-2025 account token (the user provides it; it has
# gated-llama access AND owns the private ckpt repo; other accounts get 403/404):
.venv/bin/python -c "from huggingface_hub import login; login()"  # or hf auth login
.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"  # must print AmA-2025
```

Expected GPU: H100 NVL **MIG 3g.47gb** — the SAME MIG profile as server A
(server A is not a full card either), so noise floors are comparable and
`--variant h100` is the shared label for this hardware pair. If `nvidia-smi -L`
shows a different profile, STOP and report before running anything.

Download the frozen modelopt checkpoints (private HF repo, ~70 GB):

```bash
for K in 14080 13568 12800 12288 11520; do
  .venv/bin/hf download AmA-2025/llama31-modelopt-pruned-ffn --include "llama31_modelopt_keep${K}/*" \
      --local-dir acc_protocol/models/
done
# verify: sha256 of 2 sampled shards must match acc_protocol/models_sha256.txt
# (NB: keep13568 and keep14080 legitimately share an identical LAST shard —
#  it holds only the unpruned lm_head/norm; not a copy error)
sha256sum acc_protocol/models/llama31_modelopt_keep12800/model-00001-of-*.safetensors
```

The topiary orders file `acc_protocol/runs/orders_mixed_llama31.pt` is already
in the repo (frozen artifact — do NOT recalibrate).

## 1. Audit run (~10 min, informational)

```bash
.venv/bin/python acc_protocol/generate.py --model llama31 --arm acc1 \
    --n 1000 --variant h100 --limit 20 --suffix _auditB --max-new-tokens 1024
git pull --rebase && git add acc_protocol/outputs && git commit -m "serverB audit" && git push
```

Notes: the audit deliberately runs BOTH benchmarks — it exists to diff the two
machines on identical questions, and is the one exception to the
"no `--benchmarks mmlu`" rule below. The suffix is `_auditB` (server A's audit
files are `acc1_audit_*`), so filenames stay disjoint. `_audit*` files are
excluded from scoring automatically. The protocol does not depend on the diff
matching.

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

Acc3 at B=32, GSM8K side (B=16 is the primary noise floor — deliberate, for
comparability with the gemma3/A100 250q protocol; PLAN.md's B≥32 is this
optional extra). `score.py`/`compare_backends.py` select acc3 arms by
`--acc3-bs` (default 16), so B=32 arms never pollute the B=16 thresholds:

```bash
$P acc_protocol/generate.py --model llama31 --arm acc3 --grouping 0 1 2 3 4 5 6 7 8 9 \
    --batch-size 32 $V
```

## Do NOT

- Do not recompute salience/orders or rerun puzzletron pruning on this machine
  (the pruned models are frozen artifacts shared with server A).
- Do not run any `--benchmarks mmlu` generation (that side belongs to server A).
- Do not edit files under `acc_protocol/outputs/llama31/` (A100-era data).

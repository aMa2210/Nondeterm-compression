# RUNBOOK — Server B, Phase 2: GSM8K sides of gemma3-12b and qwen3_14b (n=1000)

Advisor request: run **google/gemma-3-12b-it** and **OpenPipe/Qwen3-14B-Instruct**
through the same n=1000 protocol. Same division of labor as phase 1:
**server A = all MMLU, server B (you) = all GSM8K**, `--variant h100` everywhere.
Setup (venv, HF login as AmA-2025, git identity) carries over from
`RUNBOOK-serverB.md` — do not redo it.

Backend coverage per model:
- **gemma3**: topiary only (Model-Optimizer/puzzletron does not support Gemma).
- **qwen3_14b**: BOTH backends (Qwen3 is puzzletron-supported; server A produces
  the checkpoints).

## Frozen artifacts — dependency rules (IMPORTANT)

- gemma3 topiary orders: `acc_protocol/runs/orders_mixed_gemma3.pt` — already in
  the repo (A100-era, frozen). Never recompute.
- qwen3_14b topiary orders: `acc_protocol/runs/orders_mixed_qwen3_14b.pt` —
  server A computes and pushes it. **Do NOT run any qwen3_14b acc2 arm until
  `git pull` delivers this file** (if you run acc2 first, generate.py would
  calibrate locally and produce different pruned models — protocol violation).
- qwen3_14b modelopt checkpoints: HF repo
  `AmA-2025/qwen3-14b-modelopt-pruned-ffn` (server A uploads, ~130 GB total).
  Wait until server A's commit says they are up, then:
  ```bash
  for K in 17152 16384 15616 14848 14080; do
    .venv/bin/hf download AmA-2025/qwen3-14b-modelopt-pruned-ffn \
        --include "qwen3_14b_modelopt_keep${K}/*" --local-dir acc_protocol/models/
  done
  # verify against acc_protocol/models_sha256_qwen3_14b.txt (first+last shard per width)
  ```
- gemma3 has NO dependencies — start it immediately.

## Run order (all commands from repo root; push after each finished arm)

```bash
V="--n 1000 --variant h100 --benchmarks gsm8k --max-new-tokens 1024"
P=.venv/bin/python
```

### Stage 1 — gemma3 GSM8K (start now; ~20 h)

```bash
$P acc_protocol/generate.py --model gemma3 --arm acc1 $V
$P acc_protocol/generate.py --model gemma3 --arm acc2 --keep 15040 14592 13824 13056 12288 $V
$P acc_protocol/generate.py --model gemma3 --arm acc3 --grouping 0 1 2 3 4 5 6 7 8 9 --batch-size 16 $V
```

(keep levels = the 250q-run levels 2.1/5/10/15% plus 20%; inter=15360.)

### Stage 2 — qwen3_14b GSM8K, no-dependency arms (~17 h)

```bash
$P acc_protocol/generate.py --model qwen3_14b --arm acc1 $V
$P acc_protocol/generate.py --model qwen3_14b --arm acc3 --grouping 0 1 2 3 4 5 6 7 8 9 --batch-size 16 $V
```

### Stage 3 — qwen3_14b GSM8K, pruned arms (AFTER the frozen artifacts arrive)

```bash
git pull --rebase --autostash   # must bring runs/orders_mixed_qwen3_14b.pt
$P acc_protocol/generate.py --model qwen3_14b --arm acc2 --keep 17152 16384 15616 14848 14080 $V
$P acc_protocol/generate.py --model qwen3_14b --arm acc2ckpt --ckpt-dir \
    acc_protocol/models/qwen3_14b_modelopt_keep17152 \
    acc_protocol/models/qwen3_14b_modelopt_keep16384 \
    acc_protocol/models/qwen3_14b_modelopt_keep15616 \
    acc_protocol/models/qwen3_14b_modelopt_keep14848 \
    acc_protocol/models/qwen3_14b_modelopt_keep14080 $V
```

(keep levels = 1.5/5.9/10.3/14.7/19.1% of inter=17408, all divisible by 256.)

## Notes

- Everything resumable; rerun any command after interruption.
- Expected VRAM: gemma3-12b ≈24 GB weights, qwen3-14b ≈28 GB — both fit the
  47 GB MIG at B=16 with cap 1024.
- OpenPipe/Qwen3-14B-Instruct has no thinking blocks in its chat template;
  generations behave like llama/gemma (verified on server A).
- Sync etiquette unchanged: only write `*_gsm8k.jsonl`; never touch mmlu files;
  `git pull --rebase --autostash` before every push.
- Total ≈ 37 h + stage-3 wait; report per-arm completion in commit messages.

## Fallback if the HF checkpoint download is unavailable (rate limits)

The qwen3-14b checkpoint upload hit HuggingFace's 1000-req/5-min quota. If
`hf download` still cannot fetch `AmA-2025/qwen3-14b-modelopt-pruned-ffn`,
reconstruct the checkpoints locally instead — this is **exact, not an
approximation**: puzzletron's stage 3 (channel-importance scoring) is the only
data-dependent step, and its output is committed to this repo
(`modelopt_prune/pruning_scores_qwen3_14b/`, 16 MB). Stage 4 is deterministic
weight slicing given those scores, and `launch_score_activations` skips itself
when complete scores are present.

**VERIFIED EMPIRICALLY on server A (2026-09-02), not just claimed.** A fresh
`puzzle_dir` seeded only with the committed 16 MB scores was run through
stages 2+4 and repackaged; the log confirmed `Scoring 100% completed,
skipping...` and both sampled shards of keep14080 matched the original
byte-for-byte:

    29b91226ecbb26d7df535074...  model-00001-of-00006.safetensors  (both)
    b4bf668aa6f8535dd467a9a3...  model-00006-of-00006.safetensors  (both)

Whole run took 2.5 min for one width. The sha256 gate below is a safety
check on YOUR environment (different library versions could in principle
differ), not an admission that the procedure is approximate.

```bash
# 1. modelopt venv (same as server A's setup)
python3.11 -m venv .venv-modelopt && export PIP_USER=0
git clone --depth 1 https://github.com/NVIDIA/Model-Optimizer.git modelopt-src
.venv-modelopt/bin/pip install torch "transformers>=4.57,<5.0"
cd modelopt-src && PIP_USER=0 ../.venv-modelopt/bin/pip install -e ".[hf,puzzletron]" \
    && PIP_USER=0 ../.venv-modelopt/bin/pip install -r examples/puzzletron/requirements.txt && cd ..

# 2. point the config at YOUR local paths, then seed the scores so stage 3 is skipped
#    (edit input_hf_model_path / dataset_path / puzzle_dir in
#     modelopt_prune/config_qwen3_14b/qwen3_14b_pruneffn_memory.yaml)
mkdir -p <puzzle_dir>/pruning/pruning_scores
cp -r modelopt_prune/pruning_scores_qwen3_14b/* <puzzle_dir>/pruning/pruning_scores/

# 3. run stages 2+4 (stage 3 auto-skips), then repackage
PIP_USER=0 .venv-modelopt/bin/torchrun --nproc_per_node 1 \
    modelopt_prune/run_prune_ckpts.py --config modelopt_prune/config_qwen3_14b/qwen3_14b_pruneffn_memory.yaml
for K in 17152 16384 15616 14848 14080; do
  PIP_USER=0 .venv-modelopt/bin/python modelopt_prune/repackage_to_hf.py \
      --child-dir <puzzle_dir>/ckpts/ffn_${K}_attn_no_op \
      --teacher-dir <OpenPipe snapshot dir> \
      --out-dir acc_protocol/models/qwen3_14b_modelopt_keep${K}
done

# 4. VERIFY before using: hashes must match acc_protocol/models_sha256_qwen3_14b.txt
cd acc_protocol/models && sha256sum qwen3_14b_modelopt_keep14080/model-00001-of-*.safetensors
```

If the hashes do NOT match, stop and report — do not run the arms with
divergent models; the protocol requires both servers to score the identical
pruned checkpoints.

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

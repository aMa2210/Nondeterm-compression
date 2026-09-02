"""Repackage a puzzletron uniform-width child checkpoint into a vanilla HF
Llama checkpoint loadable by any transformers version.

The stage-4 child ckpt ({puzzle_dir}/ckpts/ffn_{K}_attn_no_op/) has standard
Llama weight names but a patched config (per-block block_configs). Since every
layer has the same FFN width K, the model is exactly a LlamaConfig with
intermediate_size=K.

Usage (modelopt venv):
    python modelopt_prune/repackage_to_hf.py \
        --child-dir modelopt_prune/puzzle_dir/ckpts/ffn_12800_attn_no_op \
        --teacher-dir <HF snapshot of Llama-3.1-8B-Instruct> \
        --out-dir acc_protocol/models/llama31_modelopt_keep12800
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-dir", required=True)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    state = {}
    shards = sorted(
        glob.glob(os.path.join(args.child_dir, "*.safetensors"))
        + glob.glob(os.path.join(args.child_dir, "subblocks_safetensors",
                                 "*.safetensors")))
    assert shards, f"no safetensors in {args.child_dir}"
    for s in shards:
        state.update(load_file(s))

    # infer uniform width from the pruned gate_proj rows, verify all layers agree
    gate_keys = [k for k in state if k.endswith("mlp.gate_proj.weight")]
    widths = {state[k].shape[0] for k in gate_keys}
    assert len(widths) == 1, f"non-uniform widths: {widths}"
    k_width = widths.pop()
    print(f"[repack] {len(gate_keys)} layers, uniform intermediate_size={k_width}")

    with open(os.path.join(args.teacher_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg["intermediate_size"] = int(k_width)
    cfg["torch_dtype"] = "bfloat16"

    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.for_model(**cfg)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    state = {k: v.to(torch.bfloat16) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    # Every non-weight file from the teacher: tokenizer, generation config, and
    # anything else the model needs. A fixed filename list is not enough — Qwen3
    # keeps its chat template in chat_template.jinja and its BPE tables in
    # vocab.json/merges.txt, none of which Llama has.
    skip = {"config.json", "model.safetensors.index.json"}
    for name in sorted(os.listdir(args.teacher_dir)):
        src = os.path.join(args.teacher_dir, name)
        if (not os.path.isfile(src) or name in skip
                or name.endswith(".safetensors") or name.endswith(".bin")):
            continue
        shutil.copy(src, os.path.join(args.out_dir, name))
        print(f"[repack] copied {name}")
    n_params = sum(v.numel() for v in state.values())
    print(f"[repack] saved {args.out_dir} ({n_params/1e9:.2f}B params)")


if __name__ == "__main__":
    main()

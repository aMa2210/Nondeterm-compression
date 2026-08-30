"""Quick logprob-level test: pruned-vs-full at B=1 (greedy, BF16).

Replicates the prior paper's generation harness (see
reference_code/gemma3_Mix_lighter_version.ipynb) at B=1 on the 10 fixed MMLU
target questions, for the full model and a sweep of topiary-style width
truncations. Output pkls share the reference format:
    {out_root}/{tag}/question_{idx:03d}/run_00.pkl
    -> {"tokens": LongTensor, "logits": [sparse top-15 logits per step]}

Calibration traffic = the MMLU filler pool (chat-templated), i.e. the same
distribution the batching experiment serves — the dense analogue of topiary's
routed "real traffic" salience.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topiary_dense import (GROUP, calibrate_salience, orders_from_salience,
                           restore_mlp_weights, snapshot_mlp_weights,
                           truncate_in_place)

MODELS = {
    "deepseek": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek_qwen3"),
    "gemma3": ("google/gemma-3-12b-it", "gemma3"),
    "llama31": ("meta-llama/Llama-3.1-8B-Instruct", "llama31"),
    "qwen3_14b": ("OpenPipe/Qwen3-14B-Instruct", "qwen3_14b"),
}
MODEL_ID, MODEL_TAG = MODELS["deepseek"]  # overridden by --model
SEED = 83
MAX_GEN_TOKENS = 500
TOP_K_STORE = 15
KEEP_FRACS = [0.995, 0.99, 0.98, 0.95, 0.90, 0.85]
CALIB_PROMPTS = 256
CALIB_MAX_LEN = 512

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT_ROOT = os.path.join(ROOT, "quick_test", "reports")
RUNS_DIR = os.path.join(ROOT, "quick_test", "runs")


class StoreTargetLogitsProcessor(LogitsProcessor):
    """Store top-15 logits of the target row each decode step (reference class)."""

    def __init__(self, target_index, top_k=TOP_K_STORE):
        self.target_index = target_index
        self.stored_logits = []
        self.top_k = top_k

    def __call__(self, input_ids, scores):
        target_scores = scores[self.target_index, :].detach().to("cpu")
        vocab_size = target_scores.shape[-1]
        top_values, top_indices = torch.topk(target_scores, self.top_k)
        sparse_logit = torch.sparse_coo_tensor(
            top_indices.unsqueeze(0), top_values, size=(vocab_size,))
        self.stored_logits.append(sparse_logit)
        return scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_prompt(sample):
    q, c = sample["question"], sample["choices"]
    return (f"{q}\n\nChoices:\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
            f"Please think step by step and then give the final answer.\nAnswer:")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def generate_b1(model, tokenizer, targets, tag):
    save_root = os.path.join(OUT_ROOT, tag)
    for idx, sample in enumerate(targets):
        q_dir = os.path.join(save_root, f"question_{idx:03d}")
        os.makedirs(q_dir, exist_ok=True)
        save_path = os.path.join(q_dir, "run_00.pkl")
        if os.path.exists(save_path):
            print(f"  [skip] {save_path}", flush=True)
            continue
        messages = [{"role": "user", "content": format_prompt(sample)}]
        inputs = tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        proc = StoreTargetLogitsProcessor(target_index=0)
        t0 = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_GEN_TOKENS, output_scores=False,
            return_dict_in_generate=True, do_sample=False,
            temperature=None, top_p=None, logits_processor=[proc],
        )
        tokens_cpu = outputs.sequences[0][prompt_length:].to("cpu")
        with open(save_path, "wb") as f:
            pickle.dump({"tokens": tokens_cpu, "logits": proc.stored_logits}, f)
        print(f"  [gen] {tag} q{idx:03d}: {len(tokens_cpu)} tokens "
              f"in {time.time()-t0:.0f}s", flush=True)
        del outputs, proc
        torch.cuda.empty_cache()


def calib_batches(tokenizer, fillers, n_prompts, max_len):
    rng = random.Random(SEED)
    picks = rng.sample(fillers, n_prompts)
    for sample in picks:
        messages = [{"role": "user", "content": format_prompt(sample)}]
        ids = tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")["input_ids"][:, :max_len]
        yield ids


def main():
    global MODEL_ID, MODEL_TAG
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="deepseek")
    parser.add_argument("--keep-fracs", type=float, nargs="*", default=KEEP_FRACS)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--ckpt-dir", nargs="*", default=[],
                        help="externally-pruned HF checkpoint dir(s); each is run "
                             "at B=1 under tag = dir basename (skips the topiary "
                             "sweep unless --keep-fracs is also given)")
    args = parser.parse_args()
    MODEL_ID, MODEL_TAG = MODELS[args.model]

    if args.ckpt_dir:
        os.makedirs(OUT_ROOT, exist_ok=True)
        targets = load_jsonl(os.path.join(DATA, "mmlu_10_random_samples.jsonl"))
        for d in args.ckpt_dir:
            tag = os.path.basename(os.path.normpath(d))
            print(f"[load] {d} (bf16, cuda)", flush=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(d)
            except (OSError, ValueError):
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                d, dtype=torch.bfloat16, device_map="cuda")
            model.eval()
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            set_seed(SEED)
            generate_b1(model, tokenizer, targets, tag)
            del model
            torch.cuda.empty_cache()
        print("[done]", flush=True)
        return

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    set_seed(SEED)

    print(f"[load] {MODEL_ID} (bf16, cuda)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    targets = load_jsonl(os.path.join(DATA, "mmlu_10_random_samples.jsonl"))
    fillers = load_jsonl(os.path.join(DATA, "mmlu_1000_random_samples_filler.jsonl"))
    from topiary_dense import decoder_layers
    inter = decoder_layers(model)[0].mlp.down_proj.in_features
    n_layers = len(decoder_layers(model))
    print(f"[model] {n_layers} layers, intermediate={inter}", flush=True)

    if not args.skip_baseline:
        print("[baseline] full model, B=1", flush=True)
        generate_b1(model, tokenizer, targets, f"{MODEL_TAG}_full")

    orders_path = os.path.join(RUNS_DIR, f"orders_{MODEL_TAG}.pt")
    if os.path.exists(orders_path):
        print(f"[salience] reusing {orders_path}", flush=True)
        orders = torch.load(orders_path)
    else:
        print(f"[salience] calibrating on {CALIB_PROMPTS} filler prompts", flush=True)
        t0 = time.time()
        salience, n_tok = calibrate_salience(
            model, calib_batches(tokenizer, fillers, CALIB_PROMPTS, CALIB_MAX_LEN))
        orders = orders_from_salience(salience)
        torch.save(orders, orders_path)
        torch.save(salience, os.path.join(RUNS_DIR, f"salience_{MODEL_TAG}.pt"))
        print(f"[salience] {n_tok} tokens in {time.time()-t0:.0f}s -> {orders_path}",
              flush=True)

    print("[snapshot] caching original MLP weights on CPU", flush=True)
    snap = snapshot_mlp_weights(model)

    for frac in args.keep_fracs:
        k = int(round(frac * inter / GROUP)) * GROUP
        tag = f"{MODEL_TAG}_keep{k}"
        print(f"[prune] keep={k}/{inter} ({100*k/inter:.1f}%)", flush=True)
        restore_mlp_weights(model, snap)
        truncate_in_place(model, orders, k)
        set_seed(SEED)
        generate_b1(model, tokenizer, targets, tag)

    restore_mlp_weights(model, snap)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

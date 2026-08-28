"""Generation engine for the accuracy protocol (audit + full runs).

Arms:
  --arm acc1               B=1, full model
  --arm acc2 --keep K      B=1, model truncated to K FFN neurons (topiary port,
                           mixed MMLU+GSM8K calibration)
  --arm acc3 --grouping i  B=BS, full model, question order shuffled with seed
                           SEED+i, chunked into fixed-size batches (last batch
                           padded with filler prompts whose outputs are dropped)

Outputs one jsonl per (arm, benchmark): acc_protocol/outputs/{tag}_{bench}.jsonl
  {id, prompt_tokens, gen_tokens, finished, text}
Resumable: existing ids are skipped and the file is appended to.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "quick_test"))
from common import build_prompt, load_jsonl  # noqa: E402
from topiary_dense import (GROUP, calibrate_salience, orders_from_salience,  # noqa: E402
                           restore_mlp_weights, snapshot_mlp_weights,
                           truncate_in_place)

MODELS = {
    "deepseek": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek_qwen3"),
    "gemma3": ("google/gemma-3-12b-it", "gemma3"),
    "llama31": ("meta-llama/Llama-3.1-8B-Instruct", "llama31"),
}
MODEL_ID, MODEL_TAG = MODELS["deepseek"]  # overridden by --model
SEED = 83
DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")  # per-model subdir set in main()
RUNS = os.path.join(HERE, "runs")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chat_ids(tokenizer, prompts):
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True)


def load_model(path=None):
    """Load MODEL_ID, or a (pruned) checkpoint dir via path (tokenizer falls
    back to MODEL_ID if the checkpoint dir has none)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(path or MODEL_ID)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        path or MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def mixed_orders(model, tokenizer):
    """Salience orders from the mixed MMLU+GSM8K calibration corpus (cached)."""
    path = os.path.join(RUNS, f"orders_mixed_{MODEL_TAG}.pt")
    if os.path.exists(path):
        return torch.load(path)
    calib = load_jsonl(os.path.join(DATA, "calib_mixed.jsonl"))

    def batches():
        for s in calib:
            ids = chat_ids(tokenizer, [build_prompt(s)])["input_ids"][:, :512]
            yield ids

    t0 = time.time()
    salience, n_tok = calibrate_salience(model, batches())
    orders = orders_from_salience(salience)
    os.makedirs(RUNS, exist_ok=True)
    torch.save(orders, path)
    torch.save(salience, os.path.join(RUNS, f"salience_mixed_{MODEL_TAG}.pt"))
    print(f"[calib] {n_tok} tokens in {time.time()-t0:.0f}s -> {path}", flush=True)
    return orders


def eos_ids(model, tokenizer):
    """All stop-token ids (generation_config may list several, e.g. gemma3's
    <eos> and <end_of_turn>)."""
    ids = model.generation_config.eos_token_id
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    ids = {i for i in ids if i is not None}
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens):
    inputs = chat_ids(tokenizer, prompts).to(model.device)
    plen = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, return_dict_in_generate=True)
    seqs = out.sequences[:, plen:].to("cpu")
    stop_ids = eos_ids(model, tokenizer)
    results = []
    for row in seqs:
        toks = row.tolist()
        stop_pos = [i for i, t in enumerate(toks) if t in stop_ids]
        finished = bool(stop_pos)
        if finished:
            # everything after the first stop token is padding (pad may equal eos)
            toks = toks[: stop_pos[0] + 1]
        elif tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
            toks = [t for t in toks if t != tokenizer.pad_token_id]
        results.append({
            "prompt_tokens": int(plen),
            "gen_tokens": len(toks),
            "finished": finished,
            "text": tokenizer.decode(toks, skip_special_tokens=True),
        })
    del out, inputs
    return results


def run_b1(model, tokenizer, samples, out_path, max_new_tokens):
    done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
    with open(out_path, "a", encoding="utf-8") as f:
        for s in samples:
            if s["id"] in done:
                continue
            t0 = time.time()
            r = generate_batch(model, tokenizer, [build_prompt(s)], max_new_tokens)[0]
            r["id"] = s["id"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [gen] {s['id']}: {r['gen_tokens']} tok "
                  f"({'eos' if r['finished'] else 'CAP'}) in {time.time()-t0:.0f}s",
                  flush=True)


def run_b32(model, tokenizer, samples, fillers, out_path, max_new_tokens,
            batch_size, grouping):
    done = {json.loads(l)["id"] for l in open(out_path)} if os.path.exists(out_path) else set()
    order = list(range(len(samples)))
    random.Random(SEED + grouping).shuffle(order)
    filler_rng = random.Random(SEED * 1000 + grouping)
    with open(out_path, "a", encoding="utf-8") as f:
        for b0 in range(0, len(order), batch_size):
            chunk = [samples[i] for i in order[b0:b0 + batch_size]]
            if all(s["id"] in done for s in chunk):
                continue
            pad_n = batch_size - len(chunk)
            pads = [{"benchmark": "mmlu", **filler_rng.choice(fillers)}
                    for _ in range(pad_n)]
            prompts = [build_prompt(s) for s in chunk + pads]
            t0 = time.time()
            results = generate_batch(model, tokenizer, prompts, max_new_tokens)
            for s, r in zip(chunk, results):  # drop filler outputs
                if s["id"] in done:
                    continue
                r["id"] = s["id"]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [gen] batch @{b0} ({len(chunk)} real): "
                  f"max {max(r['gen_tokens'] for r in results)} tok "
                  f"in {time.time()-t0:.0f}s", flush=True)


def main():
    global MODEL_ID, MODEL_TAG, OUTPUTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="deepseek")
    ap.add_argument("--arm", required=True,
                    choices=["acc1", "acc2", "acc3", "acc2ckpt"])
    ap.add_argument("--keep", type=int, nargs="*",
                    help="acc2: FFN neurons kept per layer (one or more levels)")
    ap.add_argument("--ckpt-dir", nargs="*",
                    help="acc2ckpt: pruned HF checkpoint dir(s); arm tag = "
                         "acc2_<basename minus '<model_tag>_' prefix>")
    ap.add_argument("--grouping", type=int, nargs="*",
                    help="acc3: grouping indices (0..9)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--benchmarks", nargs="*", default=["mmlu", "gsm8k"])
    ap.add_argument("--limit", type=int, help="first N questions only (audit)")
    ap.add_argument("--suffix", default="", help="output filename suffix")
    ap.add_argument("--n", type=int, default=250,
                    help="eval set size: reads data/{bench}_{n}.jsonl")
    ap.add_argument("--variant", default="",
                    help="hardware/config variant; outputs go to "
                         "outputs/{model_tag}_{variant}/")
    args = ap.parse_args()

    MODEL_ID, MODEL_TAG = MODELS[args.model]
    # variant only relocates outputs; MODEL_TAG stays the base tag so frozen
    # artifacts (orders_mixed_{tag}.pt, ckpt-dir basenames) resolve unchanged
    out_tag = f"{MODEL_TAG}_{args.variant}" if args.variant else MODEL_TAG
    OUTPUTS = os.path.join(OUTPUTS, out_tag)
    os.makedirs(OUTPUTS, exist_ok=True)
    model, tokenizer = (None, None) if args.arm == "acc2ckpt" else load_model()

    fillers = load_jsonl(os.path.join(ROOT, "data",
                                      "mmlu_1000_random_samples_filler.jsonl"))

    def gen_all(tag, grouping=None):
        for bench in args.benchmarks:
            samples = load_jsonl(os.path.join(DATA, f"{bench}_{args.n}.jsonl"))
            if args.limit:
                samples = samples[: args.limit]
            out_path = os.path.join(OUTPUTS, f"{tag}_{bench}.jsonl")
            print(f"[run] {tag} / {bench}: {len(samples)} questions "
                  f"(cap {args.max_new_tokens})", flush=True)
            set_seed(SEED)
            if grouping is not None:
                run_b32(model, tokenizer, samples, fillers, out_path,
                        args.max_new_tokens, args.batch_size, grouping)
            else:
                run_b1(model, tokenizer, samples, out_path, args.max_new_tokens)

    if args.arm == "acc2ckpt":
        assert args.ckpt_dir, "--ckpt-dir required for acc2ckpt"
        for d in args.ckpt_dir:
            base = os.path.basename(os.path.normpath(d))
            name = base[len(MODEL_TAG) + 1:] if base.startswith(MODEL_TAG + "_") else base
            model, tokenizer = load_model(d)
            print(f"[ckpt] {base}", flush=True)
            gen_all(f"acc2_{name}{args.suffix}")
            del model
            model = None
            torch.cuda.empty_cache()
    elif args.arm == "acc2":
        assert args.keep, "--keep required for acc2"
        orders = mixed_orders(model, tokenizer)
        snap = snapshot_mlp_weights(model)
        for keep in args.keep:
            restore_mlp_weights(model, snap)
            truncate_in_place(model, orders, keep)
            print(f"[prune] keep={keep}", flush=True)
            gen_all(f"acc2_keep{keep}{args.suffix}")
    elif args.arm == "acc3":
        assert args.grouping, "--grouping required for acc3"
        for g in args.grouping:
            gen_all(f"acc3_b{args.batch_size}_g{g:02d}{args.suffix}", grouping=g)
    else:
        gen_all(f"acc1{args.suffix}")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

"""Shared pieces of the accuracy protocol: dataset formatting, prompt
construction, and tiered answer extraction.

Prompt style follows DeepSeek-R1's recommended usage (final answer inside
\\boxed{}), which makes extraction near-unambiguous. Extraction is tiered and
the tier used is recorded, so extraction-failure rates can be reported per arm.
"""

from __future__ import annotations

import json
import re

LETTERS = "ABCD"


# ------------------------------------------------------------------- prompts


def mmlu_prompt(sample):
    q, c = sample["question"], sample["choices"]
    return (f"{q}\n\nChoices:\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
            "Please reason step by step, and put your final answer "
            "(only the letter A, B, C or D) within \\boxed{}.")


def gsm8k_prompt(sample):
    return (f"{sample['question']}\n\n"
            "Please reason step by step, and put your final answer "
            "(a single number) within \\boxed{}.")


def build_prompt(sample):
    return mmlu_prompt(sample) if sample["benchmark"] == "mmlu" else gsm8k_prompt(sample)


# ---------------------------------------------------------------- extraction


def post_think(text):
    """Text after the last </think>; the whole text if no think block closed."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def last_boxed(text):
    """Content of the last \\boxed{...}, handling one level of nested braces."""
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return matches[-1] if matches else None


def extract_mmlu(text):
    """-> (letter or None, tier). Tiers: boxed > answer-is > trailing-letter."""
    tail = post_think(text)
    boxed = last_boxed(tail)
    if boxed:
        m = re.search(r"[A-D]", boxed.upper())
        if m:
            return m.group(0), "boxed"
    m = re.findall(r"(?:answer\s+is|answer:)\s*\(?\**([A-D])\)?\**", tail,
                   flags=re.IGNORECASE)
    if m:
        return m[-1].upper(), "answer-is"
    m = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", tail)
    if m:
        return m[-1], "trailing-letter"
    return None, "fail"


_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _to_float(s):
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def extract_gsm8k(text):
    """-> (float or None, tier)."""
    tail = post_think(text)
    boxed = last_boxed(tail)
    if boxed:
        m = re.findall(_NUM, boxed)
        if m:
            v = _to_float(m[-1])
            if v is not None:
                return v, "boxed"
    m = re.findall(_NUM, tail)
    if m:
        v = _to_float(m[-1])
        if v is not None:
            return v, "last-number"
    return None, "fail"


def gsm8k_gold(answer_field):
    """Gold numeric answer from the '#### N' tail of the dataset answer."""
    return _to_float(answer_field.rsplit("####", 1)[1])


def score_sample(sample, gen_text):
    """-> dict(extracted, tier, correct)."""
    if sample["benchmark"] == "mmlu":
        pred, tier = extract_mmlu(gen_text)
        gold = LETTERS[sample["answer"]]
        correct = pred == gold
    else:
        pred, tier = extract_gsm8k(gen_text)
        gold = sample["gold"]
        correct = pred is not None and abs(pred - gold) < 1e-6
    return {"extracted": pred, "tier": tier, "correct": bool(correct),
            "gold": gold}


# ----------------------------------------------------------------------- I/O


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

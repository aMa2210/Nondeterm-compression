"""Sample the fixed evaluation subsets (250 MMLU + 250 GSM8K, seeded) and the
mixed calibration corpus (128 MMLU filler + 128 GSM8K train questions).

Outputs (acc_protocol/data/):
  mmlu_250.jsonl   {benchmark, id, question, choices, answer}
  gsm8k_250.jsonl  {benchmark, id, question, answer, gold}
  calib_mixed.jsonl {benchmark, question, ...}  (prompt-formatted at use time)
"""

import os
import random
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_jsonl, gsm8k_gold, load_jsonl

SEED = 83
N_EVAL = 250
N_CALIB_EACH = 128
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")


def extend_to(n_total, bench, existing_rows, pool_size, make_row, banned_idx=()):
    """Existing rows + (n_total - len) fresh indices sampled with SEED, keeping
    the existing file a strict prefix (ids encode dataset row indices)."""
    have = {int(r["id"].split("_")[1]) for r in existing_rows}
    rng = random.Random(SEED)
    candidates = [i for i in range(pool_size)
                  if i not in have and i not in set(banned_idx)]
    extra = rng.sample(candidates, n_total - len(existing_rows))
    rows = existing_rows + [make_row(i) for i in extra]
    assert len({r["id"] for r in rows}) == n_total
    dump_jsonl(rows, os.path.join(OUT, f"{bench}_{n_total}.jsonl"))
    print(f"[out] {bench}_{n_total}.jsonl ({len(rows)}, "
          f"{len(existing_rows)} reused)")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    mmlu_row = lambda i: {"benchmark": "mmlu", "id": f"mmlu_{i:05d}",
                          "question": mmlu[i]["question"],
                          "choices": mmlu[i]["choices"],
                          "answer": mmlu[i]["answer"]}
    path250 = os.path.join(OUT, "mmlu_250.jsonl")
    if os.path.exists(path250):
        rows = load_jsonl(path250)
    else:
        idx = rng.sample(range(len(mmlu)), N_EVAL)
        rows = [mmlu_row(i) for i in idx]
        dump_jsonl(rows, path250)
        print(f"[out] mmlu_250.jsonl ({len(rows)})")
    # 1000-question superset; keep Acc3 batch-filler questions out of the pool
    fillers = load_jsonl(os.path.join(ROOT, "data",
                                      "mmlu_1000_random_samples_filler.jsonl"))
    filler_q = {(s["question"], tuple(s["choices"])) for s in fillers}
    banned = [i for i in range(len(mmlu))
              if (mmlu[i]["question"], tuple(mmlu[i]["choices"])) in filler_q]
    extend_to(1000, "mmlu", rows, len(mmlu), mmlu_row, banned)

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_row = lambda i: {"benchmark": "gsm8k", "id": f"gsm8k_{i:05d}",
                         "question": gsm[i]["question"],
                         "answer": gsm[i]["answer"],
                         "gold": gsm8k_gold(gsm[i]["answer"])}
    path250 = os.path.join(OUT, "gsm8k_250.jsonl")
    if os.path.exists(path250):
        rows = load_jsonl(path250)
    else:
        idx = rng.sample(range(len(gsm)), N_EVAL)
        rows = [gsm_row(i) for i in idx]
        dump_jsonl(rows, path250)
        print(f"[out] gsm8k_250.jsonl ({len(rows)})")
    extend_to(1000, "gsm8k", rows, len(gsm), gsm_row)
    assert all(r["gold"] is not None for r in load_jsonl(
        os.path.join(OUT, "gsm8k_1000.jsonl")))

    if os.path.exists(os.path.join(OUT, "calib_mixed.jsonl")):
        print("[skip] calib_mixed.jsonl exists (frozen artifact)")
        return
    fillers = load_jsonl(os.path.join(ROOT, "data",
                                      "mmlu_1000_random_samples_filler.jsonl"))
    calib = [{"benchmark": "mmlu", **s}
             for s in rng.sample(fillers, N_CALIB_EACH)]
    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    idx = rng.sample(range(len(gsm_train)), N_CALIB_EACH)
    calib += [{"benchmark": "gsm8k", "question": gsm_train[i]["question"]}
              for i in idx]
    rng.shuffle(calib)
    dump_jsonl(calib, os.path.join(OUT, "calib_mixed.jsonl"))
    print(f"[out] calib_mixed.jsonl ({len(calib)})")


if __name__ == "__main__":
    main()

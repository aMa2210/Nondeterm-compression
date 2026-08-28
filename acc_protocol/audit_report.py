"""Audit report: generation length distribution, extraction tiers, scoring,
and sample outputs — the go/no-go gate before the full accuracy protocol."""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_jsonl, score_sample  # noqa: E402

DATA = os.path.join(HERE, "data")
OUTPUTS = os.path.join(HERE, "outputs")


def main():
    samples = {s["id"]: s for b in ("mmlu", "gsm8k")
               for s in load_jsonl(os.path.join(DATA, f"{b}_250.jsonl"))}

    for path in sorted(glob.glob(os.path.join(OUTPUTS, "*", "*_audit*.jsonl"))):
        name = (os.path.basename(os.path.dirname(path)) + "/"
                + os.path.basename(path).replace(".jsonl", ""))
        rows = load_jsonl(path)
        lens = np.array([r["gen_tokens"] for r in rows])
        fin = np.array([r["finished"] for r in rows])
        scores = [score_sample(samples[r["id"]], r["text"]) for r in rows]
        tiers = {}
        for s in scores:
            tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1
        acc = np.mean([s["correct"] for s in scores])
        print(f"\n=== {name} (n={len(rows)}) ===")
        print(f"  len: mean={lens.mean():.0f} p50={np.median(lens):.0f} "
              f"p90={np.percentile(lens, 90):.0f} max={lens.max()} | "
              f"eos-finished {fin.mean()*100:.0f}%")
        print(f"  extraction tiers: {tiers} | accuracy {acc*100:.0f}%")
        for r, s in zip(rows, scores):
            tail = r["text"][-120:].replace("\n", " ")
            mark = "OK " if s["correct"] else ("txt" if s["tier"] != "fail" else "FAIL")
            print(f"   [{mark}] {r['id']} {r['gen_tokens']}tok "
                  f"pred={s['extracted']} gold={s['gold']} tier={s['tier']} "
                  f"| ...{tail}")


if __name__ == "__main__":
    main()

"""Score the accuracy protocol and test the paper's claims.

Reads acc_protocol/outputs/{model_tag}/*.jsonl (audit files excluded), scores
every arm, and reports:
  - accuracy per (arm, benchmark) + pooled, with extraction/truncation rates
  - Acc3 statistics over groupings: mean, std (population), min, max
  - claims per pruning level k:  Acc2(k) > min_i Acc3(i)
                                 Acc2(k) > mean(Acc3) - std(Acc3)
  - paired McNemar (exact binomial) Acc1 vs each Acc2 level, and Acc1 vs the
    pooled per-question majority of Acc3 groupings

Outputs: results/{model_tag}/accuracy_report.csv, claims.md (also printed).
"""

import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_jsonl, score_sample  # noqa: E402

TAGS = {"deepseek": "deepseek_qwen3", "gemma3": "gemma3", "llama31": "llama31"}


def mcnemar_p(b, c):
    """Exact two-sided binomial McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=TAGS, required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--variant", default="")
    ap.add_argument("--acc3-bs", type=int, default=16,
                    help="batch size whose acc3 groupings define the noise floor")
    args = ap.parse_args()
    tag = TAGS[args.model]
    if args.variant:
        tag = f"{tag}_{args.variant}"
    outdir = os.path.join(HERE, "outputs", tag)
    resdir = os.path.join(HERE, "results", tag)
    os.makedirs(resdir, exist_ok=True)

    samples = {s["id"]: s for b in ("mmlu", "gsm8k")
               for s in load_jsonl(os.path.join(HERE, "data",
                                                f"{b}_{args.n}.jsonl"))}

    # arm -> {qid: row}
    arms = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(outdir, "*.jsonl"))):
        base = os.path.basename(path).replace(".jsonl", "")
        if "_audit" in base:
            continue
        m = re.match(r"(.+)_(mmlu|gsm8k)$", base)
        arm = m.group(1)
        for r in load_jsonl(path):
            s = score_sample(samples[r["id"]], r["text"])
            arms[arm][r["id"]] = {**r, **s,
                                  "benchmark": samples[r["id"]]["benchmark"]}

    # ---------------- per-arm table ----------------
    rows = []
    for arm in sorted(arms):
        d = arms[arm]
        for bench in ("mmlu", "gsm8k", "all"):
            sel = [v for v in d.values()
                   if bench == "all" or v["benchmark"] == bench]
            if not sel:
                continue
            rows.append({
                "arm": arm, "benchmark": bench, "n": len(sel),
                "accuracy": np.mean([v["correct"] for v in sel]),
                "extract_fail": np.mean([v["tier"] == "fail" for v in sel]),
                "truncated": np.mean([not v["finished"] for v in sel]),
                "mean_gen_tokens": np.mean([v["gen_tokens"] for v in sel]),
            })
    import csv
    with open(os.path.join(resdir, "accuracy_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    def acc_of(arm, bench="all"):
        d = arms[arm]
        sel = [v for v in d.values() if bench == "all" or v["benchmark"] == bench]
        return np.mean([v["correct"] for v in sel]) if sel else float("nan")

    lines = [f"# Accuracy protocol — {tag}", ""]
    lines.append("| arm | mmlu | gsm8k | all | trunc% | fail% |")
    lines.append("|---|---|---|---|---|---|")
    for arm in sorted(arms):
        sel = list(arms[arm].values())
        lines.append(
            f"| {arm} | {acc_of(arm,'mmlu')*100:.1f} | {acc_of(arm,'gsm8k')*100:.1f} "
            f"| {acc_of(arm)*100:.1f} | "
            f"{np.mean([not v['finished'] for v in sel])*100:.1f} | "
            f"{np.mean([v['tier']=='fail' for v in sel])*100:.1f} |")

    # ---------------- Acc3 stats + claims ----------------
    # only one batch size defines the noise floor — never pool B=16 with B=32
    acc3_arms = [a for a in arms if a.startswith(f"acc3_b{args.acc3_bs}_")]
    acc2_arms = [a for a in arms if a.startswith("acc2")]
    if acc3_arms and "acc1" in arms:
        for bench in ("mmlu", "gsm8k", "all"):
            a3 = np.array([acc_of(a, bench) for a in sorted(acc3_arms)])
            a1 = acc_of("acc1", bench)
            lines += ["", f"## Claims ({bench})", "",
                      f"- Acc1 (B=1 full) = {a1*100:.2f}",
                      f"- Acc3 over {len(a3)} groupings: mean {a3.mean()*100:.2f}, "
                      f"std {a3.std()*100:.2f}, min {a3.min()*100:.2f}, "
                      f"max {a3.max()*100:.2f}",
                      f"- thresholds: min = {a3.min()*100:.2f}, "
                      f"mean-std = {(a3.mean()-a3.std())*100:.2f}", ""]
            for arm in sorted(acc2_arms):
                a2 = acc_of(arm, bench)
                c1 = "PASS" if a2 > a3.min() else "fail"
                c2 = "PASS" if a2 > a3.mean() - a3.std() else "fail"
                lines.append(f"- {arm}: {a2*100:.2f} | >min: {c1} | >mean-std: {c2}")

    # ---------------- McNemar ----------------
    if "acc1" in arms:
        lines += ["", "## Paired flips vs Acc1 (McNemar exact)", ""]
        base = arms["acc1"]
        others = sorted(acc2_arms) + sorted(acc3_arms)
        for arm in others:
            d = arms[arm]
            common = [q for q in base if q in d]
            b = sum(base[q]["correct"] and not d[q]["correct"] for q in common)
            c = sum(not base[q]["correct"] and d[q]["correct"] for q in common)
            lines.append(f"- {arm}: n={len(common)}, lost {b}, gained {c}, "
                         f"p={mcnemar_p(b, c):.4f}")

    report = "\n".join(lines)
    with open(os.path.join(resdir, "claims.md"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()

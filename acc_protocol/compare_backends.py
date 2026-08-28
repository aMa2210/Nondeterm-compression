"""Compare pruning backends (topiary vs modelopt) at matched FFN widths.

Reads acc_protocol/outputs/{model_tag}/*.jsonl and pairs arms
  acc2_keep{K}          (topiary in-place truncation)
  acc2_modelopt_keep{K} (Model-Optimizer/puzzletron checkpoint)
at each width K, reporting per benchmark and pooled:
  - accuracy of both backends, delta
  - claim tests vs the Acc3 noise floor (>min, >mean-std), same as score.py
  - McNemar (exact) of each backend vs Acc1, and topiary-vs-modelopt directly
  - per-question agreement between the backends (both right/both wrong/split)

Outputs: results/{model_tag}/backend_comparison.md and backend_comparison.png.
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_jsonl, score_sample  # noqa: E402
from score import TAGS, mcnemar_p  # noqa: E402


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

    def acc_of(arm, bench="all"):
        sel = [v for v in arms[arm].values()
               if bench == "all" or v["benchmark"] == bench]
        return np.mean([v["correct"] for v in sel]) if sel else float("nan")

    widths = sorted(
        {int(m.group(1)) for a in arms
         if (m := re.match(r"acc2_modelopt_keep(\d+)$", a))
         and f"acc2_keep{m.group(1)}" in arms},
        reverse=True)
    if not widths:
        sys.exit("no matched acc2_keep{K} / acc2_modelopt_keep{K} arm pairs found")

    acc3_arms = sorted(a for a in arms
                       if a.startswith(f"acc3_b{args.acc3_bs}_"))

    lines = [f"# Pruning backend comparison — {tag}", "",
             "topiary = salience width truncation (in-place); "
             "modelopt = NVIDIA Model-Optimizer (puzzletron) checkpoint", ""]

    for bench in ("mmlu", "gsm8k", "all"):
        a1 = acc_of("acc1", bench) if "acc1" in arms else float("nan")
        lines += [f"## {bench}", ""]
        if acc3_arms:
            a3 = np.array([acc_of(a, bench) for a in acc3_arms])
            lines += [f"Acc1 = {a1*100:.2f} · Acc3 mean {a3.mean()*100:.2f} "
                      f"± {a3.std()*100:.2f}, min {a3.min()*100:.2f} "
                      f"(n={len(a3)} groupings)", ""]
        lines += ["| keep | topiary | modelopt | delta | topiary claims "
                  "| modelopt claims |",
                  "|---|---|---|---|---|---|"]
        for k in widths:
            at = acc_of(f"acc2_keep{k}", bench)
            am = acc_of(f"acc2_modelopt_keep{k}", bench)
            claims = []
            for a in (at, am):
                if acc3_arms:
                    c1 = "PASS" if a > a3.min() else "fail"
                    c2 = "PASS" if a > a3.mean() - a3.std() else "fail"
                    claims.append(f">min:{c1} >m-s:{c2}")
                else:
                    claims.append("n/a")
            lines.append(f"| {k} | {at*100:.1f} | {am*100:.1f} "
                         f"| {(am-at)*100:+.1f} | {claims[0]} | {claims[1]} |")
        lines.append("")

    # ------------- paired analyses (pooled) -------------
    lines += ["## Paired analyses (pooled, exact McNemar)", ""]
    base = arms.get("acc1", {})
    for k in widths:
        t, m_ = arms[f"acc2_keep{k}"], arms[f"acc2_modelopt_keep{k}"]
        common = [q for q in t if q in m_]
        b = sum(t[q]["correct"] and not m_[q]["correct"] for q in common)
        c = sum(not t[q]["correct"] and m_[q]["correct"] for q in common)
        agree = sum(t[q]["correct"] == m_[q]["correct"] for q in common)
        line = (f"- keep{k}: topiary-vs-modelopt n={len(common)}, "
                f"topiary-only-right {b}, modelopt-only-right {c}, "
                f"agree {agree} ({100*agree/max(1,len(common)):.0f}%), "
                f"p={mcnemar_p(b, c):.4f}")
        if base:
            for name, d in (("topiary", t), ("modelopt", m_)):
                qq = [q for q in base if q in d]
                lost = sum(base[q]["correct"] and not d[q]["correct"] for q in qq)
                gained = sum(not base[q]["correct"] and d[q]["correct"] for q in qq)
                line += (f"\n    - {name} vs Acc1: lost {lost}, gained {gained}, "
                         f"p={mcnemar_p(lost, gained):.4f}")
        lines.append(line)

    report = "\n".join(lines)
    with open(os.path.join(resdir, "backend_comparison.md"), "w") as f:
        f.write(report + "\n")
    print(report)

    # ------------- figure -------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    for ax, bench in zip(axes, ("mmlu", "gsm8k", "all")):
        at = [acc_of(f"acc2_keep{k}", bench) * 100 for k in widths]
        am = [acc_of(f"acc2_modelopt_keep{k}", bench) * 100 for k in widths]
        ax.plot(widths, at, "o-", label="topiary")
        ax.plot(widths, am, "s-", label="modelopt")
        if "acc1" in arms:
            ax.axhline(acc_of("acc1", bench) * 100, color="k", lw=1,
                       label="Acc1 (B=1 full)")
        if acc3_arms:
            a3 = np.array([acc_of(a, bench) for a in acc3_arms]) * 100
            ax.axhspan(a3.min(), a3.max(), color="gray", alpha=0.25,
                       label="Acc3 min–max (B=16)")
        ax.set_title(bench)
        ax.set_xlabel("FFN neurons kept per layer")
        ax.invert_xaxis()
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Pruning backends vs batching noise floor — {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(resdir, "backend_comparison.png"), dpi=150)
    print(f"[fig] {os.path.join(resdir, 'backend_comparison.png')}")


if __name__ == "__main__":
    main()

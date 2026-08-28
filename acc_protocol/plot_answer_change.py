"""Answer-stability figure: % of questions whose extracted answer changes vs the
B=1 greedy baseline (Acc1), pruning sweep against the batching noise floor.

Metric: a question counts as "changed" when the extracted final answer differs
from Acc1's (both-extraction-failures = unchanged, one-sided failure = changed).
This is stricter than accuracy: offsetting flips (lost vs gained) cancel in
accuracy but both count here.

Usage: python plot_answer_change.py --model gemma3
Reads acc_protocol/outputs/{tag}/*.jsonl (audit files excluded).
Writes results/{tag}/answer_change_vs_noise_floor.png (+ CSV of the rates).
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_jsonl, score_sample  # noqa: E402

TAGS = {"deepseek": "deepseek_qwen3", "gemma3": "gemma3"}
INTER = {"deepseek_qwen3": 12288, "gemma3": 15360}
BENCH_LABEL = {"mmlu": "MMLU (250 q)", "gsm8k": "GSM8K (250 q)"}

# palette (validated): pruning = blue, batching floor = aqua; ink/chrome tokens
BLUE, AQUA = "#2a78d6", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")


def same_answer(a, b, bench):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a == b if bench == "mmlu" else abs(a - b) < 1e-6


def load_arms(outdir, samples):
    arms = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(outdir, "*.jsonl"))):
        base = os.path.basename(path)[: -len(".jsonl")]
        if "_audit" in base:
            continue
        arm = re.match(r"(.+)_(mmlu|gsm8k)$", base).group(1)
        for r in load_jsonl(path):
            s = score_sample(samples[r["id"]], r["text"])
            arms[arm][r["id"]] = {**s, "bench": samples[r["id"]]["benchmark"]}
    return arms


def change_rate(base, other, bench):
    common = [q for q in base if q in other and base[q]["bench"] == bench]
    if not common:
        return None
    changed = sum(not same_answer(base[q]["extracted"], other[q]["extracted"],
                                  bench) for q in common)
    return 100.0 * changed / len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=TAGS, default="gemma3")
    args = ap.parse_args()
    tag = TAGS[args.model]
    inter = INTER[tag]

    samples = {s["id"]: s for b in ("mmlu", "gsm8k")
               for s in load_jsonl(os.path.join(HERE, "data", f"{b}_250.jsonl"))}
    arms = load_arms(os.path.join(HERE, "outputs", tag), samples)
    base = arms["acc1"]

    # pruning sweep: pct cut -> chg%, per bench (0% = baseline, 0 by definition)
    prune, batch = {"mmlu": [(0.0, 0.0)], "gsm8k": [(0.0, 0.0)]}, defaultdict(list)
    rows = []
    for arm in sorted(arms):
        for bench in ("mmlu", "gsm8k"):
            rate = change_rate(base, arms[arm], bench)
            if rate is None or arm == "acc1":
                continue
            rows.append([arm, bench, f"{rate:.2f}"])
            if m := re.match(r"acc2_keep(\d+)$", arm):
                prune[bench].append((100 * (1 - int(m.group(1)) / inter), rate))
            elif arm.startswith("acc3"):
                batch[bench].append(rate)

    resdir = os.path.join(HERE, "results", tag)
    os.makedirs(resdir, exist_ok=True)
    with open(os.path.join(resdir, "answer_change_rates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "benchmark", "changed_pct_vs_acc1"])
        w.writerows(rows)

    N_Q = 250  # questions per benchmark; binomial s.e. of the flip proportion
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    for ax, bench in zip(axes, ("mmlu", "gsm8k")):
        ax.set_facecolor(SURFACE)
        xs, ys = zip(*sorted(prune[bench]))
        floor = np.array(batch[bench])
        mu, sd = floor.mean(), floor.std()

        ax.set_xlim(-1.5, max(xs) + 2.5)
        ax.set_ylim(0, max(max(ys), floor.max()) * 1.2)
        ax.set_xticks(xs)
        labels = [f"{round(x, 1):g}" for x in xs]
        if max(xs) > 20:  # wide panel: 2.1 tick collides with 0
            labels = ["" if 0 < x < 3 else s for x, s in zip(xs, labels)]
        ax.set_xticklabels(labels)

        ax.axhspan(floor.min(), floor.max(), color=AQUA, alpha=0.13, lw=0)
        ax.axhspan(mu - sd, mu + sd, color=AQUA, alpha=0.28, lw=0)
        ax.axhline(mu, color=AQUA, lw=1.2)
        se = np.sqrt(np.array(ys) / 100 * (1 - np.array(ys) / 100) / N_Q) * 100
        line = ax.errorbar(xs, ys, yerr=se, color=BLUE, lw=2, marker="o",
                           ms=7, mec=SURFACE, mew=1.6, capsize=3,
                           elinewidth=1.1, ecolor=BLUE, alpha=None)

        # value on every pruned point (skip the definitional 0 at x=0)
        for x, y in zip(xs, ys):
            if x == 0:
                continue
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 10),
                        textcoords="offset points", ha="center", color=INK,
                        fontsize=8.5, fontweight="bold")
        ax.annotate(f"{mu:.1f}", (ax.get_xlim()[1], mu), xytext=(-4, 3),
                    textcoords="offset points", ha="right", color=INK2,
                    fontsize=8.5)

        ax.set_title(BENCH_LABEL[bench], color=INK, fontsize=11, pad=10)
        ax.set_xlabel("FFN width pruned (%, topiary salience truncation)",
                      color=INK2, fontsize=9.5)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    axes[0].set_ylabel("questions with changed answer (%)", color=INK2,
                       fontsize=9.5)
    n_g = len(batch["mmlu"])
    axes[0].legend(
        [line,
         plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.40, lw=0),
         plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.15, lw=0)],
        ["pruned (B=1) ± binomial s.e.",
         f"batching mean ± std (B=16, {n_g} groupings)",
         "batching min–max (worst/best grouping)"],
        loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)

    fig.suptitle(f"{tag}: answers changed vs deterministic B=1 greedy baseline",
                 color=INK, fontsize=12.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(resdir, "answer_change_vs_noise_floor.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"[out] {out}")
    print(f"[out] {os.path.join(resdir, 'answer_change_rates.csv')}")


if __name__ == "__main__":
    main()

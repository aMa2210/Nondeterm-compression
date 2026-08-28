"""Overlay pruning-induced logprob perturbation (B=1 pruned vs B=1 full) against
the batching noise floor from the prior paper (reference_results, A100).

Comparison space: per-(step, top-10 baseline token) probability perturbation.
- pruning series : AbsDiff = |P_pruned(v) - P_b1(v)|          (one run)
- batching series: Range_Prob_Runs (max-min over runs) and
                   |Mean_Prob_Runs - Prob_B1| (systematic bias)

Outputs (quick_test/results/):
  summary_stats.csv, overlay_ecdf.png, stable_prefix.png, findings.md
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "deepseek_qwen3"
INTER = int(sys.argv[2]) if len(sys.argv) > 2 else 12288
# reference-CSV model name may differ from our tag (e.g. llama31 has no rows;
# nearest reference is llama3.2 — indicative only)
REF_MODEL = sys.argv[3] if len(sys.argv) > 3 else MODEL_TAG
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", MODEL_TAG)
REF = os.path.join(ROOT, "reference_results",
                   "stability_token_level_report_STD_RANGE_A100.csv")


def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def stats(x, label):
    x = np.asarray(x, dtype=float)
    return {"series": label, "n": len(x), "mean": x.mean(),
            "p50": np.median(x), "p90": np.percentile(x, 90),
            "p99": np.percentile(x, 99), "max": x.max()}


def main():
    pr = pd.read_csv(os.path.join(RESULTS, "pruning_token_level_report.csv"))
    dv = pd.read_csv(os.path.join(RESULTS, "divergence_report.csv"))
    ref = pd.read_csv(REF)
    ref = ref[ref["Model"] == REF_MODEL].rename(columns={
        "Range_Prob_Runs (R_j)": "Range", "Std_Prob_Runs (sigma_j)": "Std"})
    ref["AbsBias"] = (ref["Mean_Prob_Runs"] - ref["Prob_B1"]).abs()

    rows, series = [], {}
    for keep, g in pr.groupby("Keep"):
        pct_cut = 100 * (1 - keep / INTER)
        label = f"prune {pct_cut:.1f}% (keep {keep})"
        series[label] = g["AbsDiff"].values
        rows.append(stats(g["AbsDiff"].values, label))
    ref_series = {}
    for bs, g in ref.groupby("BatchSize"):
        label = f"batch B={bs} range"
        ref_series[label] = g["Range"].values
        rows.append(stats(g["Range"].values, label))
        rows.append(stats(g["AbsBias"].values, f"batch B={bs} |bias|"))

    sm = pd.DataFrame(rows)
    sm.to_csv(os.path.join(RESULTS, "summary_stats.csv"), index=False)
    print(sm.to_string(index=False,
                       float_format=lambda v: f"{v:.5f}"))

    # ----- ECDF overlay -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap_p = plt.cm.autumn_r(np.linspace(0.35, 1, len(series)))
    cmap_b = plt.cm.winter(np.linspace(0.2, 0.9, len(ref_series)))
    for ax, xlim, title in ((axes[0], 0.05, "zoom: perturbation <= 0.05"),
                            (axes[1], 1.0, "full scale")):
        for (label, x), c in zip(sorted(series.items(),
                                        key=lambda kv: -len(kv[1])), cmap_p):
            xs, ys = ecdf(x)
            ax.plot(xs, ys, color=c, label=label, lw=1.8)
        for (label, x), c in zip(ref_series.items(), cmap_b):
            xs, ys = ecdf(x)
            ax.plot(xs, ys, color=c, label=label, lw=1.8, ls="--")
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("|ΔP| per (step, top-10 token)")
        ax.set_ylabel("ECDF")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{MODEL_TAG}: pruning perturbation (B=1) vs batching noise "
                 "floor (A100, 10 runs)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "overlay_ecdf.png"), dpi=150)

    # ----- stable prefix comparison -----
    # batching: longest analysed step per (BS, question) == group stable prefix
    ref_prefix = ref.groupby(["BatchSize", "Question"])["Token_Step"].max() + 1
    fig2, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for bs, g in ref_prefix.groupby("BatchSize"):
        data.append(g.values)
        labels.append(f"B={bs}")
    for keep, g in dv.groupby("Keep"):
        d = g["DivergenceStep"].replace(-1, g["BaselineLen"].max()).values
        data.append(d)
        labels.append(f"prune {100*(1-keep/INTER):.1f}%")
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("tokens until divergence from B=1 full baseline")
    ax.set_title(f"{MODEL_TAG}: greedy stable prefix — batching runs vs pruning")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS, "stable_prefix.png"), dpi=150)
    print(f"[out] {RESULTS}/overlay_ecdf.png, stable_prefix.png, summary_stats.csv")


if __name__ == "__main__":
    main()

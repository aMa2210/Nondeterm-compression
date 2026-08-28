"""Evaluate the quick test: pruned-vs-full B=1 perturbation, in the reference
paper's per-token metric space.

Mirrors reference_code/Evaluation_metrics_per_token.ipynb (STD_RANGE cell):
- baseline = full model B=1 run; "runs" = one pruned-model B=1 run per keep level
- stable prefix = steps before the pruned run's tokens diverge from baseline
- probs = softmax over the stored top-15 sparse logits (missing tokens -> 0)
- for each of the top-10 baseline tokens at each step: |P_pruned - P_b1|,
  the two-run analogue of Range_Prob_Runs (and of |Mean_Prob_Runs - Prob_B1|).

Outputs:
  quick_test/results/pruning_token_level_report.csv
      [Model, Keep, KeepFrac, Question, Token_Step, Token_Rank, Token_ID,
       Prob_B1, Prob_Pruned, AbsDiff, Logit_B1, Logit_Pruned, AbsDiffLogit]
  quick_test/results/divergence_report.csv
      [Model, Keep, KeepFrac, Question, DivergenceStep, BaselineLen, PrunedLen]
"""

import csv
import os
import pickle
import re

import torch
import torch.nn.functional as F

import sys
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "deepseek_qwen3"
INTER = int(sys.argv[2]) if len(sys.argv) > 2 else 12288
TOP_K_TO_ANALYZE = 10

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
RESULTS = os.path.join(HERE, "results", MODEL_TAG)


def load_run(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["tokens"], data["logits"]


def sparse_probs(sp):
    """softmax over the top-15 captured logits; dict token_id -> (prob, logit)."""
    idx = sp._indices()[0]
    vals = sp._values().float()
    p = F.softmax(vals, dim=-1)
    return {int(i): (float(pj), float(v)) for i, pj, v in zip(idx, p, vals)}


def find_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def main():
    os.makedirs(RESULTS, exist_ok=True)
    base_root = os.path.join(REPORTS, f"{MODEL_TAG}_full")
    keep_dirs = sorted(
        (int(m.group(1)), d) for d in os.listdir(REPORTS)
        if (m := re.match(rf"{MODEL_TAG}_keep(\d+)$", d)))

    token_rows, div_rows = [], []
    questions = sorted(os.listdir(base_root))
    for keep, dname in keep_dirs:
        frac = keep / INTER
        for q in questions:
            bpath = os.path.join(base_root, q, "run_00.pkl")
            ppath = os.path.join(REPORTS, dname, q, "run_00.pkl")
            if not (os.path.exists(bpath) and os.path.exists(ppath)):
                continue
            b_tokens, b_logits = load_run(bpath)
            p_tokens, p_logits = load_run(ppath)
            div = find_divergence(b_tokens.tolist(), p_tokens.tolist())
            limit = min(len(b_logits), len(p_logits),
                        div if div != -1 else 10**9)
            div_rows.append([MODEL_TAG, keep, f"{frac:.4f}", q,
                             div, len(b_tokens), len(p_tokens)])
            for t in range(limit):
                bl = b_logits[t]
                b_idx = bl._indices()[0]
                b_vals = bl._values().float()
                order = torch.argsort(b_vals, descending=True)[:TOP_K_TO_ANALYZE]
                b_p = F.softmax(b_vals, dim=-1)
                pmap = sparse_probs(p_logits[t])
                for rank, j in enumerate(order, start=1):
                    tid = int(b_idx[j])
                    prob_b1 = float(b_p[j])
                    logit_b1 = float(b_vals[j])
                    prob_pr, logit_pr = pmap.get(tid, (0.0, float("nan")))
                    token_rows.append([
                        MODEL_TAG, keep, f"{frac:.4f}", q, t, rank, tid,
                        prob_b1, prob_pr, abs(prob_pr - prob_b1),
                        logit_b1, logit_pr,
                        abs(logit_pr - logit_b1) if logit_pr == logit_pr else "",
                    ])

    with open(os.path.join(RESULTS, "pruning_token_level_report.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Keep", "KeepFrac", "Question", "Token_Step",
                    "Token_Rank", "Token_ID", "Prob_B1", "Prob_Pruned",
                    "AbsDiff", "Logit_B1", "Logit_Pruned", "AbsDiffLogit"])
        w.writerows(token_rows)
    with open(os.path.join(RESULTS, "divergence_report.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Keep", "KeepFrac", "Question", "DivergenceStep",
                    "BaselineLen", "PrunedLen"])
        w.writerows(div_rows)
    print(f"[out] {len(token_rows)} token rows, {len(div_rows)} divergence rows")


if __name__ == "__main__":
    main()

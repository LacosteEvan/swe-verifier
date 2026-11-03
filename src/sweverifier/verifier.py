"""Best-of-N selection: every method scores the 5 candidates of an issue, the top one is submitted, and the metric is
the share of issues resolved (pass@1 after selection). Learned scores are out of fold: an issue is never scored by a
model trained on any of its candidates."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .features import candidate_features
from .runner import RUNS, read

AGENT = "qwen/qwen3-coder-30b-a3b-instruct"
REFERENCE = "anthropic/claude-haiku-5.5"
FOLDS = 5
SEED = 0


def table(agent: str = AGENT) -> pd.DataFrame:
    """One row per scored candidate of `agent`, on issues where all its samples are scored: features and label."""
    attempts = {(a["instance_id"], a["sample"]): a for a in read("attempts") if a["model"] == agent}
    evals = {(e["instance_id"], e["sample"]): e for e in read("evals") if e["model"] == agent}
    lora = lora_scores()
    judges = {}
    for j in read("judgements"):
        if j["model"] == agent:
            judges[(j["instance_id"], j["sample"], j["judge"])] = j["p"]
    by_issue: dict[str, list[int]] = {}
    for iid, k in evals:
        by_issue.setdefault(iid, []).append(k)
    n = max((len(v) for v in by_issue.values()), default=0)
    rows = []
    for iid, samples in sorted(by_issue.items()):
        if len(samples) < n or any((iid, k) not in attempts for k in samples):
            continue
        patches = {k: attempts[(iid, k)]["patch"] for k in samples}
        for k in sorted(samples):
            path = RUNS / "trajectories" / f"{iid}__{agent.split('/')[-1]}__{k}.json"
            trajectory = json.loads(path.read_text()) if path.exists() else []
            others = [p for j, p in patches.items() if j != k]
            row = {"instance_id": iid, "repo": iid.split("__")[0], "sample": k,
                   "resolved": float(evals[(iid, k)]["resolved"]), "cost_usd": attempts[(iid, k)]["cost_usd"],
                   **candidate_features(attempts[(iid, k)], trajectory, others)}
            for (jid, jk, jm), p in judges.items():
                if jid == iid and jk == k:
                    row[f"judge:{jm.split('/')[-1]}"] = p
            if (iid, k) in lora:
                row["lora"] = lora[(iid, k)]
            rows.append(row)
    return pd.DataFrame(rows)


def lora_scores() -> dict[tuple[str, int], float]:
    """Out-of-fold scores from notebooks/verifier_lora.ipynb, once it has run on a GPU."""
    path = RUNS.parent / "results" / "lora_scores.jsonl"
    rows = [json.loads(line) for line in path.open()] if path.exists() else []
    return {(r["instance_id"], r["sample"]): r["p"] for r in rows}


FEATURES = ["submitted", "steps", "hit_step_limit", "tool_calls", "edits", "failed_edits", "views", "searches",
            "test_runs", "ran_tests", "last_tests_passed", "script_runs", "last_script_ok", "empty", "files", "hunks",
            "added", "removed", "log_size", "agree_max", "agree_mean", "same_edit"]


def folds(df: pd.DataFrame, k: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """Fold per row, grouped by issue."""
    issues = sorted(df["instance_id"].unique())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(issues))
    fold_of = {issues[i]: j % k for j, i in enumerate(order)}
    return df["instance_id"].map(fold_of).to_numpy()


def learned_scores(df: pd.DataFrame, columns: list[str], C: float = 0.3) -> np.ndarray:
    """Out-of-fold P(resolved) from a standardized logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, f = df[columns].to_numpy(float), df["resolved"].to_numpy(), folds(df)
    out = np.zeros(len(df))
    for j in range(FOLDS):
        train, test = f != j, f == j
        if not test.any():
            continue
        if len(set(y[train])) < 2:
            out[test] = y[train].mean()
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        model.fit(X[train], y[train])
        out[test] = model.predict_proba(X[test])[:, 1]
    return out


def selected(df: pd.DataFrame, score: np.ndarray) -> pd.Series:
    """Per issue, the expected resolution of the top-scored candidate; ties are shared equally (no lucky tie-breaks)."""
    d = df.assign(score=np.round(score, 9))
    top = d[d["score"] == d.groupby("instance_id")["score"].transform("max")]
    return top.groupby("instance_id")["resolved"].mean()


def bootstrap(per_issue: pd.Series, baseline: pd.Series, n: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """95% interval on the mean gain over the baseline, issues resampled."""
    diff = (per_issue - baseline.reindex(per_issue.index)).to_numpy()
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def methods(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every selection rule, as a score per candidate (higher is picked)."""
    n = df["sample"].nunique()
    out = {
        "Random candidate (pass@1)": np.zeros(len(df)),
        "Submitted, then smallest patch": df["submitted"] * 1e3 - df["log_size"] - 1e3 * df["empty"],
        "Agent's own tests passed": df["last_tests_passed"] + 0.1 * df["submitted"] - df["empty"],
        "Majority vote (most similar to the others)": df["agree_mean"] - df["empty"],
        "Learned: trajectory and patch features": learned_scores(df, FEATURES),
    }
    for col in [c for c in df.columns if c.startswith("judge:")]:
        name = col.split(":", 1)[1]
        out[f"LLM judge: {name}"] = df[col].to_numpy()
        out[f"Learned: features + {name} judge"] = learned_scores(df, FEATURES + [col])
    if "lora" in df and df["lora"].notna().all():
        out["Learned: LoRA verifier (Qwen2.5-Coder 1.5B)"] = df["lora"].to_numpy()
        out["Learned: features + LoRA verifier"] = learned_scores(df, FEATURES + ["lora"])
    out[f"Oracle (pass@{n})"] = df["resolved"].to_numpy()
    return out


def within_issue_auc(df: pd.DataFrame, score: np.ndarray) -> float:
    """AUC among the candidates of one issue, averaged over the issues with both right and wrong candidates: the
    only ranking selection uses. A pooled AUC also rewards telling easy issues from hard ones."""
    from sklearn.metrics import roc_auc_score

    d = df.assign(score=score)
    aucs = [roc_auc_score(g["resolved"], g["score"]) for _, g in d.groupby("instance_id")
            if g["resolved"].nunique() == 2]
    return float(np.mean(aucs)) if aucs else np.nan


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    scores = methods(df)
    base = selected(df, scores["Random candidate (pass@1)"])
    rows = []
    for name, s in scores.items():
        per = selected(df, s)
        lo, hi = bootstrap(per, base) if not name.startswith("Random") else (np.nan, np.nan)
        ranks = df["resolved"].nunique() == 2 and np.ptp(s) > 0
        rows.append({"method": name, "resolved": per.mean(), "gain": per.mean() - base.mean(), "gain_lo": lo,
                     "gain_hi": hi, "auc": roc_auc_score(df["resolved"], s) if ranks else np.nan,
                     "auc_within": within_issue_auc(df, s) if ranks else np.nan})
    return pd.DataFrame(rows)

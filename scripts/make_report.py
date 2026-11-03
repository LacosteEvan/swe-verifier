"""Every table and chart of the README, from the committed runs (attempts, evals, judgements, trajectories).

    python scripts/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sweverifier.dataset import instances
from sweverifier.features import similarity
from sweverifier.report import cost_scatter, selection_bars
from sweverifier.runner import read
from sweverifier.verifier import AGENT, REFERENCE, evaluate, table

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DOCS = ROOT / "results", ROOT / "docs"
NAMES = {AGENT: "Qwen3-Coder 30B-A3B", REFERENCE: "Claude Haiku 5.5"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    attempts = pd.DataFrame(read("attempts"))
    evals = pd.DataFrame(read("evals"))
    judgements = pd.DataFrame(read("judgements"))
    df = table()
    res = evaluate(df)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "selection.csv", index=False)
    issues = df["instance_id"].unique()
    n = df["sample"].nunique()

    runs = attempts.merge(evals[["instance_id", "model", "sample", "resolved"]], on=["instance_id", "model", "sample"])
    runs = runs[runs["instance_id"].isin(issues)]
    print(f"## The agents ({len(issues)} SWE-bench Lite issues)\n")
    print("| Agent | Attempts per issue | Resolved (one attempt) | Cost / attempt | Steps | Minutes | Submitted | "
          "Empty patch |")
    print("|---|---|---|---|---|---|---|---|")
    agents = {}
    for model in (AGENT, REFERENCE):
        g = runs[runs["model"] == model]
        agents[model] = {"resolved": g["resolved"].mean(), "cost": g["cost_usd"].mean(),
                         "cost_per_issue": g.groupby("instance_id")["cost_usd"].sum().mean()}
        print(f"| {NAMES[model]} | {g['sample'].nunique()} | **{pct(g['resolved'].mean())}** | "
              f"${g['cost_usd'].mean():.4f} | {g['steps'].mean():.1f} | {g['seconds'].mean() / 60:.1f} | "
              f"{pct(g['submitted'].mean())} | {pct((g['patch'].str.strip() == '').mean())} |")

    k = df.groupby("instance_id")["resolved"].sum().astype(int)
    spread = {int(i): int((k == i).sum()) for i in range(n + 1)}
    mixed = int(((k > 0) & (k < n)).sum())
    print(f"\nissues by candidates resolved (of {n}): {spread}; mixed: {mixed}")
    ref = evals[evals["model"] == REFERENCE].set_index("instance_id")["resolved"].reindex(issues).astype(int)
    print("reference resolved x agent pass@5:", pd.crosstab(ref, k.reindex(issues) > 0).to_dict())

    judge_cost = {m.split("/")[-1]: g["cost_usd"].sum() / len(issues) for m, g in judgements.groupby("judge")}
    extra = {name: 0.0 for name in res["method"]}
    for name in extra:
        for slug, c in judge_cost.items():
            if slug in name:
                extra[name] = c

    print(f"\n## Picking one of {n} candidates (out of fold, 95% bootstrap interval on the gain over a random pick)\n")
    print("| Method | Resolved | Gain (points) | AUC within an issue | AUC pooled | Verifier cost / issue |")
    print("|---|---|---|---|---|---|")
    for r in res.itertuples():
        ci = f"({100 * r.gain_lo:+.1f} to {100 * r.gain_hi:+.1f})"
        gain = "–" if r.gain_lo != r.gain_lo else f"{100 * r.gain:+.1f} {ci}"
        auc_w = "–" if r.auc_within != r.auc_within else f"{r.auc_within:.3f}"
        auc = "–" if r.auc != r.auc else f"{r.auc:.3f}"
        print(f"| {r.method} | {pct(r.resolved)} | {gain} | {auc_w} | {auc} | ${extra[r.method]:.4f} |")
    print(f"| *{NAMES[REFERENCE]}, one attempt (reference)* | {pct(agents[REFERENCE]['resolved'])} | | | | |")

    by = res.set_index("method")
    oracle = f"Oracle (pass@{n})"
    best = by.drop(oracle)["resolved"].idxmax()
    headroom = by.loc[oracle, "gain"]
    print(f"\nbest: {best}, {pct(by.loc[best, 'gain'] / headroom)} of the oracle's gain")

    short = {"Random candidate (pass@1)": "Random candidate", "Submitted, then smallest patch": "Submitted, smallest",
             "Agent's own tests passed": "Agent's own tests passed",
             "Majority vote (most similar to the others)": "Majority vote",
             "Learned: trajectory and patch features": "Learned: trajectory + patch",
             oracle: f"Oracle (best of {n}, hidden tests)"}
    rows = []
    for r in res.itertuples():
        label = short.get(r.method, r.method.replace("claude-", "").replace("-5.5", " 5.5")
                          .replace("LLM judge: ", "LLM judge: ").replace("haiku", "Haiku").replace("sonnet", "Sonnet"))
        kind = "baseline" if r.method in ("Random candidate (pass@1)", oracle) else (
            "best" if r.method == best else "method")
        rows.append((label, r.resolved, r.resolved - r.gain + r.gain_lo, r.resolved - r.gain + r.gain_hi, kind))
    selection_bars(rows, (f"{NAMES[REFERENCE]}, one attempt: {pct(agents[REFERENCE]['resolved'])}",
                          agents[REFERENCE]["resolved"]), DOCS / "selection")

    five = agents[AGENT]["cost_per_issue"]
    one = agents[AGENT]["cost"]
    sonnet = next(m for m in by.index if m.startswith("LLM judge") and "sonnet" in m)
    haiku = next(m for m in by.index if m.startswith("LLM judge") and "haiku" in m)
    points = [
        (f"{NAMES[AGENT]}, one attempt", one, agents[AGENT]["resolved"], "qwen", (8, 0)),
        (f"Qwen × {n}, Haiku judge picks", five + extra[haiku], by.loc[haiku, "resolved"], "qwen", (-9, 0)),
        (f"Qwen × {n}, Sonnet judge picks", five + extra[sonnet], by.loc[sonnet, "resolved"], "qwen", (9, -7)),
        (f"Qwen × {n}, best by the hidden tests (ceiling)", five, by.loc[oracle, "resolved"], "ceiling", (-10, 5)),
        (f"{NAMES[REFERENCE]}, one attempt", agents[REFERENCE]["cost"], agents[REFERENCE]["resolved"], "reference",
         (8, 0)),
    ]
    cost_scatter(points, DOCS / "cost")
    print("\n## Cost per issue\n")
    print("| System | Resolved | Cost / issue |")
    print("|---|---|---|")
    for label, cost, value, _, _ in points:
        print(f"| {label} | {pct(value)} | ${cost:.3f} |")

    # Does the judge verify, or remember? Public issues, public fixes: a judge trained on GitHub may know them.
    recall = r"upstream|actual fix|real fix|was fixed|merged|PR #|commit"
    patches = {(a.instance_id, a.sample): a.patch for a in attempts[attempts["model"] == AGENT].itertuples()}
    df["upstream_similarity"] = [similarity(patches[(i, s)], instances()[i]["patch"])
                                 for i, s in zip(df["instance_id"], df["sample"], strict=True)]
    right = df[(df["resolved"] == 1) & (df["empty"] == 0)]
    close, apart = right[right["upstream_similarity"] >= 0.5], right[right["upstream_similarity"] < 0.2]
    print("\n## Contamination: correct patches, scored by how close they are to the upstream fix\n")
    print(f"| Judge | Reviews citing the upstream fix | Correct patch close to upstream (n={len(close)}) | "
          f"Correct patch unlike upstream (n={len(apart)}) |")
    print("|---|---|---|---|")
    contamination = {}
    for model, g in judgements.groupby("judge"):
        col = f"judge:{model.split('/')[-1]}"
        texts = g["text"].fillna("")
        cites = int(texts.str.contains(recall, case=False).sum())
        contamination[model] = {"cites": cites, "reviews": int((texts != "").sum()),
                                "close": float(close[col].mean()), "apart": float(apart[col].mean())}
        c = contamination[model]
        print(f"| {model.split('/')[-1]} | {c['cites']} of {c['reviews']} ({pct(c['cites'] / c['reviews'])}) | "
              f"{c['close']:.2f} | {c['apart']:.2f} |")

    summary = {"issues": len(issues), "samples": n, "mixed_issues": mixed, "resolved_by_k": spread,
               "agents": {NAMES[m]: v for m, v in agents.items()}, "judge_cost_per_issue": judge_cost,
               "best": best, "share_of_oracle_gain": by.loc[best, "gain"] / headroom, "contamination": contamination,
               "spent": {"generation": float(attempts["cost_usd"].sum()),
                         "judges": float(judgements["cost_usd"].sum())}}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nspent: {summary['spent']}")


if __name__ == "__main__":
    main()

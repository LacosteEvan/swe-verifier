"""sweverifier generate | judge | verify | export"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .dataset import ROOT


def _env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")


def cmd_generate(args) -> int:
    from .llm import Budget
    from .runner import AGENTS, Agent, Run

    _env()
    agents = AGENTS if args.samples is None else tuple(Agent(a.model, min(a.samples, args.samples), a.temperature)
                                                       for a in AGENTS)
    start = time.monotonic()
    run = Run(args.issues, Budget(args.max_cost), agents, containers=args.containers,
              on_event=lambda msg: print(f"[{(time.monotonic() - start) / 60:6.1f} min] {msg}", file=sys.stderr,
                                         flush=True))
    run.run(workers=args.workers)
    print(f"done: {len(run.accepted)} issues, ${run.budget.spent:.2f} spent this run", file=sys.stderr)
    return 0


def cmd_judge(args) -> int:
    """Score every candidate of the agent with an LLM judge (resumable)."""
    from .dataset import instances
    from .judge import NoVerdict, judge, to_row
    from .llm import LLM, Budget, BudgetExceeded
    from .runner import RUNS, read
    from .verifier import AGENT

    _env()
    done = {(j["instance_id"], j["sample"]) for j in read("judgements") if j["judge"] == args.judge_model}
    scored = {(e["instance_id"], e["sample"]) for e in read("evals") if e["model"] == AGENT}
    todo = [a for a in read("attempts") if a["model"] == AGENT and (a["instance_id"], a["sample"]) in scored
            and (a["instance_id"], a["sample"]) not in done]
    llm = LLM(args.judge_model, Budget(args.max_cost))
    out = (RUNS / "judgements.jsonl").open("a")
    failed = []

    def one(a):
        try:
            p, cost, text = judge(llm, instances()[a["instance_id"]], a["patch"])
        except BudgetExceeded:
            return
        except NoVerdict as e:
            failed.append(a["instance_id"])
            print(f"{a['instance_id']} #{a['sample']}: {e}", file=sys.stderr)
            return
        out.write(to_row(a["instance_id"], AGENT, a["sample"], args.judge_model, p, cost, text) + "\n")
        out.flush()

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, todo))
    print(f"{len(todo) - len(failed)} candidates judged by {args.judge_model}, {len(failed)} without a verdict "
          f"(rerun to retry), ${llm.budget.spent:.2f}", file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    from .verifier import evaluate, table

    df = table()
    res = evaluate(df)
    print(f"{df['instance_id'].nunique()} issues, {len(df)} candidates", file=sys.stderr)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "selection.csv", index=False)
    return 0


def cmd_export(args) -> int:
    """The verifier's training set for the LoRA notebook: issue and patch as text, the label, the fold."""
    from .dataset import instances
    from .runner import read
    from .verifier import folds, table

    df = table()
    df["fold"] = folds(df)
    patches = {(a["instance_id"], a["sample"]): a["patch"] for a in read("attempts")}
    path = ROOT / "data" / "verifier.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as f:
        for r in df.itertuples():
            inst = instances()[r.instance_id]
            row = {"instance_id": r.instance_id, "sample": r.sample, "fold": int(r.fold), "label": int(r.resolved),
                   "repo": inst["repo"], "issue": inst["problem_statement"],
                   "patch": patches[(r.instance_id, r.sample)]}
            f.write(json.dumps(row) + "\n")
    print(f"{len(df)} candidates → {path}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sweverifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="run the agents on the subset and score every patch (resumable)")
    g.add_argument("--issues", type=int, default=100)
    g.add_argument("--samples", type=int, help="cap the samples per agent, for a pilot")
    g.add_argument("--max-cost", type=float, required=True, help="spending cap for this run, in $")
    g.add_argument("--workers", type=int, default=3, help="issues in flight")
    g.add_argument("--containers", type=int, default=4, help="containers running at the same time")
    g.set_defaults(func=cmd_generate)

    j = sub.add_parser("judge", help="score every candidate with an LLM judge (resumable)")
    j.add_argument("--judge-model", default="anthropic/claude-haiku-5.5")
    j.add_argument("--max-cost", type=float, required=True)
    j.set_defaults(func=cmd_judge)

    sub.add_parser("verify", help="compare the selection methods, out of fold").set_defaults(func=cmd_verify)
    sub.add_parser("export", help="write data/verifier.jsonl for the LoRA notebook").set_defaults(func=cmd_export)
    args = p.parse_args(argv)
    return args.func(args)

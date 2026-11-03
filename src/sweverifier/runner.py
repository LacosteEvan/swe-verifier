"""Generate and score the candidate patches, issue by issue, resumably.

For each issue in the fixed order: pull its image, check that the reference patch passes here (some tests need the
network or x86 hardware: those issues are skipped), run every agent attempt, score every patch with the harness, then
delete the image. Results are appended to runs/*.jsonl as they come, so an interrupted run picks up where it stopped.
"""
from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .agent import solve
from .dataset import ROOT, instances, order
from .env import pull
from .evaluate import evaluate
from .llm import LLM, Budget, BudgetExceeded

RUNS = ROOT / "runs"


@dataclass(frozen=True)
class Agent:
    model: str
    samples: int
    temperature: float

    @property
    def slug(self) -> str:
        return self.model.split("/")[-1]


AGENTS = (Agent("qwen/qwen3-coder-30b-a3b-instruct", 5, 0.7), Agent("anthropic/claude-haiku-5.5", 1, 0.0))


def read(name: str) -> list[dict]:
    path = RUNS / f"{name}.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


class Run:
    def __init__(self, target: int, budget: Budget, agents=AGENTS, containers: int = 4, on_event=None):
        self.target, self.budget, self.agents = target, budget, agents
        self.slots = threading.Semaphore(containers)  # agent and harness containers at the same time
        self.lock = threading.Lock()
        self.on_event = on_event or (lambda msg: None)
        RUNS.mkdir(exist_ok=True)
        (RUNS / "trajectories").mkdir(exist_ok=True)
        self.gold = {r["instance_id"]: r for r in read("gold")}
        self.attempts = {(r["instance_id"], r["model"], r["sample"]) for r in read("attempts")}
        self.evals = {(r["instance_id"], r["model"], r["sample"]) for r in read("evals")}
        self.accepted = {i for i, r in self.gold.items() if r["resolved"]}
        self.stopped = False

    def append(self, name: str, row: dict) -> None:
        with self.lock, (RUNS / f"{name}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def complete(self, iid: str) -> bool:
        return all((iid, a.model, k) in self.evals for a in self.agents for k in range(a.samples))

    def issue(self, iid: str) -> None:
        if self.stopped:
            return
        inst = instances()[iid]
        if iid in self.gold and (not self.gold[iid]["resolved"] or self.complete(iid)):
            return
        with self.lock:
            if iid not in self.accepted and len(self.accepted) >= self.target:
                return
        pull(inst["image"])
        try:
            if iid not in self.gold:
                with self.slots:
                    g = evaluate(iid, inst["patch"], "gold", "gold")
                self.gold[iid] = {"instance_id": iid, **g}
                self.append("gold", self.gold[iid])
                if not g["resolved"]:
                    self.on_event(f"{iid}: reference patch fails here, skipped")
                    return
                with self.lock:
                    if len(self.accepted) >= self.target:
                        return
                    self.accepted.add(iid)
            jobs = [(a, k) for a in self.agents for k in range(a.samples) if (iid, a.model, k) not in self.evals]
            with ThreadPoolExecutor(len(jobs) or 1) as pool:
                list(pool.map(lambda job: self.attempt(inst, *job), jobs))
        finally:  # about 2.3 GB per issue: a resumed run pulls it again
            subprocess.run(["docker", "rmi", "-f", inst["image"]], capture_output=True)

    def attempt(self, inst: dict, agent: Agent, k: int) -> None:
        iid, key = inst["instance_id"], (inst["instance_id"], agent.model, k)
        rows = ({(r["instance_id"], r["model"], r["sample"]): r for r in read("attempts")}
                if key in self.attempts else {})
        if key in rows:
            patch = rows[key]["patch"]
        else:
            if self.stopped:
                return
            try:
                self.budget.check()
            except BudgetExceeded as e:
                self.stopped = True
                self.on_event(f"stopping: {e}")
                return
            with self.slots:
                a = solve(inst, LLM(agent.model, self.budget, temperature=agent.temperature), sample=k)
            if a.error and a.error.startswith("BudgetExceeded"):
                self.stopped = True
                return
            row = asdict(a)
            trajectory = row.pop("trajectory")
            (RUNS / "trajectories" / f"{iid}__{agent.slug}__{k}.json").write_text(json.dumps(trajectory))
            self.append("attempts", row)
            self.attempts.add(key)
            patch = a.patch
        with self.slots:
            e = evaluate(iid, patch, f"{agent.slug}-{k}", agent.slug)
        self.append("evals", {"instance_id": iid, "model": agent.model, "sample": k, **e})
        self.evals.add(key)
        self.on_event(f"{iid} {agent.slug} #{k}: {'resolved' if e['resolved'] else 'not resolved'}")

    def run(self, workers: int = 3) -> None:
        def safe(iid: str) -> None:  # one broken issue must not end a night-long run; a resumed run retries it
            try:
                self.issue(iid)
            except Exception as e:
                self.on_event(f"{iid}: skipped this run, {type(e).__name__}: {e}")

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(safe, order()))

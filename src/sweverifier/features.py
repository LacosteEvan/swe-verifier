"""What a verifier can know about a candidate patch without the hidden tests: the patch itself, the agent's trajectory
(did it submit, did its own checks pass), and how much the other candidates for the same issue agree with it."""
from __future__ import annotations

import json
import math
import re
from collections import Counter

from .patch import files

TEST_COMMAND = re.compile(r"pytest|runtests|unittest|manage\.py test|bin/test|\btox\b|-m test\b")
PYTEST_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors)\b")
UNITTEST_FAIL = re.compile(r"^FAILED \(|^Ran \d+ tests?", re.M)
SYMPY_SUMMARY = re.compile(r"tests finished: (\d+) passed(?:, (\d+) failed)?")


def changed_lines(patch: str) -> list[str]:
    """Added and removed lines, whitespace-normalized: two patches making the same edit compare equal."""
    return [("+" if line[0] == "+" else "-") + " ".join(line[1:].split())
            for line in patch.splitlines()
            if line[:1] in "+-" and not line.startswith(("+++", "---")) and line[1:].strip()]


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of the changed lines."""
    x, y = Counter(changed_lines(a)), Counter(changed_lines(b))
    union = sum((x | y).values())
    return sum((x & y).values()) / union if union else 0.0


def read_test_result(output: str) -> bool | None:
    """True / False when the output reads as a test run that passed / failed, None when it does not look like one."""
    m = SYMPY_SUMMARY.search(output)
    if m:
        return not (m.group(2) and int(m.group(2)))
    counts = Counter()
    for n, kind in PYTEST_SUMMARY.findall(output):
        counts["failed" if kind != "passed" else "passed"] += int(n)
    if counts:
        return counts["failed"] == 0
    if UNITTEST_FAIL.search(output):
        return "FAILED (" not in output and "\nOK" in output
    return None


def trajectory_features(trajectory: list[dict]) -> dict:
    calls, outputs = [], {}
    for m in trajectory:
        if m["role"] == "assistant":
            calls += m.get("tool_calls") or []
        elif m["role"] == "tool":
            outputs[m["tool_call_id"]] = m["content"]
    names = [c["function"]["name"] for c in calls]
    test_runs, script_runs, failed_edits = [], [], 0
    for c in calls:
        out = outputs.get(c["id"], "")
        name = c["function"]["name"]
        if name == "edit" and ("must occur exactly once" in out or out.startswith("No such file")):
            failed_edits += 1
        if name != "bash":
            continue
        try:
            cmd = json.loads(c["function"]["arguments"] or "{}").get("command", "")
        except json.JSONDecodeError:
            continue
        if TEST_COMMAND.search(cmd):
            test_runs.append(read_test_result(out))
        elif re.search(r"\bpython3? \S+\.py\b", cmd):
            script_runs.append("(exit code" not in out)
    known = [t for t in test_runs if t is not None]
    return {
        "tool_calls": len(calls),
        "edits": names.count("edit"),
        "failed_edits": failed_edits,
        "views": names.count("view") + names.count("outline"),
        "searches": names.count("search"),
        "test_runs": len(test_runs),
        "ran_tests": float(bool(test_runs)),
        "last_tests_passed": float(known[-1]) if known else 0.5,  # 0.5: the agent never got a readable test result
        "script_runs": len(script_runs),
        "last_script_ok": float(script_runs[-1]) if script_runs else 0.5,
    }


def patch_features(patch: str) -> dict:
    lines = patch.splitlines()
    added = sum(1 for x in lines if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))
    return {"empty": float(not patch.strip()), "files": len(files(patch)),
            "hunks": sum(x.startswith("@@") for x in lines), "added": added, "removed": removed,
            "log_size": math.log1p(added + removed)}


def agreement(patch: str, others: list[str]) -> dict:
    """How the other candidates for the same issue relate to this one: the more independent samples land on the same
    edit, the likelier it is right (self-consistency)."""
    sims = [similarity(patch, o) for o in others if o.strip()] if patch.strip() else []
    return {"agree_max": max(sims, default=0.0), "agree_mean": sum(sims) / len(sims) if sims else 0.0,
            "same_edit": float(sum(s > 0.99 for s in sims))}


def candidate_features(attempt: dict, trajectory: list[dict], others: list[str]) -> dict:
    return {"submitted": float(attempt["submitted"]), "steps": attempt["steps"],
            "hit_step_limit": float(not attempt["submitted"]), **trajectory_features(trajectory),
            **patch_features(attempt["patch"]), **agreement(attempt["patch"], others)}

"""An LLM as verifier: it reads the issue and one candidate patch, and says how likely the patch is to resolve it.
It sees no tests and no trajectory, only what a reviewer would see in a pull request."""
from __future__ import annotations

import json
import re

from .llm import LLM

PROMPT = """You review a patch proposed for an issue in {repo}. Judge whether it resolves the issue: the
repository's hidden tests for this issue will be run on it.

<issue>
{issue}
</issue>

<patch>
{patch}
</patch>

Reason briefly about whether the change addresses the root cause described in the issue, and whether it could break
existing behaviour. Then end with one line: PROBABILITY: <a number from 0 to 100 that the patch resolves the issue>"""

MAX_PATCH = 12000
MAX_ISSUE = 12000


class NoVerdict(Exception):
    pass


def judge(llm: LLM, instance: dict, patch: str) -> tuple[float, float, str]:
    """(probability 0..1, cost in $, the judge's text). An empty patch is 0 without a call."""
    if not patch.strip():
        return 0.0, 0.0, ""
    prompt = PROMPT.format(repo=instance["repo"], issue=instance["problem_statement"][:MAX_ISSUE],
                           patch=patch[:MAX_PATCH])
    reply = llm.chat([{"role": "user", "content": prompt}], max_tokens=8000)  # reasoning tokens count against it
    text = reply.message["content"]
    m = re.findall(r"PROBABILITY:\s*(\d+(?:\.\d+)?)", text or "")
    if not m:  # never a silent default: the candidate stays unjudged and a rerun asks again
        raise NoVerdict(f"no PROBABILITY line in {len(text or '')} characters, ${reply.cost_usd:.4f}")
    return min(100.0, float(m[-1])) / 100, reply.cost_usd, text


def to_row(instance_id: str, model: str, sample: int, judge_model: str, p: float, cost: float, text: str) -> str:
    return json.dumps({"instance_id": instance_id, "model": model, "sample": sample, "judge": judge_model,
                       "p": p, "cost_usd": cost, "text": text})

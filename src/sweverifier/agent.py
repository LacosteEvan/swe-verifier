"""The agent loop, written by hand: the model calls tools until it submits or runs out of steps. The patch is the
repository's git diff."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .env import Container
from .llm import LLM
from .patch import strip_tests
from .tools import TOOLS, Toolbox

SYSTEM = """You are a software engineer fixing an issue in a Python repository checked out at /testbed.

Work in this order:
1. Find the code involved: search, outline and view the relevant files.
2. Reproduce the problem with a small script before changing anything, when the issue allows it.
3. Fix the library's source code with the smallest change that resolves the issue. Do not modify existing tests.
4. Run your script again and the existing tests closest to your change to check the fix and that nothing else broke.
5. Call submit.

The project's Python environment is active and there is no network access. Your edits to tracked files are the
patch; scripts you create for yourself are ignored."""

MAX_STEPS = 40


@dataclass
class Attempt:
    instance_id: str
    model: str
    sample: int
    patch: str  # what is evaluated: the diff without test files
    submitted: bool
    steps: int
    cost_usd: float
    tokens_in: int
    tokens_out: int
    seconds: float
    error: str | None = None
    raw_patch: str = ""
    trajectory: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def solve(instance: dict, llm: LLM, sample: int = 0, max_steps: int = MAX_STEPS) -> Attempt:
    start = time.monotonic()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Issue in {instance['repo']}:\n\n{instance['problem_statement']}"}]
    cost = tokens_in = tokens_out = steps = 0
    submitted, error, raw = False, None, ""
    with Container(instance["image"]) as box:
        tools = Toolbox(box)
        try:
            while steps < max_steps and not submitted:
                steps += 1
                reply = llm.chat(messages, tools=TOOLS)
                cost += reply.cost_usd
                tokens_in, tokens_out = tokens_in + reply.tokens_in, tokens_out + reply.tokens_out
                messages.append(reply.message)
                calls = reply.message.get("tool_calls") or []
                if not calls:
                    messages.append({"role": "user", "content": "Continue with a tool call, or call submit when done."})
                    continue
                for call in calls:
                    name = call["function"]["name"]
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                    except json.JSONDecodeError as e:
                        output = f"Arguments are not valid JSON: {e}"
                    else:
                        if name == "submit":
                            submitted, output = True, "Submitted."
                        elif name in {t["function"]["name"] for t in TOOLS}:
                            output = tools.call(name, args)
                        else:
                            output = f"Unknown tool: {name}"
                    if steps == max_steps - 5:
                        output += "\n\n(5 steps left: finish the fix and call submit.)"
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            if not submitted:
                error = f"no submit within {max_steps} steps"
        except Exception as e:  # noqa: BLE001  a failed attempt is data for the verifier, not a crash of the run
            error = f"{type(e).__name__}: {e}"
        raw = box.diff()
    return Attempt(instance["instance_id"], llm.model, sample, strip_tests(raw), submitted, steps, cost, tokens_in,
                   tokens_out, time.monotonic() - start, error, raw, messages)

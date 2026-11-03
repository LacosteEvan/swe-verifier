"""Chat completions with tools through OpenRouter, with the cost of every call read from its response and a hard cap."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from openai import APIError, OpenAI

OPENROUTER = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reply:
    message: dict
    cost_usd: float
    tokens_in: int
    tokens_out: int


class Budget:
    """Shared across threads: a run stops before it can spend more than `max_usd`."""

    def __init__(self, max_usd: float):
        self.max_usd, self.spent = max_usd, 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"${self.max_usd:.2f} budget reached (${self.spent:.2f} spent)")

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent += usd


class LLM:
    def __init__(self, model: str, budget: Budget, temperature: float = 0.0, client: OpenAI | None = None):
        self.model, self.budget, self.temperature = model, budget, temperature
        self.client = client or OpenAI(base_url=OPENROUTER, api_key=os.getenv("OPENROUTER_API_KEY") or "missing",
                                       max_retries=3, timeout=300)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 4096) -> Reply:
        self.budget.check()
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, temperature=self.temperature,
                    max_tokens=max_tokens, extra_body={"usage": {"include": True}})
                if not resp.choices:
                    raise APIError("empty response", request=None, body=None)
                break
            except APIError:
                if attempt == 3:
                    raise
                time.sleep(5 * 2 ** attempt)
        usage = resp.usage
        cost = float(((usage.model_extra or {}) if usage else {}).get("cost") or 0.0)
        self.budget.add(cost)
        msg = resp.choices[0].message
        message = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            message["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                     for c in msg.tool_calls]
        return Reply(message, cost, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)

"""SWE-bench Lite, and the fixed order in which issues are taken for the subset."""
from __future__ import annotations

import random
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = "SWE-bench/SWE-bench_Lite"


@cache
def instances() -> dict[str, dict]:
    from datasets import load_dataset

    return {d["instance_id"]: d for d in load_dataset(DATASET, split="test")}


def order(seed: int = 0) -> list[str]:
    """A seeded shuffle of the 300 issues. The subset is the first N whose reference patch passes on this machine."""
    ids = sorted(instances())
    random.Random(seed).shuffle(ids)
    return ids

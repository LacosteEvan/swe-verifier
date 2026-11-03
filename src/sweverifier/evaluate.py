"""Score a patch with the official SWE-bench harness: apply it, apply the issue's tests, run them in the issue's
image."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .dataset import DATASET, ROOT

HARNESS_DIR = ROOT / "runs" / "harness"  # the harness writes its logs under the working directory


def evaluate(instance_id: str, patch: str, run_id: str, name: str, timeout: int = 1800) -> dict:
    """{'resolved', 'f2p': [passed, failed], 'p2p': [passed, failed], 'error'}: f2p are the tests the fix must turn
    green, p2p the tests that must stay green."""
    if not patch.strip():
        return {"resolved": False, "f2p": None, "p2p": None, "error": "empty patch"}
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", dir=HARNESS_DIR, delete=False) as f:
        f.write(json.dumps({"instance_id": instance_id, "model_name_or_path": name, "model_patch": patch}) + "\n")
        preds = Path(f.name)
    try:
        p = subprocess.run([sys.executable, "-m", "swebench.harness.run_evaluation", "-d", DATASET, "-p", str(preds),
                            "-i", instance_id, "--max_workers", "1", "-id", run_id, "-t", str(timeout)],
                           cwd=HARNESS_DIR, capture_output=True, text=True, timeout=timeout + 600)
    finally:
        preds.unlink(missing_ok=True)
    report = HARNESS_DIR / "logs" / "run_evaluation" / run_id / name / instance_id / "report.json"
    if not report.exists():
        return {"resolved": False, "f2p": None, "p2p": None, "error": f"no report (harness exit {p.returncode}): "
                f"{(p.stdout + p.stderr)[-300:]}"}
    r = json.loads(report.read_text())[instance_id]
    status = r.get("tests_status") or {}
    count = lambda k: [len(status[k]["success"]), len(status[k]["failure"])] if k in status else None  # noqa: E731
    return {"resolved": bool(r.get("resolved")), "f2p": count("FAIL_TO_PASS"), "p2p": count("PASS_TO_PASS"),
            "error": None if r.get("patch_successfully_applied", True) else "patch did not apply"}

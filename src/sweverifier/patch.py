"""Candidate patches: split a git diff by file, and drop the agent's edits to test files."""
from __future__ import annotations

import re

TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")


def files(patch: str) -> list[tuple[str, str]]:
    """(path, the file's section of the diff), in order."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    out = []
    for part in parts:
        m = re.match(r"diff --git a/(\S+) b/", part)
        if m:
            out.append((m.group(1), part))
    return out


def strip_tests(patch: str) -> str:
    """The harness applies the issue's own tests over the patch, so edits to test files would only conflict with
    them."""
    return "".join(section for path, section in files(patch) if not TEST_PATH.search(path))

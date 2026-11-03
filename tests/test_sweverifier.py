import numpy as np
import pandas as pd
import pytest

from sweverifier.features import agreement, read_test_result, similarity, trajectory_features
from sweverifier.judge import NoVerdict, judge
from sweverifier.patch import files, strip_tests
from sweverifier.tools import numbered, outline, truncate
from sweverifier.verifier import folds, selected, within_issue_auc

PATCH = """diff --git a/src/pkg/core.py b/src/pkg/core.py
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-assert x == 1
+assert x == 2
diff --git a/testing/logging/test_formatter.py b/testing/logging/test_formatter.py
--- a/testing/logging/test_formatter.py
+++ b/testing/logging/test_formatter.py
@@ -1 +1 @@
-a
+b
"""


def test_strip_tests_keeps_source_files_only():
    assert [p for p, _ in files(PATCH)] == ["src/pkg/core.py", "tests/test_core.py",
                                            "testing/logging/test_formatter.py"]
    assert [p for p, _ in files(strip_tests(PATCH))] == ["src/pkg/core.py"]


def test_test_results_are_read_from_pytest_django_and_sympy():
    assert read_test_result("==== 12 passed in 0.3s ====") is True
    assert read_test_result("==== 1 failed, 11 passed in 0.3s ====") is False
    assert read_test_result("Ran 4 tests in 0.01s\n\nOK") is True
    assert read_test_result("Ran 4 tests in 0.01s\n\nFAILED (failures=1)") is False
    assert read_test_result("tests finished: 30 passed, 2 failed, in 1.2 seconds") is False
    assert read_test_result("tests finished: 30 passed, in 1.2 seconds") is True
    assert read_test_result("hello") is None


def test_similarity_ignores_whitespace_and_counts_shared_edits():
    a = "--- a/f\n+++ b/f\n-x = 1\n+x =  2\n"
    b = "--- a/f\n+++ b/f\n-x = 1\n+x = 2\n"
    c = "--- a/f\n+++ b/f\n-y = 1\n+y = 3\n"
    assert similarity(a, b) == 1.0
    assert similarity(a, c) == 0.0
    assert agreement(a, [b, c, ""]) == {"agree_max": 1.0, "agree_mean": 0.5, "same_edit": 1.0}
    assert agreement("", [b])["agree_max"] == 0.0


def test_trajectory_features_find_the_last_test_run():
    call = lambda i, name, args: {"id": i, "type": "function", "function": {"name": name, "arguments": args}}  # noqa: E731
    traj = [
        {"role": "assistant", "content": "", "tool_calls": [call("1", "bash", '{"command": "python -m pytest t.py"}')]},
        {"role": "tool", "tool_call_id": "1", "content": "1 failed, 3 passed"},
        {"role": "assistant", "content": "", "tool_calls": [call("2", "edit", '{"path": "a.py"}')]},
        {"role": "tool", "tool_call_id": "2", "content": "old_str occurs 0 times in a.py; it must occur exactly once."},
        {"role": "assistant", "content": "", "tool_calls": [call("3", "bash", '{"command": "python -m pytest t.py"}')]},
        {"role": "tool", "tool_call_id": "3", "content": "4 passed"},
    ]
    f = trajectory_features(traj)
    assert (f["test_runs"], f["last_tests_passed"], f["failed_edits"], f["edits"]) == (2, 1.0, 1, 1)


def test_outline_lists_nested_definitions_with_lines():
    src = "class A:\n    @property\n    def f(self):\n        return 1\n\ndef g():\n    pass\n"
    assert outline(src).splitlines() == ["class A  (lines 1-4)", "  def f  (lines 2-4)", "def g  (lines 6-7)"]


def test_truncate_and_numbered():
    assert truncate("abc", 10) == "abc"
    t = truncate("x" * 100, 40)
    assert t.startswith("x" * 30) and "characters cut" in t
    assert numbered("a\nb", 9) == "     9  a\n    10  b"


def test_selection_shares_ties_and_folds_group_issues():
    df = pd.DataFrame({"instance_id": ["a", "a", "b", "b"], "resolved": [1.0, 0.0, 0.0, 0.0]})
    assert selected(df, np.zeros(4)).to_dict() == {"a": 0.5, "b": 0.0}
    assert selected(df, np.array([1, 0, 0, 0])).to_dict() == {"a": 1.0, "b": 0.0}
    f = folds(pd.DataFrame({"instance_id": list("aabbccddeeff")}), k=3)
    assert all(f[i] == f[i + 1] for i in range(0, 12, 2))


def test_within_issue_auc_ignores_telling_easy_issues_from_hard_ones():
    # issue a: every candidate right, issue b: every candidate wrong, issue c: one of two right
    df = pd.DataFrame({"instance_id": list("aabbcc"), "resolved": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]})
    easy_vs_hard = np.array([0.9, 0.9, 0.1, 0.1, 0.5, 0.5])  # perfect pooled ranking, no help within c
    assert within_issue_auc(df, easy_vs_hard) == 0.5
    assert within_issue_auc(df, np.array([0, 0, 0, 0, 0.6, 0.4])) == 1.0


class _Reply:
    def __init__(self, content):
        self.message, self.cost_usd = {"role": "assistant", "content": content}, 0.01


class _LLM:
    def __init__(self, content):
        self.content = content

    def chat(self, messages, max_tokens):
        return _Reply(self.content)


def test_judge_reads_the_last_probability_and_never_defaults():
    inst = {"repo": "r", "problem_statement": "issue"}
    assert judge(_LLM("PROBABILITY: 10 ... on reflection\nPROBABILITY: 85"), inst, "+x")[0] == 0.85
    assert judge(_LLM("anything"), inst, "   ")[0] == 0.0  # an empty patch is not sent
    for content in ("", None, "cut off before the verdict"):  # reasoning can eat the whole token budget
        with pytest.raises(NoVerdict):
            judge(_LLM(content), inst, "+x")

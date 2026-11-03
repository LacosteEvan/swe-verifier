"""The agent's tools: a shell, a file viewer, a code search, a tree-sitter outline, an exact-match editor, and
submit."""
from __future__ import annotations

import tree_sitter_python
from tree_sitter import Language, Parser

from .env import Container, quote

PY = Parser(Language(tree_sitter_python.language()))
MAX_OUTPUT = 8000  # characters of tool output the model sees
VIEW_LINES = 250

SCHEMAS = [
    {"name": "bash", "description": "Run a shell command from the repository root (/testbed), with the project's "
     "Python environment active and no network. Use it to run Python scripts and tests, or to list files. Output is "
     "truncated to the first and last few thousand characters.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"},
         "timeout": {"type": "integer", "description": "seconds, default 120, at most 600"}},
         "required": ["command"]}},
    {"name": "view", "description": f"Show a file with line numbers, {VIEW_LINES} lines at most per call.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "relative to the repository root"},
         "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
         "required": ["path"]}},
    {"name": "search", "description": "Search the repository for a regular expression (grep -E). Returns "
     "file:line:text, at most 60 matches.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"},
         "path": {"type": "string", "description": "file or directory, default the whole repository"},
         "glob": {"type": "string", "description": "file name filter, default *.py"}},
         "required": ["pattern"]}},
    {"name": "outline", "description": "List the classes, functions and methods of a Python file with their line "
     "ranges, from its syntax tree. Cheaper than viewing a large file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "edit", "description": "Replace an exact snippet of a file. old_str must occur exactly once; include a "
     "few lines of context to make it unique. With an empty old_str, creates the file.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
         "required": ["path", "old_str", "new_str"]}},
    {"name": "submit", "description": "Finish: your changes to the repository's files are the patch. Call it once the "
     "fix is in place and checked.", "parameters": {"type": "object", "properties": {}}},
]
TOOLS = [{"type": "function", "function": s} for s in SCHEMAS]


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 3 // 4], text[-limit // 4:]
    return f"{head}\n... ({len(text) - len(head) - len(tail):,} characters cut) ...\n{tail}"


def numbered(text: str, start: int = 1) -> str:
    return "\n".join(f"{start + i:6}  {line}" for i, line in enumerate(text.splitlines()))


def outline(source: str) -> str:
    """Classes and functions, nested by indentation, with their 1-based line ranges."""
    lines = []

    def walk(node, depth):
        for child in node.children:
            target = child.child_by_field_name("definition") if child.type == "decorated_definition" else child
            if target is not None and target.type in ("class_definition", "function_definition"):
                kind = "class" if target.type == "class_definition" else "def"
                name = target.child_by_field_name("name").text.decode()
                span = f"{child.start_point[0] + 1}-{child.end_point[0] + 1}"
                lines.append(f"{'  ' * depth}{kind} {name}  (lines {span})")
                body = target.child_by_field_name("body")
                if body is not None:
                    walk(body, depth + 1)

    walk(PY.parse(source.encode()).root_node, 0)
    return "\n".join(lines) or "(no classes or functions)"


class Toolbox:
    def __init__(self, box: Container):
        self.box = box

    def call(self, name: str, args: dict) -> str:
        try:
            return getattr(self, f"_{name}")(**args)
        except TypeError as e:
            return f"Invalid arguments for {name}: {e}"

    def _bash(self, command: str, timeout: int = 120) -> str:
        r = self.box.run(command, timeout=max(1, min(int(timeout), 600)))
        return truncate(r.output) + (f"\n(exit code {r.exit_code})" if r.exit_code else "") or "(no output)"

    def _view(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        text = self.box.read(path)
        if text is None:
            return f"No such file: {path}"
        lines = text.splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line) if end_line else start + VIEW_LINES - 1, start + VIEW_LINES - 1)
        more = f"\n({len(lines)} lines in total)" if end < len(lines) or start > 1 else ""
        return numbered("\n".join(lines[start - 1:end]), start) + more

    def _search(self, pattern: str, path: str = ".", glob: str = "*.py") -> str:
        r = self.box.run(f"grep -rnE --include={quote(glob)} -e {quote(pattern)} -- {quote(path)} | head -n 61")
        hits = r.output.strip().splitlines()
        if not hits:
            return "No matches."
        more = "\n(more than 60 matches: narrow the search)" if len(hits) > 60 else ""
        return truncate("\n".join(hits[:60]) + more)

    def _outline(self, path: str) -> str:
        text = self.box.read(path)
        return f"No such file: {path}" if text is None else truncate(outline(text))

    def _edit(self, path: str, old_str: str, new_str: str) -> str:
        text = self.box.read(path)
        if not old_str:
            if text is not None:
                return f"{path} exists: give an old_str to edit it."
            self.box.write(path, new_str)
            return f"Created {path}."
        if text is None:
            return f"No such file: {path}"
        count = text.count(old_str)
        if count != 1:
            return f"old_str occurs {count} times in {path}; it must occur exactly once."
        updated = text.replace(old_str, new_str)
        self.box.write(path, updated)
        at = text[: text.index(old_str)].count("\n") + 1
        lines = updated.splitlines()
        lo, hi = max(1, at - 3), min(len(lines), at + new_str.count("\n") + 3)
        return f"Edited {path}. Lines {lo}-{hi} now read:\n" + numbered("\n".join(lines[lo - 1:hi]), lo)

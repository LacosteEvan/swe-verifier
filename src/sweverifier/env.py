"""A SWE-bench task container: the repository at /testbed in the issue's own evaluation image, driven by docker exec."""
from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass

PLATFORM = "linux/amd64"  # SWE-bench publishes x86 images only; Docker emulates them on Apple silicon
WORKDIR = "/testbed"


@dataclass
class Result:
    output: str
    exit_code: int

    @property
    def timed_out(self) -> bool:
        return self.exit_code == 124


def pull(image: str, tries: int = 5) -> None:
    """Docker Hub drops a pull now and then: retry with backoff before giving up."""
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return
    for i in range(tries):
        r = subprocess.run(["docker", "pull", "-q", "--platform", PLATFORM, image], capture_output=True, text=True)
        if r.returncode == 0:
            return
        time.sleep(15 * 2**i)
    raise RuntimeError(f"docker pull {image} failed {tries} times: {r.stderr.strip()[-300:]}")


def quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


class Container:
    """No network: the agent works from the repository and the issue only."""

    def __init__(self, image: str):
        self.image = image
        self.name = f"swev-{uuid.uuid4().hex[:10]}"

    def __enter__(self) -> Container:
        pull(self.image)
        subprocess.run(["docker", "run", "-d", "--rm", "--platform", PLATFORM, "--name", self.name, "--network", "none",
                        "-w", WORKDIR, self.image, "sleep", "infinity"], check=True, capture_output=True)
        return self

    def __exit__(self, *exc) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def run(self, command: str, timeout: int = 120, stdin: bytes | None = None) -> Result:
        """A login shell (the image's conda environment is active) from the repository root, killed after `timeout`."""
        script = f"cd {WORKDIR} && {command}"
        args = ["docker", "exec", *(["-i"] if stdin is not None else []), self.name,
                "timeout", "-k", "5", str(timeout), "bash", "-lc", script]
        try:
            p = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout + 30)
        except subprocess.TimeoutExpired:
            return Result(f"(timed out after {timeout} s)", 124)
        out = (p.stdout + p.stderr).decode("utf-8", "replace")
        if p.returncode == 124:
            out += f"\n(timed out after {timeout} s)"
        return Result(out, p.returncode)

    def read(self, path: str) -> str | None:
        r = self.run(f"cat -- {quote(path)}")
        return r.output if r.exit_code == 0 else None

    def write(self, path: str, text: str) -> None:
        self.run(f"cat > {quote(path)}", stdin=text.encode())

    def diff(self) -> str:
        """The candidate patch: tracked files only, so the agent's scratch scripts are left out."""
        return self.run("git -c core.fileMode=false diff --no-color").output

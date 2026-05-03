"""End-to-end live run: drive LucidCoder through a real toy task using a
local subprocess as the 'Green orchestrator'. Uses real LLM calls.

Setup: a temp task dir is created with:
  - README.md describing what to do
  - failing-test scaffolding (tests/test_solution.py)
  - run-tests.sh that runs pytest

The agent must:
  1. Read README + tests
  2. Author solution.py implementing the function
  3. Make tests pass

Run: python tests/e2e_local_run.py
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv

WORKSPACE = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE / ".env")
sys.path.insert(0, str(WORKSPACE))

from src import controller, protocol, session as session_mod  # noqa: E402


TASK_INSTRUCTION = textwrap.dedent("""\
    Implement a function `kebab_case(s: str) -> str` in a new file `solution.py`
    at the task root. It should convert any whitespace-separated or
    camelCase / snake_case string into kebab-case (lowercase with hyphens).

    Examples:
      kebab_case("Hello World")     -> "hello-world"
      kebab_case("fooBarBaz")       -> "foo-bar-baz"
      kebab_case("snake_case_str")  -> "snake-case-str"
      kebab_case("  Mixed  Stuff ") -> "mixed-stuff"

    The test suite is in tests/test_solution.py. Run `bash run-tests.sh`
    to verify; it must exit 0 with all tests passing.
""")


SOLUTION_TEST = textwrap.dedent("""\
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from solution import kebab_case


    def test_simple_words():
        assert kebab_case("Hello World") == "hello-world"


    def test_camel_case():
        assert kebab_case("fooBarBaz") == "foo-bar-baz"


    def test_snake_case():
        assert kebab_case("snake_case_str") == "snake-case-str"


    def test_extra_whitespace():
        assert kebab_case("  Mixed  Stuff ") == "mixed-stuff"


    def test_already_kebab():
        assert kebab_case("already-kebab") == "already-kebab"
""")


# Use python3 (WSL/Linux/Docker friendly). LF endings enforced via newline="\n".
RUN_TESTS_SH = (
    "#!/usr/bin/env bash\n"
    "set -e\n"
    'cd "$(dirname "$0")"\n'
    "python3 -m pytest tests/ -q\n"
)


def _write_lf(path: Path, content: str) -> None:
    """Write a text file with LF line endings (matters on Windows where the
    default would be CRLF — bash on WSL/Linux can't parse CRLF shell scripts)."""
    path.write_bytes(content.replace("\r\n", "\n").encode("utf-8"))


def make_task_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="lucidcoder-e2e-"))
    _write_lf(d / "README.md", TASK_INSTRUCTION)
    (d / "tests").mkdir()
    _write_lf(d / "tests" / "test_solution.py", SOLUTION_TEST)
    _write_lf(d / "run-tests.sh", RUN_TESTS_SH)
    try:
        os.chmod(d / "run-tests.sh", 0o755)
    except Exception:
        pass
    return d


def run_command(cwd: Path, command: str, timeout: int) -> dict:
    """Mimic the Green's exec: run the command in a bash subprocess, capture."""
    started = time.time()
    bash = shutil.which("bash") or "bash"
    try:
        proc = subprocess.run(
            [bash, "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return {
            "kind": "exec_result",
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "kind": "exec_result",
            "exit_code": 124,
            "stdout": (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout}s",
        }
    except Exception as e:
        return {
            "kind": "exec_result",
            "exit_code": 127,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
        }
    finally:
        elapsed = time.time() - started
        print(f"  [exec {elapsed:.1f}s]")


def main(max_turns: int = 60):
    task_dir = make_task_dir()
    print(f"\n=== Task dir: {task_dir} ===\n")
    print(TASK_INSTRUCTION)
    print(f"\n=== Starting LucidCoder run (max {max_turns} turns) ===\n")

    sess = session_mod.get_or_create("e2e-run-1")

    # Turn 1: send the task
    inbound = {"kind": "task", "instruction": TASK_INSTRUCTION}
    for turn in range(1, max_turns + 1):
        outbound = controller.step(sess, inbound)
        kind = outbound.get("kind")

        if kind == "final":
            print(f"\n[Turn {turn}] FINAL")
            print("-" * 60)
            print(outbound.get("output", ""))
            print("-" * 60)
            print(f"\nScratch dir: workspace/scratch/{sess.session_id}/")

            # Final ground-truth: run the verify command ourselves
            verify_cmd = (sess.decompose.get("verify_cmd") or "bash run-tests.sh").strip()
            ground = run_command(task_dir, verify_cmd, 60)
            print(f"\nGround-truth verify: exit_code={ground['exit_code']}")
            print("stdout:", ground["stdout"][-500:])
            print("stderr:", ground["stderr"][-500:])
            return ground["exit_code"] == 0

        if kind != "exec_request":
            print(f"[Turn {turn}] UNEXPECTED kind={kind}; payload={outbound}")
            return False

        cmd = outbound["command"]
        timeout = outbound.get("timeout", 30)
        # Truncate long commands for printing
        printed = cmd if len(cmd) < 200 else (cmd[:160] + " ...")
        print(f"[Turn {turn}] stage={sess.stage} step_idx={sess.step_idx} cmd:")
        print(f"  $ {printed}")

        result = run_command(task_dir, cmd, timeout)
        rc = result["exit_code"]
        out_tail = result["stdout"][-200:].replace("\n", " | ") if result["stdout"] else ""
        err_tail = result["stderr"][-200:].replace("\n", " | ") if result["stderr"] else ""
        print(f"  exit={rc} out={out_tail!r} err={err_tail!r}")
        inbound = result

    print(f"\nReached max_turns={max_turns} without final. Stage={sess.stage}")
    return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

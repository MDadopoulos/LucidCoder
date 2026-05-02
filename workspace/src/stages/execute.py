"""Stage 3 — EXECUTE.

Per-step execution helpers. Decisions:

- Each step is one exec_request to the green orchestrator. The result arrives
  on the next A2A turn — the controller is responsible for the turn boundary;
  this module just supplies the next command + judges the previous result.

- Fix-attempt limit = 3 per step. After 3 failed attempts:
   * fail_action="abort"   -> session goes to "failed", emits final
   * fail_action="diagnose" -> session goes to "diagnose" (re-plan with gap)
   * fail_action="retry"   -> treated like diagnose after 3 attempts

- Anti-paralysis guard: if 5 consecutive read-only commands have been issued
  without any write, force a write or trigger diagnose.

- A "read-only" command is one that does not start with: cat-write (`>`,
  `>>`, `tee`), package install, sed -i, mv, cp, rm, mkdir, touch, python
  with a script that writes, or a heredoc.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src import prompts, scratch
from src.model import get_model
from src.session import ExecRecord, Session

logger = logging.getLogger(__name__)


READ_ONLY_PREFIXES = (
    "ls", "cat", "head", "tail", "grep", "rg", "find", "which",
    "pwd", "echo", "printenv", "env", "wc", "file", "stat",
    "diff", "git status", "git diff", "git log", "git show",
)

WRITE_INDICATORS = (
    " > ", " >> ", " | tee ", " | tee -a", "| sudo tee",
    " sed -i", " mv ", " cp ", " rm ", " mkdir ", " touch ",
    " chmod ", " chown ", " ln -s", " patch ",
    " pip install", " uv add", " uv pip install", " npm install",
    " apt ", " apt-get ", " yum ", " pacman ",
    "<<EOF", "<<'EOF'", "<<\"EOF\"",
    " python -c", " python3 -c", " bash -c",
    " git apply", " git commit", " git add", " git checkout -b",
)


def is_read_only(cmd: str) -> bool:
    s = (cmd or "").strip()
    if not s:
        return True
    # Cheap heuristics — any write indicator -> not read-only
    for ind in WRITE_INDICATORS:
        if ind in (" " + s):
            return False
    # Any redirection writes
    if re.search(r"(?<!\d)>\s*\S", s) or ">>" in s:
        return False
    # Shell builtin write-y constructs
    if "<<" in s and ("EOF" in s or "'EOF'" in s):
        return False
    # Otherwise: starts with a known read-only program?
    first = s.split()[0]
    for p in READ_ONLY_PREFIXES:
        if first == p or s.startswith(p + " "):
            return True
    # default: assume might write
    return False


def current_step(s: Session) -> dict[str, Any] | None:
    if 0 <= s.step_idx < len(s.plan):
        return s.plan[s.step_idx]
    return None


def step_passes_expectations(step: dict[str, Any], result: ExecRecord) -> tuple[bool, str]:
    expected_exit = step.get("expected_exit", 0)
    if expected_exit is not None and result.exit_code != expected_exit:
        return False, f"exit code {result.exit_code} != expected {expected_exit}"
    needle = step.get("expected_stdout_contains")
    if needle:
        if needle not in (result.stdout or ""):
            return False, f"stdout missing expected substring: {needle!r}"
    return True, "ok"


def propose_fix(s: Session, step: dict[str, Any], result: ExecRecord) -> dict[str, Any]:
    """Ask the model for a single fix command."""
    model = get_model()
    user = prompts.FIX_USER_TEMPLATE.format(
        decompose_json=json.dumps(s.decompose, ensure_ascii=False, indent=2),
        step_json=json.dumps(step, ensure_ascii=False, indent=2),
        exit_code=result.exit_code,
        stdout=(result.stdout or "")[-2000:],
        stderr=(result.stderr or "")[-2000:],
        attempt=s.attempts_for(s.step_idx),
    )
    try:
        fix = model.generate_json(
            system=prompts.FIX_SYSTEM,
            user=user,
            temperature=0.25,
            max_output_tokens=800,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("FIX call failed (%s); diagnosing", e)
        return {"command": "", "diagnose": True, "intent": "fix-call-failed", "rerun_step_after": False, "timeout": 30}
    if not isinstance(fix, dict):
        return {"command": "", "diagnose": True, "intent": "fix-non-dict", "rerun_step_after": False, "timeout": 30}
    fix.setdefault("command", "")
    fix.setdefault("intent", "fix")
    fix.setdefault("rerun_step_after", False)
    fix.setdefault("timeout", 30)
    fix.setdefault("diagnose", False)
    try:
        fix["timeout"] = max(1, min(int(fix["timeout"]), 300))
    except (TypeError, ValueError):
        fix["timeout"] = 30
    return fix


def append_step_record(s: Session, rec: ExecRecord) -> None:
    s.step_records.append(rec)
    scratch.append_jsonl(s.session_id, "trace.jsonl", rec.to_dict())
    if is_read_only(rec.command):
        s.read_only_streak += 1
    else:
        s.read_only_streak = 0

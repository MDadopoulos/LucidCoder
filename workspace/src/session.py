"""Per-A2A-context session state.

The terminal-bench-shell-v1 protocol is multi-turn: the Green sends a `task`,
then alternates `exec_result` <-> `exec_request` until the purple sends
`final`. Each exchange is one A2A turn, so the controller cannot run all
four stages in a single function call — it must persist state across turns.

`Session` holds:
  - `instruction`            (task text)
  - `stage`                  current stage in {"probe", "plan", "execute", "verify", "diagnose", "done"}
  - `probe_results`          list of (cmd, exit, stdout, stderr) from the canned + LLM probes
  - `decompose`              dict from Stage 1 (truths, artifacts, key_links, verify_cmd)
  - `plan`                   list of Step dicts from Stage 2
  - `step_idx`               int — next step to execute
  - `step_attempts`          {step_idx: int} — retry counter per step
  - `read_only_streak`       int — anti-paralysis guard
  - `verify_attempts`        int
  - `pending_command`        the command we just emitted (awaiting result)
  - `last_result`            most recent exec_result
  - `summary_so_far`         running list of step outcomes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecRecord:
    stage: str
    intent: str
    command: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "intent": self.intent,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_tail": (self.stdout or "")[-2000:],
            "stderr_tail": (self.stderr or "")[-2000:],
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }


@dataclass
class Session:
    session_id: str
    instruction: str = ""
    stage: str = "init"  # init -> probe -> plan -> execute -> verify -> done|failed
    probe_idx: int = 0
    probe_records: list[ExecRecord] = field(default_factory=list)
    decompose: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)
    step_idx: int = 0
    step_attempts: dict[int, int] = field(default_factory=dict)
    step_records: list[ExecRecord] = field(default_factory=list)
    read_only_streak: int = 0
    verify_attempts: int = 0
    verify_records: list[ExecRecord] = field(default_factory=list)
    replan_count: int = 0
    pending: ExecRecord | None = None
    started_at: float = field(default_factory=time.time)
    final_summary: str = ""

    def attempts_for(self, idx: int) -> int:
        return self.step_attempts.get(idx, 0)

    def bump_attempts(self, idx: int) -> int:
        n = self.step_attempts.get(idx, 0) + 1
        self.step_attempts[idx] = n
        return n

    def elapsed_s(self) -> float:
        return time.time() - self.started_at


# Global registry: context_id -> Session
_SESSIONS: dict[str, Session] = {}


def get_or_create(context_id: str) -> Session:
    if context_id not in _SESSIONS:
        _SESSIONS[context_id] = Session(session_id=context_id)
    return _SESSIONS[context_id]


def drop(context_id: str) -> None:
    _SESSIONS.pop(context_id, None)

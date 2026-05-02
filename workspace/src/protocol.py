"""terminal-bench-shell-v1 protocol encoder/decoder.

The Green agent (terminal-bench orchestrator) and our purple agent communicate
via single A2A text messages whose payload is a JSON object. Three message
kinds are defined:

  Inbound (green -> purple):
    { "kind": "task",
      "protocol": "terminal-bench-shell-v1",
      "instruction": "<natural-language task description>" }

    { "kind": "exec_result",
      "exit_code": <int>,
      "stdout": "<str>",
      "stderr": "<str>" }

  Outbound (purple -> green):
    { "kind": "exec_request",
      "command": "<shell command>",
      "timeout": <int 1..300> }

    { "kind": "final",
      "output": "<str summary>" }

If a payload is not valid JSON, the green decodes it as a final message
with the raw text as `output`. We never rely on that — always emit JSON.
"""

from __future__ import annotations

import json
from typing import Any, Literal


PROTOCOL_VERSION = "terminal-bench-shell-v1"


def encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def decode(raw: str) -> dict[str, Any]:
    """Decode an inbound message; tolerate non-JSON by treating it as final-text."""
    raw = (raw or "").strip()
    if not raw:
        return {"kind": "exec_result", "exit_code": 0, "stdout": "", "stderr": ""}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"kind": "task", "protocol": PROTOCOL_VERSION, "instruction": raw}
    if not isinstance(payload, dict):
        return {"kind": "task", "protocol": PROTOCOL_VERSION, "instruction": str(payload)}
    return payload


def make_exec_request(command: str, timeout: int = 30) -> dict[str, Any]:
    timeout = max(1, min(int(timeout), 300))
    return {"kind": "exec_request", "command": command, "timeout": timeout}


def make_final(output: str) -> dict[str, Any]:
    return {"kind": "final", "output": output or ""}


Kind = Literal["task", "exec_result", "exec_request", "final"]

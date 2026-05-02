"""Per-session scratch directory lifecycle.

Mirrors the VeritasX pattern: each session gets `scratch/{context_id}/` with
its own decompose.md, plan.md, trace.jsonl, and verify.md. Wiped on first
turn so re-issued context_ids start clean.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _scratch_root() -> Path:
    root = Path(os.environ.get("SCRATCH_DIR", "")).expanduser()
    if not root:
        root = Path(__file__).resolve().parent.parent / "scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def session_dir(session_id: str) -> Path:
    return _scratch_root() / _safe_name(session_id)


def prepare(session_id: str) -> Path:
    d = session_dir(session_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_text(session_id: str, name: str, content: str) -> None:
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(content, encoding="utf-8")


def append_jsonl(session_id: str, name: str, record: dict[str, Any]) -> None:
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    with (d / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_text(session_id: str, name: str) -> str:
    p = session_dir(session_id) / name
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:128]

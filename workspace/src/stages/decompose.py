"""Stage 1 — DECOMPOSE.

Strategy: deterministic canned probe sequence first, then ONE LLM call to derive
must_haves. The probes are designed to surface 90% of TB-2 task signals without
spending model tokens:

  1. `pwd && ls -la` (orient)
  2. recursive find for code-relevant files (depth 4, capped)
  3. cat README*, cat task description, cat any test/run-tests script
  4. detect package manager (uv, pip, npm) by lockfile presence
  5. enumerate test entrypoints (pytest --collect-only if Python; package.json
     scripts if Node)

Each probe maps to one A2A turn. The session.probe_idx tracks progress.
After all canned probes finish, we call decompose_with_llm() once. If the LLM
returns extra_probes, we run them and call again (max 1 extra round).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src import prompts, scratch
from src.model import get_model
from src.session import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canned probe sequence
# ---------------------------------------------------------------------------

# Each entry: (intent, command, timeout)
# We deliberately keep these tight — most TB-2 tasks expose a README, a
# tests/ dir, and a run-tests.sh; we don't need to exhaustively introspect.
CANNED_PROBES: list[tuple[str, str, int]] = [
    (
        "orient: pwd + top-level listing",
        "pwd && ls -la",
        15,
    ),
    (
        "discover: code-relevant files (depth 4)",
        # Use printf-friendly find; cap output.
        "find . -maxdepth 4 -type f \\( "
        "-name '*.py' -o -name '*.sh' -o -name '*.md' "
        "-o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' "
        "-o -name 'Dockerfile' -o -name 'Makefile' "
        "-o -name '*.json' -o -name '*.txt' "
        "-o -name '*.js' -o -name '*.ts' -o -name '*.go' -o -name '*.rs' "
        "-o -name '*.c' -o -name '*.h' -o -name '*.cpp' "
        "\\) 2>/dev/null | head -120",
        30,
    ),
    (
        "read: README and task instructions",
        "for f in README.md README.txt README INSTRUCTIONS.md TASK.md task.md NOTES.md; do "
        "if [ -f \"$f\" ]; then echo \"=== $f ===\"; cat \"$f\"; fi; done; "
        "find . -maxdepth 3 -name 'task.yaml' -exec sh -c 'echo \"=== $1 ===\"; cat \"$1\"' _ {} \\; 2>/dev/null | head -300",
        20,
    ),
    (
        "read: test scripts and entrypoints",
        "for f in run-tests.sh tests/run-tests.sh setup.sh tests/setup.sh "
        "Makefile package.json pyproject.toml requirements.txt setup.py setup.cfg; do "
        "if [ -f \"$f\" ]; then echo \"=== $f ===\"; head -100 \"$f\"; fi; done; "
        "if [ -d tests ]; then echo '=== tests/ ==='; ls -la tests/ | head -30; fi",
        20,
    ),
    (
        "detect: lockfiles + tooling hints",
        "echo '=== lockfiles ==='; ls -1 uv.lock poetry.lock requirements*.txt "
        "package-lock.json pnpm-lock.yaml yarn.lock Cargo.lock go.sum 2>/dev/null; "
        "echo '=== which interpreters ==='; which python3 uv pip pytest node npm cargo go 2>/dev/null",
        15,
    ),
]


def first_probe(s: Session) -> tuple[str, str, int]:
    """Return the first canned probe (intent, cmd, timeout)."""
    return CANNED_PROBES[0]


def next_probe_or_none(s: Session) -> tuple[str, str, int] | None:
    """Return the next canned probe, or None if all canned probes done."""
    if s.probe_idx >= len(CANNED_PROBES):
        return None
    return CANNED_PROBES[s.probe_idx]


def has_more_canned(s: Session) -> bool:
    return s.probe_idx < len(CANNED_PROBES)


# ---------------------------------------------------------------------------
# LLM decompose call
# ---------------------------------------------------------------------------

def _format_probe_block(s: Session, max_chars: int = 12000) -> str:
    """Render probe results into a single text block for the LLM, oldest first."""
    chunks: list[str] = []
    for rec in s.probe_records:
        block = (
            f"$ {rec.command}\n"
            f"# intent: {rec.intent}\n"
            f"# exit={rec.exit_code} elapsed={rec.elapsed_ms}ms\n"
            f"--- stdout ---\n{(rec.stdout or '').strip()}\n"
            f"--- stderr ---\n{(rec.stderr or '').strip()}\n"
        )
        chunks.append(block)
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[-max_chars:]
        text = "...(truncated)...\n" + text
    return text


def derive(s: Session) -> dict[str, Any]:
    """Call the model to produce the DECOMPOSE JSON. Persists to scratch."""
    model = get_model()
    user = prompts.DECOMPOSE_USER_TEMPLATE.format(
        instruction=s.instruction.strip(),
        probe_block=_format_probe_block(s),
    )
    decompose = model.generate_json(
        system=prompts.DECOMPOSE_SYSTEM,
        user=user,
        temperature=0.15,
        max_output_tokens=2048,
    )
    if not isinstance(decompose, dict):
        raise ValueError(f"DECOMPOSE returned non-dict: {decompose!r}")

    # Defensive defaults
    decompose.setdefault("summary", "")
    decompose.setdefault("truths", [])
    decompose.setdefault("artifacts", [])
    decompose.setdefault("key_links", [])
    decompose.setdefault("verify_cmd", "")
    decompose.setdefault("verify_cwd", ".")
    decompose.setdefault("extra_probes", [])
    decompose.setdefault("confidence", "medium")

    scratch.write_text(s.session_id, "decompose.md", _render_decompose(decompose))
    return decompose


def _render_decompose(d: dict[str, Any]) -> str:
    parts = [
        "# Decompose\n",
        f"## Summary\n{d.get('summary', '').strip()}\n",
        "## Truths\n" + "\n".join(f"- {t}" for t in d.get("truths", [])) + "\n",
        "## Artifacts\n" + "\n".join(
            f"- `{a.get('path')}` exists={a.get('must_exist')} modify={a.get('must_modify')} — {a.get('note', '')}"
            for a in d.get("artifacts", [])
        ) + "\n",
        "## Key Links\n" + "\n".join(
            f"- `{kl.get('from')}` -> `{kl.get('to')}` via `{kl.get('via')}`"
            for kl in d.get("key_links", [])
        ) + "\n",
        f"## Verify Command\n```\n{d.get('verify_cmd', '')}\n```\n(cwd: `{d.get('verify_cwd', '.')}`)\n",
        f"## Confidence\n{d.get('confidence', '')}\n",
    ]
    return "\n".join(parts)

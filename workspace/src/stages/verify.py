"""Stage 4 — VERIFY.

Two-phase verification:

  Phase A — RUN: emit `verify_cmd` (from decompose) and capture exit/stdout/stderr.
  Phase B — JUDGE: combine the verify result + an anti-pattern grep + an
            artifact-check + ask the model to render PASS/FAIL.

Anti-pattern grep panel (cribbed from GSD verifier and superpowers
verification-before-completion): a single shell command we emit between the
verify run and the LLM judge, scanning *modified* paths only.

Artifact check: for each must_exist artifact in decompose.artifacts, emit a
`test -e <path>` style probe.

To stay within the A2A turn budget, we batch grep + artifact checks into a
single shell command. The LLM judge then sees verify + scan together.
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import Any

from src import prompts, scratch
from src.model import get_model
from src.session import ExecRecord, Session

logger = logging.getLogger(__name__)


# Two-stage verify state lives in session via verify_records list.
#   verify_records[0] -> the verify_cmd run
#   verify_records[1] -> the anti-pattern + artifact-check command
# After both records exist, we judge.


def verify_command(s: Session) -> tuple[str, str, int]:
    """Return (intent, shell_cmd, timeout) for the verify_cmd step."""
    cmd = (s.decompose.get("verify_cmd") or "").strip()
    cwd = (s.decompose.get("verify_cwd") or ".").strip()
    if not cmd:
        # Fallback heuristics
        cmd = "if [ -x ./run-tests.sh ]; then bash ./run-tests.sh; "\
              "elif [ -x tests/run-tests.sh ]; then bash tests/run-tests.sh; "\
              "elif command -v pytest >/dev/null 2>&1; then pytest -q tests/ || pytest -q; "\
              "else echo 'NO_VERIFY_CMD'; exit 2; fi"
    if cwd and cwd != "." and cwd != "":
        wrapped = f"cd {shlex.quote(cwd)} && ( {cmd} )"
    else:
        wrapped = cmd
    return ("run verify_cmd", wrapped, 240)


def panel_command(s: Session) -> tuple[str, str, int]:
    """Build a single shell command that scans for anti-patterns and checks artifact existence."""
    paths = []
    for st in s.plan:
        for w in st.get("writes", []) or []:
            if isinstance(w, str) and w.strip():
                paths.append(w.strip())
    artifacts = []
    for a in s.decompose.get("artifacts", []) or []:
        p = a.get("path") if isinstance(a, dict) else None
        if isinstance(p, str) and p.strip():
            artifacts.append((p.strip(), a.get("must_exist", True)))

    # Anti-pattern grep — scan written files (or fall back to all py/sh files)
    if paths:
        scan_targets = " ".join(shlex.quote(p) for p in paths if p)
        grep_cmd = (
            "echo '=== ANTI-PATTERN GREP ==='; "
            f"for p in {scan_targets}; do "
            "if [ -f \"$p\" ]; then "
            "echo \"--- $p ---\"; "
            "grep -nE '(TODO|FIXME|XXX|placeholder|NotImplementedError|raise\\s+NotImplemented|return\\s+None\\s*#\\s*placeholder|pass\\s*#\\s*todo)' \"$p\" || echo '(clean)'; "
            "fi; done; "
        )
    else:
        grep_cmd = (
            "echo '=== ANTI-PATTERN GREP (fallback: cwd) ==='; "
            "grep -rnE --include='*.py' --include='*.sh' "
            "'(TODO|FIXME|XXX|placeholder|NotImplementedError|raise\\s+NotImplemented)' . 2>/dev/null | head -50 || echo '(clean)'; "
        )

    artifact_cmd = "echo '=== ARTIFACT CHECK ==='; "
    if artifacts:
        for path, must_exist in artifacts:
            qp = shlex.quote(path)
            artifact_cmd += (
                f"if [ -e {qp} ]; then echo 'EXISTS: {path}'; "
                f"if [ -f {qp} ]; then echo \"  size=$(wc -c < {qp}) lines=$(wc -l < {qp})\"; fi; "
                f"else echo 'MISSING: {path}'; fi; "
            )
    else:
        artifact_cmd += "echo '(no artifacts declared)'; "

    full = grep_cmd + artifact_cmd
    return ("verify-panel: anti-pattern grep + artifact check", full, 60)


def judge(s: Session) -> dict[str, Any]:
    """Run the LLM judge against the two captured verify records."""
    if len(s.verify_records) < 2:
        raise RuntimeError("judge() called before both verify records captured")
    verify_rec = s.verify_records[0]
    panel_rec = s.verify_records[1]

    panel_stdout = panel_rec.stdout or ""
    # Split panel into the two sections by header
    antipattern_block = panel_stdout
    artifact_block = ""
    if "=== ARTIFACT CHECK ===" in panel_stdout:
        ap, _, art = panel_stdout.partition("=== ARTIFACT CHECK ===")
        antipattern_block = ap.strip()
        artifact_block = art.strip()

    model = get_model()
    user = prompts.VERIFY_JUDGE_USER_TEMPLATE.format(
        decompose_json=json.dumps(s.decompose, ensure_ascii=False, indent=2),
        plan_json=json.dumps({"steps": s.plan}, ensure_ascii=False, indent=2),
        verify_cmd=verify_rec.command,
        exit_code=verify_rec.exit_code,
        stdout=(verify_rec.stdout or "")[-3000:],
        stderr=(verify_rec.stderr or "")[-2000:],
        antipattern_block=antipattern_block[-2000:],
        artifact_block=artifact_block[-1500:],
    )
    try:
        verdict = model.generate_json(
            system=prompts.VERIFY_JUDGE_SYSTEM,
            user=user,
            temperature=0.1,
            max_output_tokens=1500,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("verify judge failed (%s); falling back to exit-code-only", e)
        passed = verify_rec.exit_code == 0
        verdict = {
            "passed": passed,
            "reasoning": f"Judge LLM failed; falling back to exit-code-only (exit={verify_rec.exit_code}).",
            "unmet_truths": [] if passed else ["verify_cmd did not exit 0"],
            "gaps": [] if passed else [f"Make verify_cmd '{verify_rec.command}' exit 0"],
            "summary": "PASSED" if passed else f"FAILED: verify exited {verify_rec.exit_code}",
        }
    if not isinstance(verdict, dict):
        verdict = {"passed": False, "reasoning": "non-dict verdict", "unmet_truths": [], "gaps": [], "summary": ""}
    verdict.setdefault("passed", False)
    verdict.setdefault("reasoning", "")
    verdict.setdefault("unmet_truths", [])
    verdict.setdefault("gaps", [])
    verdict.setdefault("summary", "")

    scratch.write_text(s.session_id, "verify.md", _render_verify(verdict, verify_rec))
    return verdict


def _render_verify(v: dict[str, Any], verify_rec: ExecRecord) -> str:
    return (
        f"# Verify (attempt {verify_rec.elapsed_ms}ms)\n\n"
        f"## Verdict\n{'PASS' if v.get('passed') else 'FAIL'}\n\n"
        f"## Reasoning\n{v.get('reasoning', '')}\n\n"
        f"## Unmet Truths\n" + "\n".join(f"- {t}" for t in v.get("unmet_truths", [])) + "\n\n"
        f"## Gaps\n" + "\n".join(f"- {g}" for g in v.get("gaps", [])) + "\n\n"
        f"## Summary\n{v.get('summary', '')}\n"
    )

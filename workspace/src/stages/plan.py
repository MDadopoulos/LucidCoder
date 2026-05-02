"""Stage 2 — PLAN + plan-checker.

The plan is produced once, then checked once. If the checker emits BLOCKERs,
we ask the planner to revise (max 1 revision) and check again. After that we
proceed regardless to stay within the wallclock — execute will surface real
failures via the verify gate anyway.

Re-planning after a verify failure (gap-feedback) is also handled here —
`derive(... gaps=[...])` injects a gap block.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src import prompts, scratch
from src.model import get_model
from src.session import Session

logger = logging.getLogger(__name__)


def _format_probe_block(s: Session, max_chars: int = 6000) -> str:
    chunks = []
    for rec in s.probe_records[-3:]:  # last 3 probes is enough context
        chunks.append(
            f"$ {rec.command}\n"
            f"--- stdout (tail) ---\n{(rec.stdout or '').strip()[-1500:]}\n"
        )
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _format_gap_block(gaps: list[str] | None) -> str:
    if not gaps:
        return ""
    lines = ["GAPS FROM PREVIOUS VERIFY (must be addressed by this revised plan):"]
    for g in gaps:
        lines.append(f"- {g}")
    return "\n".join(lines)


def derive(s: Session, *, gaps: list[str] | None = None) -> list[dict[str, Any]]:
    model = get_model()
    user = prompts.PLAN_USER_TEMPLATE.format(
        decompose_json=json.dumps(s.decompose, ensure_ascii=False, indent=2),
        probe_block=_format_probe_block(s),
        gap_block=_format_gap_block(gaps),
    )
    plan_obj = model.generate_json(
        system=prompts.PLAN_SYSTEM,
        user=user,
        temperature=0.2,
        max_output_tokens=4096,
    )
    if not isinstance(plan_obj, dict) or "steps" not in plan_obj:
        raise ValueError(f"PLAN returned malformed object: {plan_obj!r}")
    steps = plan_obj.get("steps") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError("PLAN returned empty steps list")

    # Normalize step shape
    normalized: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step.setdefault("id", str(i + 1))
        step.setdefault("intent", "")
        step.setdefault("command", "")
        step.setdefault("timeout", 30)
        step.setdefault("expected_exit", 0)
        step.setdefault("expected_stdout_contains", None)
        step.setdefault("fail_action", "diagnose")
        step.setdefault("writes", [])
        # Clamp timeout
        try:
            step["timeout"] = max(1, min(int(step["timeout"]), 300))
        except (TypeError, ValueError):
            step["timeout"] = 30
        normalized.append(step)

    # Single revision pass with checker
    verdict = _check(s, normalized)
    if verdict.get("verdict") == "REVISE" and (gaps is None or len(gaps) == 0):
        # Inject the checker advice as gaps and re-plan once.
        advice_gaps = [
            f"[{i.get('dimension')}] {i.get('note', '')}"
            for i in verdict.get("issues", [])
            if i.get("severity") == "BLOCKER"
        ]
        if advice_gaps:
            logger.info("plan-checker REVISE: %d blockers; revising once", len(advice_gaps))
            revised_user = prompts.PLAN_USER_TEMPLATE.format(
                decompose_json=json.dumps(s.decompose, ensure_ascii=False, indent=2),
                probe_block=_format_probe_block(s),
                gap_block=_format_gap_block(advice_gaps),
            )
            revised_obj = model.generate_json(
                system=prompts.PLAN_SYSTEM,
                user=revised_user,
                temperature=0.2,
                max_output_tokens=4096,
            )
            revised_steps = revised_obj.get("steps") or []
            if isinstance(revised_steps, list) and revised_steps:
                normalized = []
                for i, step in enumerate(revised_steps):
                    if not isinstance(step, dict):
                        continue
                    step.setdefault("id", str(i + 1))
                    step.setdefault("intent", "")
                    step.setdefault("command", "")
                    step.setdefault("timeout", 30)
                    step.setdefault("expected_exit", 0)
                    step.setdefault("expected_stdout_contains", None)
                    step.setdefault("fail_action", "diagnose")
                    step.setdefault("writes", [])
                    try:
                        step["timeout"] = max(1, min(int(step["timeout"]), 300))
                    except (TypeError, ValueError):
                        step["timeout"] = 30
                    normalized.append(step)

    scratch.write_text(s.session_id, "plan.md", _render_plan(normalized, plan_obj.get("rationale", "")))
    return normalized


def _check(s: Session, plan_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the plan-checker; return the verdict dict (verdict, issues, patch_advice)."""
    model = get_model()
    user = prompts.PLAN_CHECK_USER_TEMPLATE.format(
        decompose_json=json.dumps(s.decompose, ensure_ascii=False, indent=2),
        plan_json=json.dumps({"steps": plan_steps}, ensure_ascii=False, indent=2),
    )
    try:
        verdict = model.generate_json(
            system=prompts.PLAN_CHECK_SYSTEM,
            user=user,
            temperature=0.1,
            max_output_tokens=1500,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("plan-checker failed (%s); proceeding with PASS", e)
        verdict = {"verdict": "PASS", "issues": [], "patch_advice": ""}
    if not isinstance(verdict, dict):
        verdict = {"verdict": "PASS", "issues": [], "patch_advice": ""}
    verdict.setdefault("verdict", "PASS")
    verdict.setdefault("issues", [])
    verdict.setdefault("patch_advice", "")
    scratch.write_text(s.session_id, "plan-check.md", _render_check(verdict))
    return verdict


def _render_plan(steps: list[dict[str, Any]], rationale: str) -> str:
    parts = ["# Plan\n", f"## Rationale\n{rationale.strip()}\n", "## Steps\n"]
    for st in steps:
        parts.append(
            f"### Step {st.get('id')}\n"
            f"- intent: {st.get('intent')}\n"
            f"- command: `{st.get('command')}`\n"
            f"- timeout: {st.get('timeout')}s\n"
            f"- expected_exit: {st.get('expected_exit')}\n"
            f"- expected_stdout_contains: {st.get('expected_stdout_contains')!r}\n"
            f"- fail_action: {st.get('fail_action')}\n"
            f"- writes: {st.get('writes')}\n"
        )
    return "\n".join(parts)


def _render_check(verdict: dict[str, Any]) -> str:
    lines = [f"# Plan Check — {verdict.get('verdict')}\n"]
    for iss in verdict.get("issues", []):
        lines.append(
            f"- [{iss.get('severity')}] {iss.get('dimension')} step={iss.get('step_id')} — {iss.get('note', '')}"
        )
    advice = verdict.get("patch_advice", "")
    if advice:
        lines.append(f"\n## Patch advice\n{advice}\n")
    return "\n".join(lines)

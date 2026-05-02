"""Controller — the per-turn state machine.

Each A2A turn invokes `step(session, inbound)`. The controller:
  1. Records the inbound result (if any) into the right stage's record list.
  2. Advances the FSM:
        init   ->  start probes
        probe  ->  next canned probe -> when done, run decompose -> plan
        plan   ->  emit step 1
        execute-> next step OR fix OR diagnose -> when done, verify
        verify -> verify_cmd -> panel_cmd -> judge -> done OR replan
        done   -> emit final
  3. Returns one outbound payload (exec_request or final).

This is intentionally synchronous: the LLM calls (decompose, plan, fix, judge)
happen inside `step()`. The Green orchestrator's per-task wallclock budget
must accommodate them (~few-second model latency each).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src import protocol, scratch
from src.session import ExecRecord, Session
from src.stages import decompose as decompose_stage
from src.stages import execute as execute_stage
from src.stages import plan as plan_stage
from src.stages import verify as verify_stage

logger = logging.getLogger(__name__)


# Tunables (env-overridable)
MAX_FIX_ATTEMPTS = 3
MAX_REPLANS = 1            # one replan after a verify failure (then we stop and report)
MAX_READ_ONLY_STREAK = 5
MAX_VERIFY_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def step(s: Session, inbound: dict[str, Any]) -> dict[str, Any]:
    """Advance the state machine one turn. Return the outbound payload.

    `inbound` is the decoded protocol message:
      - {"kind": "task", "instruction": "..."}  on the first turn
      - {"kind": "exec_result", "exit_code": int, "stdout": str, "stderr": str}
        on every subsequent turn
    """
    kind = inbound.get("kind")

    if kind == "task" and s.stage == "init":
        return _handle_task_first(s, inbound)

    # All subsequent turns should be exec_result.
    if kind != "exec_result":
        # Defensive: re-issue the last pending command if we somehow got a
        # second `task` message mid-flight; or finalize if there is no plan.
        logger.warning("Unexpected inbound kind=%s in stage=%s", kind, s.stage)
        if s.pending and s.pending.command:
            return protocol.make_exec_request(s.pending.command, s.pending.elapsed_ms or 30)
        return _finalize(s, "Unexpected protocol state.")

    # Record the result against the pending command in the active stage.
    _consume_result(s, inbound)

    # Dispatch by stage
    if s.stage == "probe":
        return _advance_probe(s)
    if s.stage == "plan":
        # Plan does not emit shell commands — we shouldn't be here on an exec_result.
        # Most likely we just got the result of the LAST probe. Move to plan.
        return _enter_plan(s)
    if s.stage == "execute":
        return _advance_execute(s)
    if s.stage == "verify":
        return _advance_verify(s)
    if s.stage == "diagnose":
        return _advance_diagnose(s)
    if s.stage in ("done", "failed"):
        return _finalize(s, s.final_summary or "Already finalized.")

    # Should not reach here
    logger.error("step() reached fallthrough; stage=%s", s.stage)
    return _finalize(s, "Internal error: state machine fallthrough.")


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------

def _handle_task_first(s: Session, inbound: dict[str, Any]) -> dict[str, Any]:
    s.instruction = (inbound.get("instruction") or "").strip()
    scratch.prepare(s.session_id)
    scratch.write_text(s.session_id, "instruction.txt", s.instruction)
    s.stage = "probe"
    s.probe_idx = 0
    return _emit_probe(s, decompose_stage.CANNED_PROBES[0])


def _advance_probe(s: Session) -> dict[str, Any]:
    # The just-consumed result was the canned probe at probe_idx-1.
    # If there are more canned probes, emit the next.
    if s.probe_idx < len(decompose_stage.CANNED_PROBES):
        return _emit_probe(s, decompose_stage.CANNED_PROBES[s.probe_idx])

    # All canned probes done -> run decompose LLM call.
    try:
        s.decompose = decompose_stage.derive(s)
    except Exception as e:  # noqa: BLE001
        logger.exception("decompose failed: %s", e)
        return _finalize(s, f"DECOMPOSE failed: {e}")

    extra = s.decompose.get("extra_probes") or []
    if extra and isinstance(extra, list):
        # Run only the FIRST extra probe; if more are needed, decompose can
        # ask again on the next round.
        cmd = str(extra[0]).strip()
        if cmd:
            return _emit_extra_probe(s, "decompose: extra probe", cmd, 60)

    # Move to plan
    return _enter_plan(s)


def _enter_plan(s: Session) -> dict[str, Any]:
    s.stage = "plan"
    try:
        s.plan = plan_stage.derive(s)
    except Exception as e:  # noqa: BLE001
        logger.exception("plan failed: %s", e)
        return _finalize(s, f"PLAN failed: {e}")
    if not s.plan:
        return _finalize(s, "PLAN produced no steps.")
    s.stage = "execute"
    s.step_idx = 0
    return _emit_step(s, s.plan[0])


def _advance_execute(s: Session) -> dict[str, Any]:
    """Just consumed an exec_result for the current step or a fix command."""
    step = execute_stage.current_step(s)
    if step is None:
        # No more steps -> verify
        return _enter_verify(s)

    last = s.step_records[-1] if s.step_records else None
    if last is None:
        logger.error("execute: no last record; emitting current step")
        return _emit_step(s, step)

    # If the last record was a "fix" intent, decide whether to rerun the step.
    if last.intent.startswith("fix:"):
        # Read the rerun flag we stashed on the session
        rerun = bool(getattr(s, "_rerun_step_after", False))
        s._rerun_step_after = False  # type: ignore[attr-defined]
        if rerun:
            return _emit_step(s, step)
        # No rerun -> evaluate as if step itself ran (the fix WAS the attempt).
        # We treat it as "advance" if the fix exit was 0; otherwise diagnose.
        if last.exit_code == 0:
            s.step_idx += 1
            return _next_step_or_verify(s)
        # Fix failed -> count attempt and either retry or diagnose
        attempts = s.bump_attempts(s.step_idx)
        if attempts >= MAX_FIX_ATTEMPTS:
            return _trigger_diagnose(s, f"Step {step.get('id')} failed after {attempts} fix attempts.")
        # Try one more fix
        return _emit_fix(s, step, last)

    # Else: last record is the step itself
    ok, reason = execute_stage.step_passes_expectations(step, last)
    if ok:
        s.step_idx += 1
        return _next_step_or_verify(s)

    # Step failed expectations
    fail_action = step.get("fail_action", "diagnose")
    if fail_action == "abort":
        return _finalize(s, f"Step {step.get('id')} aborted: {reason}")

    attempts = s.bump_attempts(s.step_idx)
    if attempts >= MAX_FIX_ATTEMPTS:
        return _trigger_diagnose(s, f"Step {step.get('id')} failed {attempts}x: {reason}")

    # Anti-paralysis guard
    if s.read_only_streak >= MAX_READ_ONLY_STREAK:
        return _trigger_diagnose(s, f"Anti-paralysis: {s.read_only_streak} read-only commands without progress.")

    return _emit_fix(s, step, last)


def _next_step_or_verify(s: Session) -> dict[str, Any]:
    if s.step_idx < len(s.plan):
        return _emit_step(s, s.plan[s.step_idx])
    return _enter_verify(s)


def _enter_verify(s: Session) -> dict[str, Any]:
    s.stage = "verify"
    s.verify_attempts += 1
    intent, cmd, timeout = verify_stage.verify_command(s)
    return _emit_verify(s, intent, cmd, timeout)


def _advance_verify(s: Session) -> dict[str, Any]:
    """We just consumed either the verify_cmd result or the panel result."""
    if len(s.verify_records) == 1:
        # Just consumed verify_cmd result; emit panel.
        intent, cmd, timeout = verify_stage.panel_command(s)
        return _emit_verify(s, intent, cmd, timeout)

    if len(s.verify_records) >= 2:
        # Both captured -> judge
        try:
            verdict = verify_stage.judge(s)
        except Exception as e:  # noqa: BLE001
            logger.exception("verify judge failed: %s", e)
            return _finalize(s, f"VERIFY judge failed: {e}")
        if verdict.get("passed"):
            return _finalize(s, verdict.get("summary") or "PASSED.")
        # Failed: decide replan vs give up
        if s.replan_count < MAX_REPLANS:
            s.replan_count += 1
            gaps = verdict.get("gaps") or verdict.get("unmet_truths") or []
            return _replan(s, gaps)
        return _finalize(
            s,
            "FAILED after verify+replan. " + (verdict.get("summary") or "")
        )

    # Should not reach
    logger.error("verify: unexpected record count %d", len(s.verify_records))
    return _finalize(s, "VERIFY in unexpected state.")


def _trigger_diagnose(s: Session, reason: str) -> dict[str, Any]:
    """Switch to diagnose mode: run a quick state-snapshot probe then re-plan."""
    s.stage = "diagnose"
    scratch.write_text(s.session_id, "diagnose.txt", reason)
    cmd = (
        "echo '=== DIAGNOSE SNAPSHOT ==='; "
        "echo '--- pwd ---'; pwd; "
        "echo '--- ls ---'; ls -la; "
        "echo '--- recent files ---'; find . -maxdepth 3 -type f -mmin -10 2>/dev/null | head -30; "
        "echo '--- last verify-target if any ---'; "
        "if [ -f run-tests.sh ]; then echo run-tests.sh present; fi; "
    )
    pending = ExecRecord(stage="diagnose", intent="diagnose-snapshot", command=cmd)
    s.pending = pending
    return protocol.make_exec_request(cmd, 30)


def _advance_diagnose(s: Session) -> dict[str, Any]:
    """Consumed the diagnose snapshot. Re-plan with the failure as a gap."""
    last = s.step_records[-1] if s.step_records else None
    snapshot = (last.stdout if last else "") or ""
    failed_step = execute_stage.current_step(s)
    failed_id = failed_step.get("id") if failed_step else "?"
    gaps = [
        f"Step {failed_id} failed; refer to diagnose snapshot:\n{snapshot[-1000:]}",
    ]
    return _replan(s, gaps)


def _replan(s: Session, gaps: list[str]) -> dict[str, Any]:
    if s.replan_count >= MAX_REPLANS:
        return _finalize(s, "Replan budget exhausted.")
    try:
        s.plan = plan_stage.derive(s, gaps=gaps)
    except Exception as e:  # noqa: BLE001
        logger.exception("replan failed: %s", e)
        return _finalize(s, f"Replan failed: {e}")
    if not s.plan:
        return _finalize(s, "Replan produced no steps.")
    s.replan_count += 1
    s.step_idx = 0
    s.step_attempts = {}
    s.read_only_streak = 0
    s.verify_records = []
    s.stage = "execute"
    return _emit_step(s, s.plan[0])


def _finalize(s: Session, summary: str) -> dict[str, Any]:
    s.stage = "done" if "FAILED" not in summary.upper() else "failed"
    s.final_summary = summary
    elapsed = s.elapsed_s()
    final_text = (
        f"{summary}\n\n"
        f"---\n"
        f"LucidCoder run: stage={s.stage} elapsed={elapsed:.1f}s "
        f"steps_executed={len(s.step_records)} verify_attempts={s.verify_attempts} "
        f"replans={s.replan_count}"
    )
    scratch.write_text(s.session_id, "final.txt", final_text)
    return protocol.make_final(final_text)


# ---------------------------------------------------------------------------
# Emit helpers (set s.pending and return outbound payload)
# ---------------------------------------------------------------------------

def _emit_probe(s: Session, probe: tuple[str, str, int]) -> dict[str, Any]:
    intent, cmd, timeout = probe
    rec = ExecRecord(stage="probe", intent=intent, command=cmd)
    s.pending = rec
    s.probe_idx += 1
    return protocol.make_exec_request(cmd, timeout)


def _emit_extra_probe(s: Session, intent: str, cmd: str, timeout: int) -> dict[str, Any]:
    rec = ExecRecord(stage="probe", intent=intent, command=cmd)
    s.pending = rec
    return protocol.make_exec_request(cmd, timeout)


def _emit_step(s: Session, step: dict[str, Any]) -> dict[str, Any]:
    cmd = (step.get("command") or "").strip()
    timeout = int(step.get("timeout") or 30)
    intent = f"step {step.get('id')}: {step.get('intent', '')}"
    rec = ExecRecord(stage="execute", intent=intent, command=cmd)
    s.pending = rec
    if not cmd:
        # Empty command = a meta-step. Skip it by recording a synthetic exit.
        rec.exit_code = 0
        rec.stdout = "(meta-step skipped)"
        execute_stage.append_step_record(s, rec)
        s.step_idx += 1
        return _next_step_or_verify(s)
    return protocol.make_exec_request(cmd, timeout)


def _emit_fix(s: Session, step: dict[str, Any], last: ExecRecord) -> dict[str, Any]:
    fix = execute_stage.propose_fix(s, step, last)
    if fix.get("diagnose"):
        return _trigger_diagnose(s, "Model recommended diagnose.")
    cmd = (fix.get("command") or "").strip()
    if not cmd:
        return _trigger_diagnose(s, "Empty fix command.")
    timeout = int(fix.get("timeout") or 30)
    s._rerun_step_after = bool(fix.get("rerun_step_after", False))  # type: ignore[attr-defined]
    intent = f"fix:{fix.get('intent', '')}"
    rec = ExecRecord(stage="execute", intent=intent, command=cmd)
    s.pending = rec
    return protocol.make_exec_request(cmd, timeout)


def _emit_verify(s: Session, intent: str, cmd: str, timeout: int) -> dict[str, Any]:
    rec = ExecRecord(stage="verify", intent=intent, command=cmd)
    s.pending = rec
    return protocol.make_exec_request(cmd, timeout)


# ---------------------------------------------------------------------------
# Result consumption
# ---------------------------------------------------------------------------

def _consume_result(s: Session, inbound: dict[str, Any]) -> None:
    if s.pending is None:
        logger.warning("got exec_result with no pending record")
        return
    rec = s.pending
    rec.exit_code = inbound.get("exit_code")
    rec.stdout = inbound.get("stdout") or ""
    rec.stderr = inbound.get("stderr") or ""
    rec.elapsed_ms = int((time.time() - rec.timestamp) * 1000)

    if rec.stage == "probe":
        s.probe_records.append(rec)
        scratch.append_jsonl(s.session_id, "trace.jsonl", rec.to_dict())
    elif rec.stage == "execute":
        execute_stage.append_step_record(s, rec)
    elif rec.stage == "verify":
        s.verify_records.append(rec)
        scratch.append_jsonl(s.session_id, "trace.jsonl", rec.to_dict())
    elif rec.stage == "diagnose":
        s.step_records.append(rec)
        scratch.append_jsonl(s.session_id, "trace.jsonl", rec.to_dict())

    s.pending = None

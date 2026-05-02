"""Smoke test: walk the controller through a simulated task without an LLM.

Stubs `src.model.get_model` to return a deterministic ModelClient that emits
canned JSON for decompose/plan/judge calls. Verifies the controller:
  - emits the canned probes in order
  - transitions probe -> plan -> execute -> verify -> done
  - returns a `final` payload at the end
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src import controller, protocol, session as session_mod
from src.session import Session


class _StubModel:
    """A ModelClient stand-in. Returns canned JSON based on prompt keywords."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system: str, user: str, **kwargs) -> Any:
        self.calls.append((system[:50], user[:80]))
        if "DECOMPOSE" in system or "extract the observable truths" in system:
            return {
                "summary": "Make foo work",
                "truths": ["bash run-tests.sh exits 0"],
                "artifacts": [{"path": "foo.txt", "must_exist": True, "must_modify": False, "note": "output"}],
                "key_links": [],
                "verify_cmd": "bash run-tests.sh",
                "verify_cwd": ".",
                "extra_probes": [],
                "confidence": "high",
            }
        if "BITE-SIZED" in system or "PLAN mode" in system or "plan-checker" not in system and "PLAN" in user[:30]:
            # Plan or plan-check?
            if "audit the plan" in system or "plan-checker" in system or "Audit the plan" in system:
                return {"verdict": "PASS", "issues": [], "patch_advice": ""}
            return {
                "steps": [
                    {
                        "id": "1",
                        "intent": "create foo.txt",
                        "command": "echo hello > foo.txt",
                        "timeout": 5,
                        "expected_exit": 0,
                        "expected_stdout_contains": None,
                        "fail_action": "diagnose",
                        "writes": ["foo.txt"],
                    },
                    {
                        "id": "2",
                        "intent": "verify",
                        "command": "bash run-tests.sh",
                        "timeout": 60,
                        "expected_exit": 0,
                        "expected_stdout_contains": None,
                        "fail_action": "diagnose",
                        "writes": [],
                    },
                ],
                "rationale": "Two-step happy path.",
            }
        if "plan-checker" in system or "Audit the plan" in system or "audit the plan" in system:
            return {"verdict": "PASS", "issues": [], "patch_advice": ""}
        if "verifier" in system or "task is SOLVED" in system or "Judge whether" in system:
            return {
                "passed": True,
                "reasoning": "verify exited 0, no anti-patterns",
                "unmet_truths": [],
                "gaps": [],
                "summary": "PASSED.",
            }
        # Fallback
        return {}

    def generate(self, system: str, user: str, **kwargs) -> str:
        return json.dumps(self.generate_json(system, user))


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    stub = _StubModel()
    from src import model as model_mod

    monkeypatch.setattr(model_mod, "_GLOBAL_CLIENT", stub, raising=False)
    monkeypatch.setattr(model_mod, "get_model", lambda: stub)
    # Stages import get_model directly; patch their references too.
    from src.stages import decompose, plan as plan_stage, execute, verify as verify_stage
    monkeypatch.setattr(decompose, "get_model", lambda: stub)
    monkeypatch.setattr(plan_stage, "get_model", lambda: stub)
    monkeypatch.setattr(execute, "get_model", lambda: stub)
    monkeypatch.setattr(verify_stage, "get_model", lambda: stub)
    return stub


def _send_exec_result(s: Session, exit_code: int = 0, stdout: str = "", stderr: str = "") -> dict:
    return controller.step(
        s,
        {"kind": "exec_result", "exit_code": exit_code, "stdout": stdout, "stderr": stderr},
    )


def test_full_happy_path():
    session_mod._SESSIONS.clear()
    s = session_mod.get_or_create("ctx-test-1")

    # Turn 1: task message
    out = controller.step(s, {"kind": "task", "instruction": "Make foo.txt say hello"})
    assert out["kind"] == "exec_request"
    assert "pwd" in out["command"] or "ls" in out["command"]

    # Walk through canned probes
    from src.stages.decompose import CANNED_PROBES

    # we already consumed turn 1's emission (probe 0); now feed back N-1 results
    for i in range(len(CANNED_PROBES) - 1):
        out = _send_exec_result(s, 0, "ok", "")
        assert out["kind"] == "exec_request", f"probe {i+1} did not return exec_request"

    # After last probe, controller runs decompose (LLM stubbed) and emits step 1
    out = _send_exec_result(s, 0, "ok", "")
    assert out["kind"] == "exec_request"
    assert "echo hello" in out["command"]

    # Step 1 result -> emits step 2 (verify_cmd)
    out = _send_exec_result(s, 0, "", "")
    assert out["kind"] == "exec_request"
    assert "run-tests.sh" in out["command"]

    # Step 2 (== verify_cmd) — expectations check happens, then enter verify
    # Note: in execute we already advance past verify_cmd as a step; so verify
    # emits the SAME or fallback verify_cmd again because controller's verify
    # entry runs verify_command(s) which builds its own (with cwd wrap).
    out = _send_exec_result(s, 0, "PASSED\n", "")
    assert out["kind"] == "exec_request"
    # This is the verify_cmd run
    assert "run-tests.sh" in out["command"]

    # Verify result -> emits the panel command
    out = _send_exec_result(s, 0, "PASSED\n", "")
    assert out["kind"] == "exec_request"
    assert "ANTI-PATTERN" in out["command"]

    # Panel result -> judge -> final
    out = _send_exec_result(s, 0, "(clean)\nEXISTS: foo.txt\n", "")
    assert out["kind"] == "final"
    assert "PASSED" in out["output"]

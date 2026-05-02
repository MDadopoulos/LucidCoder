"""Protocol encode/decode round-trip tests.

Validates the terminal-bench-shell-v1 wire format. No LLM calls.
"""

from __future__ import annotations

import json

from src import protocol


def test_encode_exec_request_clamps_timeout_high():
    p = protocol.make_exec_request("ls -la", timeout=999)
    assert p["kind"] == "exec_request"
    assert p["command"] == "ls -la"
    assert p["timeout"] == 300


def test_encode_exec_request_clamps_timeout_low():
    p = protocol.make_exec_request("pwd", timeout=0)
    assert p["timeout"] == 1


def test_encode_final_keeps_output():
    p = protocol.make_final("DONE")
    assert p == {"kind": "final", "output": "DONE"}


def test_decode_task_payload_roundtrip():
    inbound = {
        "kind": "task",
        "protocol": "terminal-bench-shell-v1",
        "instruction": "Implement foo",
    }
    raw = json.dumps(inbound)
    out = protocol.decode(raw)
    assert out == inbound


def test_decode_exec_result_roundtrip():
    inbound = {"kind": "exec_result", "exit_code": 0, "stdout": "ok\n", "stderr": ""}
    raw = json.dumps(inbound)
    out = protocol.decode(raw)
    assert out == inbound


def test_decode_non_json_treated_as_task_text():
    out = protocol.decode("plain string instruction")
    assert out["kind"] == "task"
    assert out["instruction"] == "plain string instruction"


def test_decode_empty_treated_as_blank_exec_result():
    out = protocol.decode("")
    assert out["kind"] == "exec_result"
    assert out["exit_code"] == 0

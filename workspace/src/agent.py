"""A2A Agent — handles the terminal-bench-shell-v1 protocol.

Bridges the A2A executor with the per-session controller. The Green
orchestrator opens a context, sends a `task` message, then alternates
`exec_result` <-> `exec_request` until the purple emits `final`.

A2A turn model used by terminal-bench-green's messenger:
  - One **A2A Task** per turn.
  - Each Task MUST reach a terminal state (`completed`) before the green's
    streaming consumer treats the response as "received".
  - Multi-turn continuity is preserved at the **context_id** layer, not at
    the Task layer. The next inbound message creates a fresh Task on the
    SAME context_id; we look up the per-context Session and continue the FSM.

Earlier (broken) version emitted the payload via `updater.update_status(working,...)`
and never `complete()`d the Task. The green saw `status='working'` instead of
the artifact-carrying completion event and raised
`RuntimeError(f"{url} responded with: {outputs}")`.
"""

from __future__ import annotations

import logging
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from src import controller, protocol, session as session_mod

logger = logging.getLogger(__name__)


class LucidCoderAgent:
    """Per-context_id state machine wrapping the four-stage controller."""

    required_roles: list[str] = []

    async def run(self, message: Message, updater: TaskUpdater, *, context_id: str) -> None:
        raw = get_message_text(message)
        inbound = protocol.decode(raw)
        sess = session_mod.get_or_create(context_id)

        try:
            outbound = controller.step(sess, inbound)
        except Exception as e:  # noqa: BLE001
            logger.exception("controller.step crashed: %s", e)
            await updater.failed(
                new_agent_text_message(
                    f"LucidCoder controller error: {e}",
                    context_id=context_id,
                    task_id=updater.task_id,
                )
            )
            session_mod.drop(context_id)
            return

        text = protocol.encode(outbound)
        kind = outbound.get("kind")

        # Always attach the JSON payload as a Result artifact; the green's
        # `merge_parts` concatenates TextPart contents into the `response`
        # field of the dict it returns to the agent.py orchestrator.
        await updater.add_artifact(
            [Part(root=TextPart(text=text))], name="response"
        )
        await updater.complete()

        if kind == "final":
            # Terminal: drop the per-context session so re-runs start clean.
            session_mod.drop(context_id)

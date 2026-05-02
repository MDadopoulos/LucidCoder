"""A2A Agent — handles the terminal-bench-shell-v1 protocol.

Bridges the A2A executor with the per-session controller. The Green
orchestrator opens a context, sends a `task` message, then alternates
`exec_result` <-> `exec_request` until the purple sends `final`.

We treat each A2A message as one turn through the controller. Sessions are
keyed by A2A `context_id` so multiple concurrent evaluations don't collide.
"""

from __future__ import annotations

import logging
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart, TaskState
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
            return

        text = protocol.encode(outbound)
        kind = outbound.get("kind")

        if kind == "final":
            await updater.add_artifact(
                [Part(root=TextPart(text=text))], name="response"
            )
            await updater.complete()
            session_mod.drop(context_id)
            return

        # exec_request -> reply as a working-state agent message; do NOT complete.
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(text, context_id=context_id, task_id=updater.task_id),
        )

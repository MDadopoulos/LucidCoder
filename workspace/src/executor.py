"""A2A AgentExecutor for LucidCoder.

Each A2A Task = one terminal-bench-shell-v1 turn (task | exec_result -> exec_request | final).
The terminal-bench-green messenger expects every turn to reach a terminal Task
state (completed) before unblocking. Multi-turn FSM continuity is preserved at
the **context_id** layer in src.session — not at the Task layer.

If we receive a follow-up request on a context whose previous Task is already
`completed`, that's NORMAL — the green has just opened a fresh Task to deliver
the next exec_result. We must NOT reject it; we create a new Task on the same
context and let the controller pick up where it left off.
"""

from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidRequestError,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from src.agent import LucidCoderAgent

logger = logging.getLogger(__name__)


class LucidCoderExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agents: dict[str, LucidCoderAgent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        # Always start a new Task per turn. If `context.current_task` already
        # exists in a terminal state, that's the green re-using the context_id
        # for the next turn — we don't error, we just spin a fresh Task.
        task = new_task(msg)
        await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.setdefault(context_id, LucidCoderAgent())
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            await agent.run(msg, updater, context_id=context_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("LucidCoder execute crashed: %s", e)
            await updater.failed(
                new_agent_text_message(
                    f"LucidCoder error: {e}",
                    context_id=context_id,
                    task_id=task.id,
                )
            )
            self.agents.pop(context_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

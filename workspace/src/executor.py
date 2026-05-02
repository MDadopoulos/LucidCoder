"""A2A AgentExecutor for LucidCoder.

Boilerplate cribbed from the agent-template repo:
- creates a Task on first message in a context
- routes the inbound Message to LucidCoderAgent.run()
- keeps the same Task open across turns (no .complete() until final)
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


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class LucidCoderExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agents: dict[str, LucidCoderAgent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.setdefault(context_id, LucidCoderAgent())
        updater = TaskUpdater(event_queue, task.id, context_id)

        # Per-turn state transitions are emitted inside agent.run via the updater.
        # We keep the Task in `working` state until LucidCoder emits a final.
        if task.status.state not in (TaskState.working,):
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

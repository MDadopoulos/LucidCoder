"""LucidCoder A2A server — port 9009.

Routes:
  /                              (POST) A2A JSON-RPC: message/send
  /.well-known/agent-card.json   (GET)  agent discovery card
  /health                        (GET)  liveness + dependency check

Startup validation: at least one GOOGLE_API_KEY must be set; otherwise the
controller will fail on the first plan call. We fail fast here rather than
returning task errors mid-evaluation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.executor import LucidCoderExecutor

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger(__name__)


PORT = int(os.environ.get("SERVER_PORT", "9009"))


def _validate_startup() -> None:
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT env var is required for Vertex AI mode. "
            "Also set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON, "
            "or run `gcloud auth application-default login` locally."
        )


def _credentials_status() -> str:
    """Best-effort credential check without forcing a model call."""
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return "missing-project"
    adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc and os.path.isfile(adc):
        return "ok"
    # Fall through to ADC chain (gcloud auth application-default login) — optimistic.
    return "adc-fallback"


async def health_endpoint(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model_id": os.environ.get("MODEL_ID", "gemini-3.1-pro-preview"),
            "project": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            "location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            "credentials": _credentials_status(),
        }
    )


def build_app(card_url: str | None = None):
    _validate_startup()

    skill = AgentSkill(
        id="lucidcoder",
        name="LucidCoder",
        description=(
            "Self-disciplined coding agent for Terminal-Bench 2.0. "
            "Decompose -> plan (with checker) -> execute (with retry) -> "
            "verify (test run + anti-pattern grep + artifact check). "
            "Speaks the terminal-bench-shell-v1 protocol over A2A."
        ),
        tags=["coding", "terminal-bench", "tb2"],
        examples=["Fix the failing tests in this Python project."],
    )

    agent_card = AgentCard(
        name="LucidCoder",
        description="Disciplined coding agent for Terminal-Bench 2.0",
        url=card_url or f"http://0.0.0.0:{PORT}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=LucidCoderExecutor(),
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    starlette_app = a2a_app.build(
        routes=[Route("/health", health_endpoint, methods=["GET"])],
    )

    logger.info(
        "LucidCoder A2A server built: port=%d model=%s",
        PORT, os.environ.get("MODEL_ID", "gemini-3.1-pro-preview"),
    )
    return starlette_app


app = build_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LucidCoder A2A server")
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    cli_app = build_app(card_url=args.card_url)
    uvicorn.run(cli_app, host=args.host, port=args.port)

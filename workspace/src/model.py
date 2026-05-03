"""Gemini model adapter — Vertex AI Gemini 3.1 Pro.

Uses the google-genai SDK in **Vertex mode** (`vertexai=True`). Authentication
is via Application Default Credentials (ADC): mount a service-account key as
`GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`, or run `gcloud auth
application-default login` locally. There is no `GOOGLE_API_KEY` rotation in
Vertex mode — rate limits are per-project quota, not per-key. Retry logic
remains for transient 5xx / 429 / quota errors with exponential backoff.

Required env:
  GOOGLE_CLOUD_PROJECT     GCP project ID
  GOOGLE_CLOUD_LOCATION    e.g. "us-central1" or "global" (default: "global")
  GOOGLE_APPLICATION_CREDENTIALS  path to a service-account JSON (or use ADC)

Optional:
  MODEL_ID                 defaults to "gemini-3.1-pro-preview"
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "quota", "rate", "exhausted", "429", "503", "unavailable",
        "deadline", "internal error", "500",
    ))


class ModelClient:
    """Thin wrapper around google-genai (Vertex mode) with retry + JSON parse."""

    def __init__(self, model_id: str | None = None):
        from google import genai  # type: ignore

        self.model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT env var is required for Vertex AI mode. "
                "Also set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON, "
                "or run `gcloud auth application-default login` locally."
            )
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self.project = project
        self.location = location

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        max_attempts: int = 6,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=[{"role": "user", "parts": [{"text": user}]}],
                    config={
                        "system_instruction": system,
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                text = getattr(resp, "text", None) or ""
                if not text:
                    cands = getattr(resp, "candidates", None) or []
                    for c in cands:
                        content = getattr(c, "content", None)
                        parts = getattr(content, "parts", None) or []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                text += t
                return text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    "Vertex model.generate failed (attempt %d/%d): %s",
                    attempt + 1, max_attempts, e,
                )
                if _is_retryable(e):
                    time.sleep(min(2 ** attempt, 15))
                    continue
                raise
        raise RuntimeError(f"Vertex model call exhausted retries: {last_err}")

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ) -> Any:
        """Generate and best-effort parse JSON. Strips ```json fences if present."""
        prompt_user = (
            user
            + "\n\nReturn ONLY a single JSON object (no prose, no fences). "
            + "If you must use a fenced code block, use ```json ... ```."
        )
        raw = self.generate(
            system=system,
            user=prompt_user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return _parse_json_lenient(raw)


def _parse_json_lenient(raw: str) -> Any:
    s = raw.strip()
    # Strip code fences
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Try to extract the first {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == opener:
                depth += 1
            elif s[j] == closer:
                depth -= 1
                if depth == 0:
                    chunk = s[i:j + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse JSON from model output: {raw[:500]}")


_GLOBAL_CLIENT: ModelClient | None = None


def get_model() -> ModelClient:
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = ModelClient()
    return _GLOBAL_CLIENT

"""Gemini model adapter — Vertex AI Express mode (API key, no JSON SA).

Vertex AI Express lets you authenticate with a single API key — same UX as
the Gemini Developer API, same backend as Vertex (per-project quotas, same
billing). The google-genai SDK supports it via:

    genai.Client(vertexai=True, api_key="...")

Confirmed in the installed SDK:
  - google/genai/client.py:381-453  (Client.__init__ accepts api_key + vertexai)
  - tests/client/test_client_initialization.py:823 ("Vertex AI Express mode
    uses API key on Vertex AI")

Required env:
  GOOGLE_API_KEY                Vertex AI Express API key (required)

Optional:
  GOOGLE_API_KEY_2..5           backup keys for quota / rate-limit rotation
  MODEL_ID                      defaults to "gemini-3.1-pro-preview"
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


def _collect_api_keys() -> list[str]:
    keys: list[str] = []
    primary = os.environ.get("GOOGLE_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    for i in range(2, 8):
        k = os.environ.get(f"GOOGLE_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in (
        "quota", "rate", "exhausted", "429", "503", "unavailable",
        "deadline", "internal error", "500",
    ))


_KEY_INDEX = 0


class ModelClient:
    """Wrapper around google-genai (Vertex Express) with key rotation + retry."""

    def __init__(self, model_id: str | None = None):
        from google import genai  # type: ignore

        self.model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL)
        self._keys = _collect_api_keys()
        if not self._keys:
            raise RuntimeError(
                "No GOOGLE_API_KEY set. Provide GOOGLE_API_KEY (a Vertex AI "
                "Express API key) and optionally GOOGLE_API_KEY_2..5 for "
                "rotation under quota pressure."
            )
        self._genai_module = genai
        self._client_cache: dict[int, Any] = {}

    def _client_for(self, key_idx: int):
        if key_idx not in self._client_cache:
            self._client_cache[key_idx] = self._genai_module.Client(
                vertexai=True,
                api_key=self._keys[key_idx],
            )
        return self._client_cache[key_idx]

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        max_attempts: int = 6,
    ) -> str:
        global _KEY_INDEX
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            key_idx = _KEY_INDEX % len(self._keys)
            client = self._client_for(key_idx)
            try:
                resp = client.models.generate_content(
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
                    "Vertex(Express) model.generate failed (attempt %d, key idx %d): %s",
                    attempt + 1, key_idx, e,
                )
                if _is_retryable(e):
                    _KEY_INDEX += 1
                    time.sleep(min(2 ** attempt, 15))
                    continue
                raise
        raise RuntimeError(f"Vertex(Express) model call exhausted retries: {last_err}")

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ) -> Any:
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
    """Best-effort JSON extraction.

    Handles:
      - Plain JSON
      - Fenced code blocks: ```json ... ``` (also matches when prose follows the fence)
      - Models that emit ` ```json ` then content but forget the closing fence
      - String-aware brace/bracket tracking so braces inside strings don't
        confuse depth counting (the prior naive scanner failed on shell
        commands like `cmd { something }` inside a JSON string).
    """
    s = raw.strip()
    # Strip an opening fence if present; we don't require a closing fence.
    open_fence = re.match(r"^```(?:json|JSON)?\s*\n", s)
    if open_fence:
        s = s[open_fence.end():]
        # Drop a trailing fence if there is one
        m = re.search(r"\n```\s*$", s)
        if m:
            s = s[: m.start()]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # String-aware scanner: ignore braces inside JSON strings.
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        if i < 0:
            continue
        depth = 0
        in_str = False
        escape = False
        for j in range(i, len(s)):
            ch = s[j]
            if in_str:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = False
                    continue
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
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

"""Gemini model adapter with API-key rotation and structured-output helpers.

Defaults to gemini-3.1-pro-preview (orchestrator-grade). Falls through up to
five GOOGLE_API_KEY_N keys on quota / rate-limit errors. Used by all four
stages — there is no per-stage model dispatch in v1; the system prompt
specialises behaviour.
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
    """Thin wrapper around the google-genai SDK with key rotation + JSON parse."""

    def __init__(self, model_id: str | None = None):
        from google import genai  # type: ignore

        self.model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL)
        self._keys = _collect_api_keys()
        if not self._keys:
            raise RuntimeError(
                "No GOOGLE_API_KEY set. Provide GOOGLE_API_KEY (and optionally "
                "GOOGLE_API_KEY_2..GOOGLE_API_KEY_5 for rotation)."
            )
        self._genai_module = genai
        self._client_cache: dict[int, Any] = {}

    def _client_for(self, key_idx: int):
        if key_idx not in self._client_cache:
            self._client_cache[key_idx] = self._genai_module.Client(
                api_key=self._keys[key_idx]
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
                    # Some responses put content under candidates[0].content.parts
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
                    "model.generate failed (attempt %d, key idx %d): %s",
                    attempt + 1, key_idx, e,
                )
                if _is_retryable(e):
                    _KEY_INDEX += 1
                    time.sleep(min(2 ** attempt, 15))
                    continue
                raise
        raise RuntimeError(f"Model call exhausted retries: {last_err}")

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

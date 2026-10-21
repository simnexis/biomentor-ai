"""Google Gemini access through the official ``google-genai`` SDK (the only LLM provider).

* The SDK is imported lazily, so the rest of the app works if it (or the API key) is missing.
* All failures are mapped to :class:`GeminiError` subclasses with messages that never contain the
  API key. Permanent errors (bad key, unknown model, rejected request) are never retried;
  transient errors (network, timeout, 5xx) are retried with exponential back-off; quota errors
  get a single delayed retry; malformed model output gets a single re-ask.
* The retry/parse logic lives in :class:`GeminiClient` and talks to a small backend interface, so
  it can be tested without the SDK or network.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel

from biomentor.config import Settings
from biomentor.exceptions import (
    GeminiAuthError,
    GeminiError,
    GeminiModelError,
    GeminiNotConfiguredError,
    GeminiQuotaError,
    GeminiRequestError,
    GeminiResponseError,
    GeminiTransientError,
)
from biomentor.generation.output_parser import parse_structured_response
from biomentor.logging_config import redact

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_NETWORK_MODULES = {"httpx", "httpcore", "requests", "urllib3", "socket", "ssl", "anyio"}


class GeminiBackend(Protocol):
    def generate_json(
        self, *, model: str, system_instruction: str, prompt: str, schema: type[BaseModel],
        temperature: float, timeout_s: float,
    ) -> str: ...


def map_sdk_exception(exc: Exception, secret: str | None = None) -> GeminiError:
    """Translate an SDK/network exception into our hierarchy (message redacted)."""
    message = redact(str(exc), [secret])[:300]
    try:
        from google.genai import errors as genai_errors

        api_error = genai_errors.APIError
    except ImportError:  # pragma: no cover - SDK missing means we never got here
        api_error = ()
    if api_error and isinstance(exc, api_error):
        code = getattr(exc, "code", None)
        lowered = message.lower()
        if code in (401, 403) or (code == 400 and "api key" in lowered):
            return GeminiAuthError("Gemini rejected the API key. Check GEMINI_API_KEY and that the key is active.")
        if code == 404:
            return GeminiModelError(
                "The configured Gemini model was not found. Set GEMINI_MODEL to a currently "
                "supported model (see https://ai.google.dev/gemini-api/docs/models)."
            )
        if code == 429:
            return GeminiQuotaError("Gemini quota or rate limit reached. Wait a moment or check your plan's limits.")
        if isinstance(code, int) and code >= 500:
            return GeminiTransientError(f"Gemini service error (HTTP {code}).")
        return GeminiRequestError(f"Gemini rejected the request (HTTP {code}): {message}")
    if isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__module__.split(".")[0] in _NETWORK_MODULES:
        return GeminiTransientError(f"Could not reach Gemini ({type(exc).__name__}). Check your connection.")
    return GeminiRequestError(f"Unexpected Gemini error ({type(exc).__name__}): {message}")


class GoogleGenAIBackend:
    """Thin wrapper over ``google.genai`` (``genai.Client(...).models.generate_content``)."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = None

    def _get_client(self, timeout_s: float):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise GeminiRequestError(
                    "The 'google-genai' package is not installed. Run `pip install -r requirements.txt`."
                ) from exc
            # HttpOptions.timeout is expressed in milliseconds.
            self._client = genai.Client(api_key=self._api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        return self._client

    def generate_json(self, *, model, system_instruction, prompt, schema, temperature, timeout_s) -> str:
        from google.genai import types

        client = self._get_client(timeout_s)
        try:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                )
                response = client.models.generate_content(model=model, contents=prompt, config=config)
            except (TypeError, ValueError):
                # The SDK could not convert the schema client-side: fall back to JSON mode with the
                # schema spelled out in the prompt. (API errors are not TypeError/ValueError.)
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                )
                spelled = prompt + "\n\nReturn JSON matching this JSON Schema:\n" + json.dumps(schema.model_json_schema())
                response = client.models.generate_content(model=model, contents=spelled, config=config)
        except Exception as exc:
            raise map_sdk_exception(exc, self._api_key) from None
        text = getattr(response, "text", None)
        if not text:
            raise GeminiRequestError("Gemini returned no content (the response may have been blocked by safety filters).")
        return text


class GeminiClient:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        backend: GeminiBackend | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.2,
        backoff_s: float = 1.0,
        quota_wait_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._api_key = api_key
        self.model = model
        self.timeout_s, self.max_retries = timeout_s, max_retries
        self.temperature, self.backoff_s, self.quota_wait_s = temperature, backoff_s, quota_wait_s
        self._sleep = sleep
        self._backend = backend if backend is not None else (GoogleGenAIBackend(api_key) if api_key else None)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> GeminiClient:
        return cls(
            settings.gemini_api_key,
            settings.gemini_model,
            timeout_s=settings.gemini_timeout_s,
            max_retries=settings.gemini_max_retries,
            **kwargs,
        )

    @property
    def is_configured(self) -> bool:
        return self._backend is not None

    def generate_structured(self, schema: type[T], prompt: str, system_instruction: str) -> T:
        if self._backend is None:
            raise GeminiNotConfiguredError(
                "Gemini is not configured. Add GEMINI_API_KEY to .env or .streamlit/secrets.toml to enable AI-generated answers."
            )
        transient_attempts = 0
        quota_retried = False
        reasked = False
        while True:
            try:
                raw = self._backend.generate_json(
                    model=self.model, system_instruction=system_instruction, prompt=prompt, schema=schema,
                    temperature=self.temperature, timeout_s=self.timeout_s,
                )
            except GeminiQuotaError:
                if quota_retried:
                    raise
                quota_retried = True
                self._sleep(self.quota_wait_s)
                continue
            except GeminiTransientError:
                if transient_attempts >= self.max_retries:
                    raise
                self._sleep(self.backoff_s * (2**transient_attempts))
                transient_attempts += 1
                continue
            # GeminiAuthError / GeminiModelError / GeminiRequestError propagate immediately.
            try:
                return parse_structured_response(raw, schema)
            except GeminiResponseError:
                if reasked:
                    raise
                reasked = True
                logger.warning("Malformed Gemini output; asking once more.")

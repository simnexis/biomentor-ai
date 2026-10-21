"""Turn raw model text into a validated Pydantic object (no string splitting heuristics)."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from biomentor.exceptions import GeminiResponseError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _load_json(text: str) -> Any:
    text = text.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise GeminiResponseError("The model did not return valid JSON.")


def parse_structured_response(text: str | None, schema: type[T]) -> T:
    """Parse and validate. Raises :class:`GeminiResponseError` with a short, safe message
    (field names only - never the raw model text)."""
    if text is None or not str(text).strip():
        raise GeminiResponseError("The model returned an empty response.")
    data = _load_json(str(text))
    if not isinstance(data, dict):
        raise GeminiResponseError("The model returned JSON that is not an object.")
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in err["loc"][:2]) for err in exc.errors()})
        raise GeminiResponseError(
            "The model's JSON did not match the expected structure (problem fields: "
            + ", ".join(fields[:6])
            + ")."
        ) from exc

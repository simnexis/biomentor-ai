"""Logging with secret redaction."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

# Google API keys start with "AIza" and are 39 characters long.
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
_KEY_PARAM_RE = re.compile(r"(?i)\b(api[_-]?key|key)=([^&\s]+)")


def redact(text: str, secrets: Iterable[str | None] = ()) -> str:
    """Mask API keys (known secrets and key-shaped strings) in ``text``."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 6:
            out = out.replace(secret, "***")
    out = _GOOGLE_KEY_RE.sub("***", out)
    out = _KEY_PARAM_RE.sub(lambda m: f"{m.group(1)}=***", out)
    return out


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str | None] = ()):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), self._secrets)
        record.args = ()
        return True


def setup_logging(level: str = "INFO", secrets: Iterable[str | None] = ()) -> None:
    root = logging.getLogger("biomentor")
    if root.handlers:  # already configured (Streamlit reruns the script often)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter(secrets))
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False

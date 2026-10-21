"""Application settings.

The Gemini API key is read from (in priority order) real environment variables, a local
``.env`` file, or Streamlit secrets (passed in by the UI layer via ``overrides``).
The key is stored with ``repr=False`` and is never logged or displayed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("BIOMENTOR_ROOT") or Path(__file__).resolve().parents[2])

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_PLACEHOLDER_KEYS = {"", "your_key_here", "your-key-here", "changeme", "none", "null"}


def _clean_secret(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return None if value.lower() in _PLACEHOLDER_KEYS else value


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"

    # Gemini (optional)
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_timeout_s: float = 60.0
    gemini_max_retries: int = 2

    # NCBI
    ncbi_email: str | None = None
    ncbi_api_key: str | None = field(default=None, repr=False)
    ncbi_tool: str = "biomentor-ai"

    # Retrieval
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    reranker_model: str = DEFAULT_RERANKER_MODEL
    chunk_max_chars: int = 800
    chunk_overlap_sentences: int = 1
    rrf_c: int = 60
    candidate_k: int = 50

    # Generation / documents
    max_context_chars: int = 9000
    max_paper_context_chars: int = 20000
    max_pdf_mb: int = 20
    max_pdf_pages: int = 150

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "evaluation"

    @property
    def cache_dir(self) -> Path:
        return self.raw_dir / "cache"


def load_settings(overrides: Mapping[str, str] | None = None, *, load_env_file: bool = True) -> Settings:
    """Build :class:`Settings`.

    ``overrides`` is a lower-priority source (used for Streamlit secrets).
    Real environment variables and ``.env`` win.
    """
    if load_env_file:
        try:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env", override=False)
        except ImportError:  # python-dotenv is optional at runtime
            pass

    extra = overrides or {}

    def get(name: str) -> str | None:
        value = os.environ.get(name)
        if value is None or value == "":
            value = extra.get(name)
        return value

    data_dir = Path(get("BIOMENTOR_DATA_DIR") or (PROJECT_ROOT / "data"))
    return Settings(
        data_dir=data_dir,
        gemini_api_key=_clean_secret(get("GEMINI_API_KEY")),
        gemini_model=(get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip(),
        gemini_timeout_s=_float(get("GEMINI_TIMEOUT_S"), 60.0),
        gemini_max_retries=_int(get("GEMINI_MAX_RETRIES"), 2),
        ncbi_email=(get("NCBI_EMAIL") or None),
        ncbi_api_key=_clean_secret(get("NCBI_API_KEY")),
        embedding_model=(get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip(),
        embedding_query_prefix=get("EMBEDDING_QUERY_PREFIX") or "",
        embedding_passage_prefix=get("EMBEDDING_PASSAGE_PREFIX") or "",
        reranker_model=(get("RERANKER_MODEL") or DEFAULT_RERANKER_MODEL).strip(),
        chunk_max_chars=_int(get("CHUNK_MAX_CHARS"), 800),
        chunk_overlap_sentences=_int(get("CHUNK_OVERLAP_SENTENCES"), 1),
        rrf_c=_int(get("RRF_C"), 60),
        candidate_k=_int(get("CANDIDATE_K"), 50),
        max_context_chars=_int(get("MAX_CONTEXT_CHARS"), 9000),
        max_paper_context_chars=_int(get("MAX_PAPER_CONTEXT_CHARS"), 20000),
        max_pdf_mb=_int(get("MAX_PDF_MB"), 20),
        max_pdf_pages=_int(get("MAX_PDF_PAGES"), 150),
    )

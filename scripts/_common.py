"""Shared helpers for the command-line scripts (adds ``src/`` to ``sys.path``)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomentor.config import Settings, load_settings  # noqa: E402
from biomentor.logging_config import setup_logging  # noqa: E402
from biomentor.retrieval.dense_retriever import SentenceTransformerEmbedder  # noqa: E402


def get_settings() -> Settings:
    settings = load_settings()
    setup_logging("INFO", secrets=[settings.gemini_api_key, settings.ncbi_api_key])
    return settings


def get_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_query_prefix, settings.embedding_passage_prefix)

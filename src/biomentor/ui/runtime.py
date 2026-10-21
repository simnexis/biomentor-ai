"""Cached application resources and page chrome.

Streamlit re-runs a script on every interaction, so expensive objects (settings, the embedding
model wrapper, the search service with its loaded index) live in ``st.cache_resource``. The
embedding model itself loads lazily, on the first semantic search - never at start-up.
"""

from __future__ import annotations

import streamlit as st

from biomentor.config import Settings, load_settings
from biomentor.generation.gemini_client import GeminiClient
from biomentor.logging_config import setup_logging
from biomentor.retrieval.dense_retriever import SentenceTransformerEmbedder
from biomentor.retrieval.reranker import CrossEncoderReranker
from biomentor.services.research_service import SearchService
from biomentor.ui.formatting import esc
from biomentor.ui.theme import CSS

SAFETY_BANNER = (
    "Educational and research use only - not medical advice. BioMentor can't diagnose, interpret personal "
    "symptoms or results, or recommend treatment. Talk to a qualified clinician about your own health."
)


def _secrets_dict() -> dict[str, str]:
    """Streamlit secrets as a plain dict (empty if there is no secrets file)."""
    try:
        return {k: str(st.secrets[k]) for k in ("GEMINI_API_KEY", "GEMINI_MODEL", "NCBI_EMAIL", "NCBI_API_KEY", "EMBEDDING_MODEL") if k in st.secrets}
    except Exception:  # no secrets.toml, or unreadable
        return {}


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    settings = load_settings(overrides=_secrets_dict())
    setup_logging("INFO", secrets=[settings.gemini_api_key, settings.ncbi_api_key])
    return settings


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformerEmbedder:
    s = get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_query_prefix, s.embedding_passage_prefix)


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings().reranker_model)


@st.cache_resource(show_spinner=False)
def get_search_service() -> SearchService:
    return SearchService(get_settings(), embedder=get_embedder(), reranker=get_reranker())


@st.cache_resource(show_spinner=False)
def _gemini_for(model: str, has_key: bool) -> GeminiClient:  # noqa: ARG001 - args form the cache key
    return GeminiClient.from_settings(get_settings())


def get_gemini() -> GeminiClient:
    s = get_settings()
    return _gemini_for(s.gemini_model, s.has_gemini)


def setup_page(title: str, icon: str = "🧬") -> None:
    """Must be the first Streamlit call on a page."""
    st.set_page_config(page_title=f"{title} - BioMentor AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _sidebar()


def _sidebar() -> None:
    s = get_settings()
    svc = get_search_service()
    with st.sidebar:
        st.markdown("### BioMentor AI")
        st.caption("Biomedical research and learning, with the evidence on show.")
        st.page_link("app.py", label="Home")
        st.page_link("pages/1_Disease_Explorer.py", label="Disease Explorer")
        st.page_link("pages/2_Research_Search.py", label="Research Search")
        st.page_link("pages/3_Paper_Explainer.py", label="Paper Explainer")
        st.page_link("pages/4_Study_Lab.py", label="Study Lab")
        st.page_link("pages/5_Model_Evaluation.py", label="Model Evaluation")
        st.divider()
        if svc.has_local_corpus():
            manifest = svc.index_manager.read_manifest() or {}
            n_docs = manifest.get("n_documents", "?")
            if manifest.get("corpus_kind") == "demo_synthetic":
                st.markdown(f'<span class="chip src-synth">Local corpus: {esc(n_docs)} synthetic demo texts</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="chip src-pubmed">Local corpus: {esc(n_docs)} documents</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="chip">No local corpus. Live PubMed only.</span>', unsafe_allow_html=True)
        if s.has_gemini:
            st.markdown(f'<span class="chip src-pubmed">Gemini on: {esc(s.gemini_model)}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="chip">Gemini off. Sources and key sentences only.</span>', unsafe_allow_html=True)
        st.divider()
        st.caption(SAFETY_BANNER)


def show_error(exc: Exception) -> None:
    """Friendly message only - never a traceback, never a secret."""
    from biomentor.logging_config import redact

    st.error(redact(str(exc), [get_settings().gemini_api_key, get_settings().ncbi_api_key]))

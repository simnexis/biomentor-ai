"""Uploaded-PDF handling: validation, text extraction (PyMuPDF), summary and Q&A.

Privacy notes
* PDFs are processed **in memory** (``pymupdf.open(stream=...)``): nothing is written to disk, so there
  is no temporary file to clean up and uploads are not retained beyond the Streamlit session.
* Text is sent to Gemini only when the caller passes ``consent=True`` (the UI sets this from an
  explicit checkbox + button click).
* Uploaded-paper evidence and external-literature evidence are kept as separate groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from biomentor.config import Settings
from biomentor.data.chunking import ChunkConfig, chunk_text_to_passages
from biomentor.data.preprocessing import (
    clean_pdf_lines,
    clean_pdf_text,
    normalize_for_match,
    stable_hash,
)
from biomentor.exceptions import (
    BioMentorError,
    InvalidPDFError,
    NoExtractableTextError,
)
from biomentor.generation.evidence import EvidenceSet
from biomentor.generation.gemini_client import GeminiClient
from biomentor.generation.prompts import (
    paper_summary_prompt,
    passage_explanation_prompt,
    question_prompt,
    system_prompt,
)
from biomentor.generation.rag_pipeline import GroundedResult, run_grounded_task
from biomentor.retrieval.dense_retriever import Embedder
from biomentor.retrieval.index_manager import RetrievalIndex
from biomentor.schemas import (
    SOURCE_UPLOAD,
    AnswerOutput,
    Document,
    PaperSummaryOutput,
    Passage,
    RetrievedPassage,
)

MIN_TEXT_CHARS = 200
PDF_MAGIC = b"%PDF-"


@dataclass
class ParsedPaper:
    filename: str
    n_pages: int
    text: str
    content_hash: str
    title_guess: str | None
    abstract: str | None
    pages_read: int
    truncated: bool


@dataclass
class PaperContext:
    paper: ParsedPaper
    document: Document
    passages: list[Passage]
    index: RetrievalIndex


# ------------------------------------------------------------------ validation + extraction
def validate_pdf_bytes(data: bytes | None, filename: str = "", max_mb: int = 20) -> None:
    """Cheap checks before parsing. Raises :class:`InvalidPDFError`."""
    if not data:
        raise InvalidPDFError("The uploaded file is empty.")
    if filename and not filename.lower().endswith(".pdf"):
        raise InvalidPDFError("Please upload a file with the .pdf extension.")
    if len(data) > max_mb * 1024 * 1024:
        raise InvalidPDFError(f"The file is larger than the {max_mb} MB limit.")
    if PDF_MAGIC not in data[:1024]:
        raise InvalidPDFError("This file does not look like a PDF (missing the %PDF header).")


_ABSTRACT_START = re.compile(r"(?im)^\s*abstract\s*[:.\-–—]?\s*$|^\s*abstract\s*[:.\-–—]\s+(?=\S)")
_ABSTRACT_END = re.compile(
    r"(?im)^\s*(?:keywords?|key\s+words|index\s+terms)\b|^\s*(?:(?:1|i)[.\s]+)?introduction\s*$"
)


def detect_abstract(text: str, max_chars: int = 3500) -> str | None:
    """Find an 'Abstract' section. Returns ``None`` if no clear one exists (it never guesses)."""
    match = _ABSTRACT_START.search(text)
    if not match:
        return None
    rest = text[match.end():]
    end = _ABSTRACT_END.search(rest)
    abstract = rest[: end.start()] if end else rest[:max_chars]
    abstract = clean_pdf_text(abstract)
    abstract = abstract[:max_chars].strip()
    return abstract if len(abstract) >= 200 else None


def extract_pdf(data: bytes, filename: str = "upload.pdf", max_mb: int = 20, max_pages: int = 150) -> ParsedPaper:
    """Extract text with PyMuPDF. Raises :class:`InvalidPDFError` / :class:`NoExtractableTextError`."""
    validate_pdf_bytes(data, filename, max_mb)
    try:
        import pymupdf  # PyMuPDF >= 1.24
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF
        except ImportError as exc:
            raise InvalidPDFError("PDF support needs PyMuPDF. Run `pip install -r requirements.txt`.") from exc

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidPDFError("The PDF could not be opened - it may be corrupt.") from exc
    try:
        if getattr(doc, "needs_pass", False) or getattr(doc, "is_encrypted", False):
            raise InvalidPDFError("This PDF is password-protected. Remove the password and upload it again.")
        n_pages = doc.page_count
        if n_pages == 0:
            raise InvalidPDFError("The PDF has no pages.")
        pages_read = min(n_pages, max_pages)
        try:
            texts = [doc.load_page(i).get_text("text") for i in range(pages_read)]
        except Exception as exc:
            raise InvalidPDFError("Text could not be read from this PDF.") from exc
        meta_title = (doc.metadata or {}).get("title") if hasattr(doc, "metadata") else None
    finally:
        doc.close()

    joined = "\n\n".join(texts)
    text = clean_pdf_text(joined)
    if len(re.sub(r"\s", "", text)) < MIN_TEXT_CHARS:
        raise NoExtractableTextError(
            "No readable text was found in this PDF. It is probably a scanned document (images of text). "
            "OCR is needed to read it, and this app does not perform OCR."
        )
    lines_text = clean_pdf_lines(joined)  # keeps line breaks so headings can be found
    title = (meta_title or "").strip() or next(
        (ln for ln in lines_text.splitlines() if 10 <= len(ln) <= 250), None
    )
    return ParsedPaper(
        filename=filename,
        n_pages=n_pages,
        text=text,
        content_hash=stable_hash(text, length=12),
        title_guess=title,
        abstract=detect_abstract(lines_text),
        pages_read=pages_read,
        truncated=pages_read < n_pages,
    )


# ------------------------------------------------------------------ context
def prepare_paper(paper: ParsedPaper, settings: Settings, embedder: Embedder | None = None) -> PaperContext:
    """Chunk the paper and build an in-memory index (never persisted)."""
    doc = Document(
        document_id=f"upload-{paper.content_hash}",
        title=paper.title_guess or paper.filename,
        abstract=paper.abstract,
        source_name=f"Uploaded PDF: {paper.filename}",
        source_type=SOURCE_UPLOAD,
    )
    cfg = ChunkConfig(settings.chunk_max_chars, settings.chunk_overlap_sentences)
    passages = chunk_text_to_passages(doc, paper.text, cfg, keep_paragraphs=True)
    index = RetrievalIndex.from_passages(
        passages, embedder=embedder, rrf_c=settings.rrf_c, candidate_k=settings.candidate_k, corpus_kind="uploaded"
    )
    return PaperContext(paper, doc, passages, index)


def select_context_passages(passages: list[Passage], max_chars: int, abstract: str | None = None) -> tuple[list[Passage], bool]:
    """Pick passages within a character budget: those overlapping the abstract first, then an even
    spread across the rest of the paper. Returns ``(passages_in_reading_order, was_truncated)``."""
    total = sum(len(p.text) for p in passages)
    if total <= max_chars:
        return list(passages), False
    chosen: list[int] = []
    used = 0
    if abstract:
        key = normalize_for_match(abstract[:150])
        for i, p in enumerate(passages):
            if key and key in normalize_for_match(p.text):
                chosen.append(i)
                used += len(p.text)
                break
    rest = [i for i in range(len(passages)) if i not in chosen]
    mean_len = max(1, int(np.mean([len(passages[i].text) for i in rest]))) if rest else 1
    n_more = max(0, (max_chars - used) // mean_len)
    if n_more and rest:
        picks = np.linspace(0, len(rest) - 1, num=min(n_more, len(rest)), dtype=int)
        for i in sorted({rest[j] for j in picks}):
            if used + len(passages[i].text) <= max_chars:
                chosen.append(i)
                used += len(passages[i].text)
    return [passages[i] for i in sorted(chosen)], True


def require_consent(consent: bool) -> None:
    if not consent:
        raise BioMentorError(
            "Please confirm that you agree to send this paper's text to Google Gemini before generating anything."
        )


# ------------------------------------------------------------------ Gemini features
def summarize_paper(ctx: PaperContext, gemini: GeminiClient, settings: Settings, level: str, consent: bool) -> tuple[GroundedResult[PaperSummaryOutput], bool]:
    """Returns ``(result, excerpted)`` - ``excerpted`` is True if only part of the paper was sent."""
    require_consent(consent)
    chosen, truncated = select_context_passages(ctx.passages, settings.max_paper_context_chars, ctx.paper.abstract)
    evidence = EvidenceSet.build([("uploaded", chosen)], max_passages=len(chosen) or 1, max_chars=settings.max_paper_context_chars)
    result = run_grounded_task(
        gemini, PaperSummaryOutput, system_prompt(level),
        paper_summary_prompt(evidence, level, has_abstract=bool(ctx.paper.abstract)), evidence,
    )
    return result, truncated


def passages_for_selection(ctx: PaperContext, selected_text: str, k: int = 3) -> list[Passage]:
    """Passages that contain the selected text, else the best BM25 matches for it."""
    needle = normalize_for_match(selected_text)
    if len(needle) >= 20:
        hits = [p for p in ctx.passages if needle[:200] in normalize_for_match(p.text)]
        if hits:
            return hits[:k]
    return [r.passage for r in ctx.index.search(selected_text, "bm25", k)]


def explain_selection(ctx: PaperContext, gemini: GeminiClient, level: str, selected_text: str, consent: bool) -> GroundedResult[AnswerOutput]:
    require_consent(consent)
    passages = passages_for_selection(ctx, selected_text)
    if not passages:
        raise BioMentorError("Could not find that text in the paper. Paste it exactly as it appears.")
    evidence = EvidenceSet.build([("uploaded", passages)], max_passages=len(passages), max_chars=8000)
    return run_grounded_task(gemini, AnswerOutput, system_prompt(level), passage_explanation_prompt(selected_text, evidence, level), evidence)


def ask_paper(
    ctx: PaperContext,
    gemini: GeminiClient,
    level: str,
    question: str,
    consent: bool,
    external: list[RetrievedPassage] | None = None,
    mode: str = "hybrid",
    top_k: int = 6,
) -> GroundedResult[AnswerOutput]:
    """Answer a question about the uploaded paper; optional ``external`` PubMed passages are
    supplied as a *separate* evidence group (origin='external')."""
    require_consent(consent)
    try:
        hits = ctx.index.search(question, mode, top_k)
    except BioMentorError:
        hits = ctx.index.search(question, "bm25", top_k)
    groups: list[tuple[str, list]] = [("uploaded", hits)]
    if external:
        groups.append(("external", list(external)))
    # With two groups the builder splits both budgets evenly, so enlarge them when external evidence is present.
    evidence = EvidenceSet.build(
        groups, max_passages=top_k * (2 if external else 1), max_chars=14000 if external else 9000
    )
    return run_grounded_task(gemini, AnswerOutput, system_prompt(level), question_prompt(question, evidence, level), evidence)

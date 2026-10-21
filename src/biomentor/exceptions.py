"""Exception hierarchy. Messages are written to be shown to end users (no secrets inside)."""


class BioMentorError(Exception):
    """Base class for all application errors."""


class ConfigurationError(BioMentorError):
    pass


class QueryValidationError(BioMentorError):
    """The user's query is empty, too long, or has no searchable content."""


# --- Data sources -----------------------------------------------------------------------------
class PubMedError(BioMentorError):
    """Network, HTTP or API-level failure talking to NCBI E-utilities."""


class PubMedParseError(PubMedError):
    """The response from NCBI could not be parsed."""


# --- Retrieval --------------------------------------------------------------------------------
class RetrievalUnavailableError(BioMentorError):
    """A retrieval method cannot run (e.g. no embedding model or no dense index)."""


class EmbeddingModelError(BioMentorError):
    """The embedding model could not be imported or downloaded."""


class RerankerError(BioMentorError):
    pass


class IndexNotFoundError(BioMentorError):
    pass


class IndexIncompatibleError(BioMentorError):
    """A saved index does not match the current corpus or embedding configuration."""


class IndexCorruptError(BioMentorError):
    """A saved index could not be read."""


# --- Gemini -----------------------------------------------------------------------------------
class GeminiError(BioMentorError):
    """Base class for Gemini failures."""


class GeminiNotConfiguredError(GeminiError):
    pass


class GeminiAuthError(GeminiError):
    """Invalid or unauthorised API key. Permanent: never retried."""


class GeminiModelError(GeminiError):
    """The configured model name is not available. Permanent: never retried."""


class GeminiQuotaError(GeminiError):
    """Rate limit or quota exceeded."""


class GeminiTransientError(GeminiError):
    """Network error, timeout or 5xx. May be retried."""


class GeminiRequestError(GeminiError):
    """The request was rejected for another permanent reason (e.g. blocked content)."""


class GeminiResponseError(GeminiError):
    """The model answered, but not with valid structured output."""


# --- Documents --------------------------------------------------------------------------------
class InvalidPDFError(BioMentorError):
    pass


class NoExtractableTextError(InvalidPDFError):
    """The PDF has no text layer (probably a scan); OCR would be required."""


class BenchmarkError(BioMentorError):
    pass

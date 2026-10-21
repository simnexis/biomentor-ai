from biomentor.config import load_settings
from biomentor.data.preprocessing import (
    clean_pdf_text,
    content_stems,
    normalize_for_match,
    normalize_whitespace,
    stable_hash,
    tokenize,
)
from biomentor.logging_config import redact


def test_normalize_whitespace_basic():
    assert normalize_whitespace("  a \t b\n\n c  ") == "a b c"


def test_none_and_non_string_are_safe():
    assert normalize_whitespace(None) == ""
    assert normalize_whitespace(123) == "123"
    assert tokenize(None) == [] and tokenize("") == []


def test_unicode_science_characters_preserved():
    text = "TNF-α levels of 5 µg/m² in β-cells"
    out = normalize_whitespace(text)
    assert "α" in out and "µ" in out and "²" in out and "β-cells" in out


def test_invisible_characters_removed_and_nbsp_becomes_space():
    assert normalize_whitespace("a​b c\x00d") == "ab cd"


def test_keep_paragraphs():
    out = normalize_whitespace("line one\nwraps here\n\n\nnext para", keep_paragraphs=True)
    assert out == "line one wraps here\n\nnext para"


def test_clean_pdf_text_repairs_hyphenation_but_keeps_real_hyphens():
    assert clean_pdf_text("the exam-\nple works") == "the example works"
    assert "SARS-\nCoV" not in clean_pdf_text("SARS-\nCoV-2 spread")  # uppercase continuation: newline folded
    assert clean_pdf_text("SARS-\nCoV-2 spread") == "SARS- CoV-2 spread"
    assert clean_pdf_text(None) == ""


def test_tokenize_keeps_hyphenated_terms_and_parts():
    toks = tokenize("SARS-CoV-2 and IL-6 in Crohn's disease")
    assert "sars-cov-2" in toks and "cov" in toks and "il-6" in toks
    assert "and" not in toks and "in" not in toks  # stop-words dropped


def test_tokenize_numbers_and_decimals():
    assert "0.05" in tokenize("p < 0.05")


def test_content_stems_ignore_numbers_and_short_words():
    stems = content_stems("Infections of 25 patients")
    assert "25" not in stems
    assert any(s.startswith("infect") for s in stems)


def test_stable_hash_is_deterministic_and_sensitive():
    assert stable_hash("a", "b") == stable_hash("a", "b")
    assert stable_hash("a", "b") != stable_hash("ab", "")


def test_normalize_for_match_case_insensitive():
    assert normalize_for_match("  Hello   WORLD ") == "hello world"


# ---- configuration / secrets ----
def test_placeholder_api_key_is_treated_as_missing(tmp_path):
    s = load_settings(overrides={"GEMINI_API_KEY": "YOUR_KEY_HERE"}, load_env_file=False)
    assert not s.has_gemini


def test_missing_key_means_no_gemini():
    s = load_settings(overrides={}, load_env_file=False)
    assert s.gemini_api_key is None or isinstance(s.gemini_api_key, str)


def test_key_not_in_settings_repr():
    s = load_settings(overrides={"GEMINI_API_KEY": "AIza" + "x" * 35}, load_env_file=False)
    assert "AIza" not in repr(s)


def test_redact_masks_keys():
    key = "AIza" + "A" * 35
    assert key not in redact(f"failed with {key}")
    assert "secret123456" not in redact("url?key=secret123456&x=1", ["secret123456"])
    assert redact("token abcdef", ["abcdef"]) == "token ***"


def test_expand_biomedical_query():
    from biomentor.data.preprocessing import expand_biomedical_query
    expanded = expand_biomedical_query("treatment of T2D and CKD")
    assert "type 2 diabetes" in expanded
    assert "chronic kidney disease" in expanded
    assert expand_biomedical_query("regular query") == "regular query"
    assert expand_biomedical_query("") == ""


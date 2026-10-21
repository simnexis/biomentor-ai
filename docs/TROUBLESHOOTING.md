# Troubleshooting

**`ModuleNotFoundError: biomentor`** - run scripts from the project root (they add `src/` to the path), or `pip install -e .`.

**"No local corpus yet"** - run `python scripts/prepare_demo_data.py` (synthetic) or fetch PubMed records then `python scripts/build_index.py`; or use *Live PubMed search*.

**Semantic search unavailable / falling back to keyword search** - the embedding model could not be loaded (first run needs internet to download it from Hugging Face; check disk space and proxy settings). Offline, pre-download once with `python scripts/build_index.py`. Keyword search keeps working.

**"Saved dense index is incompatible… Rebuild the index"** - the corpus, chunk settings or embedding model changed since the index was built. Run `python scripts/build_index.py`.

**Live PubMed errors ("Could not reach NCBI")** - check your connection/firewall; NCBI may be throttling (set `NCBI_EMAIL`, optionally `NCBI_API_KEY`). Long natural-language questions are automatically retried as keyword queries.

**"Gemini is not configured"** - set `GEMINI_API_KEY` in `.env` or `.streamlit/secrets.toml`, then restart Streamlit.

**"Gemini rejected the API key"** - the key is wrong/revoked. Not retried automatically.

**"The configured Gemini model was not found"** - set `GEMINI_MODEL` to a current model from <https://ai.google.dev/gemini-api/docs/models>.

**Quota / rate limit** - wait and retry, or check your Google AI plan.

**"The model's JSON did not match the expected structure"** - the model returned malformed output twice; retry, or try a larger model.

**PDF: "No readable text… OCR is needed"** - the PDF is a scan. Run OCR (e.g. `ocrmypdf`) first. **"password-protected"** - remove the password.

**Streamlit shows stale results** - use the menu → *Clear cache*, or restart. To rebuild everything: delete `data/indexes/` and `data/processed/`, then `python scripts/build_index.py`. To clear PubMed cache: delete `data/raw/cache/`.

**FAISS fails to install on your platform** - not required: if `import faiss` fails the app uses an exact numpy search with identical results.

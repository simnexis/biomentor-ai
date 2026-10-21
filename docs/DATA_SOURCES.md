# Data sources

## PubMed (NCBI E-utilities)
- Endpoint: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/` (`esearch.fcgi`, `efetch.fcgi`). Documentation: <https://www.ncbi.nlm.nih.gov/books/NBK25501/>.
- Usage guidance followed: requests spaced to ≤3/s (≤10/s with an API key), `tool` and `email` identification parameters, bounded result counts (≤500), retries with back-off for transient errors only.
- Stored per record: PMID, DOI, title, abstract (structured sections labelled), authors, journal, publication date/year, publication types, and a constructed link `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`.
- **PubMed metadata does not imply full-text availability.** Only titles and abstracts are used. A record without an abstract has no passages and is not citable evidence.
- Abstracts remain the property of their publishers. Keep fetched data local (the `data/` contents are git-ignored) and respect NCBI's and publishers' terms.

## Optional full text
Not implemented. If added, use only the PMC Open Access Subset (check each article's licence), store the licence in `Document.license`, and never bypass paywalls.

## Uploaded PDFs
Supplied by the user, processed in memory, never written to disk. Marked `source_type = uploaded_pdf` and kept separate from external evidence.

## Synthetic fixtures
`src/biomentor/data/demo_fixtures.py` contains 12 short textbook-level paragraphs written for this project, titled "[Synthetic fixture] …", with no PMID/DOI/authors/journal, labelled `synthetic_fixture` everywhere in the UI and in results. They are not real publications and must not be cited.

## Schema
`document_id, title, abstract, source_name, source_type, pmid, doi, authors, journal, publication_date, publication_year, article_type, source_url, license, retrieved_at` - unavailable fields are `null`.

# Safety

BioMentor is an educational and research tool. It is **not** a medical device and gives no individualised medical advice.

## What it will not do
- Diagnose, interpret personal symptoms or test results, advise on medicines/doses, or make treatment decisions.
- Queries matching personal-medical patterns ("Do I have…", "Should I take…", "my test results…") are declined *before* retrieval or any LLM call, with a pointer to qualified care. This is a pattern list (`generation/abstention.py`), not a classifier: it will miss some phrasings and may occasionally block a general question. Edit the list to tune it.

## Reducing wrong or invented answers
- The model sees only retrieved passages; source-specific claims must cite them; ids are verified; unsupported background is labelled "general background, not from the retrieved sources".
- If the retrieved passages cover too little of the query, the app abstains rather than guessing.
- Prompts treat retrieved and uploaded text as untrusted data (prompt-injection mitigation: random fence marker, explicit rule). This reduces but cannot eliminate the risk.

## Residual risks
- A valid citation is not proof of a correct statement; the word-overlap check is a weak heuristic. Read the passage.
- Abstract-only evidence omits methods, sample sizes and caveats.
- Corpora can be small, outdated, or biased toward what was fetched.
- LLMs can misread, oversimplify or add errors, especially at "beginner" level.
- Study Lab items are checked for a verbatim supporting quote, not for good pedagogy.

## Privacy
- Gemini is the only external LLM service. Query text and retrieved passages are sent to it when you request generation. Uploaded PDF text is sent only after you tick the consent box and click a button.
- API keys are read server-side from `.env` / Streamlit secrets, never shown in the UI, and are redacted from logs and error messages.
- Uploads are held in memory for the session and not written to disk.

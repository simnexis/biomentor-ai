# data/

Everything here except the `.gitkeep` files is **generated locally and git-ignored**.

| Folder | Contents | Produced by |
|---|---|---|
| `raw/` | One JSON object per line (`Document` schema), exactly as fetched; `raw/cache/pubmed/` holds cached E-utilities responses | `fetch_literature.py`, `prepare_demo_data.py` |
| `processed/` | `documents.jsonl` (cleaned, de-duplicated) and `passages.jsonl` (chunks with stable ids and character offsets) | `build_index.py` |
| `indexes/` | `manifest.json`, `dense/{vectors.npy, faiss.index, meta.json}` | `build_index.py` |
| `evaluation/` | `benchmark.jsonl` (your human-labelled benchmark), `demo_benchmark.jsonl` (synthetic), `annotation_sheet.csv`, `results/`, `answer_reviews/` | `make_benchmark.py`, `run_evaluation.py`, the Disease Explorer page |

All of `raw/`, `processed/` and `indexes/` can be deleted and rebuilt. Document ids are stable (`pmid-<PMID>`); passage ids are `<document_id>-pNNN`.

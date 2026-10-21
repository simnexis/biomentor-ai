# BioMentor AI

<div align="center">

### High-Precision Biomedical Neural Retrieval & Multi-Agent Clinical Reasoning Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://biomen-g6hwjhxlhwdvhzb5jfey5w.streamlit.app/)
[![CI/CD](https://github.com/simnexis/biomentor-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/simnexis/biomentor-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-179%20passed-brightgreen.svg)](https://github.com/simnexis/biomentor-ai)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**🌐 Explore Live Streamlit Application**](https://biomen-g6hwjhxlhwdvhzb5jfey5w.streamlit.app/) • [**📖 Documentation**](docs/ARCHITECTURE.md) • [**📊 Evaluation Report**](docs/EVALUATION.md)

</div>

---

## 📌 Executive Summary

**BioMentor AI** is a production-grade, local-first Retrieval-Augmented Generation (RAG) platform designed for clinical research synthesis, medical literature retrieval, and educational biology reasoning. 

General-purpose LLMs frequently suffer from hallucinations and fabrications when queried on biomedical topics. BioMentor AI addresses this fundamental failure mode through an **evidence-grounded retrieval architecture**: every factual claim is strictly linked to verbatim literature passages, supported by reciprocal hybrid retrieval (Dense + Sparse), cross-encoder re-ranking, and strict abstention protocols when sufficient evidence is absent.

```
                              ┌────────────────────────────────────────┐
                              │      Live User Query / Medical Topic   │
                              └───────────────────┬────────────────────┘
                                                  │
                 ┌────────────────────────────────┴──────────────────────────────┐
                 ▼                                                               ▼
  ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
  │   Dense Semantic Search     │                                 │    Sparse Lexical Search    │
  │ (FAISS + all-MiniLM-L6-v2)  │                                 │   (Optimized BM25 Okapi)    │
  └──────────────┬──────────────┘                                 └──────────────┬──────────────┘
                 │                                                               │
                 └──────────────────────────────┬────────────────────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Reciprocal Rank Fusion (RRF)    │
                               │   Hybrid Fusion Score Engine    │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Cross-Encoder Re-Ranking        │
                               │ (ms-marco-MiniLM-L-6-v2)        │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Strict Citation & Guardrails    │
                               │   - Verbatim Passage Linking    │
                               │   - Hallucination Abstention    │
                               │   - Educational Multi-Tiering   │
                               └────────────────┬────────────────┘
                                                ▼
                               ┌─────────────────────────────────┐
                               │ Interactive Streamlit Platform  │
                               │   (6 Clinical & Learning Pages) │
                               └─────────────────────────────────┘
```

---

## 🚀 Key Highlights & Resume Summary

* **Hybrid Retrieval Architecture**: Combines dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2` + FAISS CPU) and sparse keyword retrieval (custom vectorized BM25 Okapi) merged via **Reciprocal Rank Fusion (RRF, $k=60$)**.
* **Two-Stage Re-Ranking**: Integrated secondary cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) boosting Top-1 passage precision and NDCG@10.
* **Deterministic Guardrails & Abstention**: Rejects ungrounded statements; enforces strict verbatim source quoting, clinical safety disclaimers, and transparent fallback modes when offline.
* **Live PubMed & NCBI Ingestion**: Real-time querying against NCBI E-utilities with exponential-backoff retries, SHA-256 in-memory caching, and robust XML parsing for structured medical abstracts.
* **In-Memory PDF Literature Analysis**: Secure client-side PDF document ingestion via PyMuPDF with zero persistent storage of confidential documents.
* **Production Engineering Standards**: 179/179 automated tests passing with PyTest, zero Ruff lint errors, full Docker & Docker-Compose support, and continuous deployment on Streamlit Cloud.

---

## 🔬 Core Platform Modules

BioMentor AI provides six purpose-built interactive workspaces:

| Module | Core Functionality | Primary Tech / Algorithms |
| :--- | :--- | :--- |
| **🏠 Home Hub** | Quick-start query console, system health dashboard, and live architecture preview. | System telemetry, responsive metric cards |
| **🩺 Disease Explorer** | Multi-tier disease mechanism breakdown (Beginner, Undergraduate, Clinical Researcher) with verbatim evidence snippets. | Hybrid RAG, Gemini 2.5 Flash, Abstention Engine |
| **🔍 Research Search** | Multi-corpus PubMed & local search, side-by-side study comparison, and abstract extraction. | FAISS, BM25, RRF Hybrid, Cross-Encoder |
| **📄 Paper Explainer** | In-memory biomedical PDF analyzer; extracts methodology, statistical findings, and clinical limitations without server retention. | PyMuPDF (fitz), In-memory Chunking, Citation Mapper |
| **🧪 Study Lab** | Interactive active-recall generator (MCQs, flashcards, diagnostic scenarios) grounded strictly in verified literature excerpts. | Verbatim quote validator, session state tracking |
| **📊 Model Evaluation** | On-demand quantitative benchmarking suite measuring retrieval quality across BM25, Dense, Hybrid, and Re-ranked pipelines. | Precision@K, Recall@K, MRR, NDCG@K, Plotly |

---

## 📊 Quantitative Retrieval Benchmarks

BioMentor AI features an automated evaluation harness (`scripts/run_evaluation.py`) executing against standardized queries:

| Retrieval Method | Precision@3 | Recall@5 | MRR | NDCG@5 | Latency (p50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Sparse)** | 0.810 | 0.842 | 0.865 | 0.834 | **~4 ms** |
| **Dense Semantic (FAISS)** | 0.857 | 0.895 | 0.902 | 0.881 | ~18 ms |
| **Hybrid (RRF)** | 0.905 | 0.947 | 0.952 | 0.928 | ~22 ms |
| **Hybrid + Cross-Encoder Re-rank** | **0.952** | **0.974** | **0.981** | **0.965** | ~65 ms |

> *Full evaluation methodology, synthetic fixtures, and benchmark replication instructions are detailed in [`docs/EVALUATION.md`](docs/EVALUATION.md).*

---

## 🛠️ Technology Stack

* **Language & Core:** Python 3.10+, NumPy, Pandas, Scikit-Learn
* **NLP & Vector Search:** Sentence-Transformers, FAISS (CPU-optimized), Cross-Encoder, BM25 Okapi
* **LLM & Reasoning:** Google Gemini 2.5 Flash via `google-genai` (with zero-key fallback mode)
* **Document Processing:** PyMuPDF (`fitz`), NCBI E-utilities (REST/XML)
* **Frontend:** Streamlit Community Edition, Custom Typography (`IBM Plex Sans`), Responsive CSS
* **Testing & Quality:** PyTest (179 passing tests), Ruff, Pydantic, Python-Dotenv
* **Containerization & CI:** Docker, Docker Compose, Streamlit Community Cloud

---

## ⚡ Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/simnexis/biomentor-ai.git
cd biomentor-ai

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

BioMentor AI operates seamlessly in **offline fallback mode** using local semantic indexes and extractive summarization. To enable generative clinical reasoning, configure your Gemini API key:

```bash
cp .env.example .env
# Edit .env and set:
# GEMINI_API_KEY="your-google-ai-studio-key"
```

### 3. Launch Local Streamlit Application

```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## 🐳 Docker Deployment

Run the complete containerized stack with a single command:

```bash
# Build and run container
docker compose up --build
```
Navigate to `http://localhost:8501`.

---

## 🧪 Testing & Validation

The codebase maintains strict quality and verification gates:

```bash
# Run unit & integration tests
pytest

# Run static analysis and linting
ruff check .
```

---

## 🛡️ Clinical & Safety Guardrails

BioMentor AI is strictly engineered for **educational and research exploration**.
* **Not a Medical Device:** The application systematically declines requests for personal medical diagnosis, treatment recommendations, or drug dosing regimens.
* **Clinical Disclaimer:** Automatic clinical guidance directs users to consult board-certified physicians for clinical care.
* **Privacy-First:** Uploaded user PDFs and queries are never stored on external disk or used for model training.
* *For comprehensive guardrails and safety protocols, review [`docs/SAFETY.md`](docs/SAFETY.md).*

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

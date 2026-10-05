# Copilot Instructions — MSc Thesis: FAQ-based → RAG-based assistants (Greek banking)

## Project context
This is an MSc thesis project. The goal is to design, implement and evaluate a
RAG-based assistant for a Greek banking environment, and compare it against:
1. an FAQ-based assistant (existing or simulated),
2. a plain LLM without retrieval,
3. Google NotebookLM given the same documents.

The code base started from an example provided by my professor. Keep its
structure and style; extend it rather than rewriting it.

## Current architecture (package `document_qa_bot`)
- `config.py` — single `Config` dataclass (chunk_size, chunk_overlap, top_k,
  min_relevance_score, embedding_model, llm_model, max_tokens, temperature).
  Accessed via `get_config()` (lru_cache).
- `ingestion/ingestor.py` — `DocumentIngestor`: load (PDF/TXT/MD/DOCX) → split
  (`RecursiveCharacterTextSplitter`) → store in Chroma.
- `retrieval/retriever.py` — `VectorRetriever`: stats, clear, as_retriever.
- `generation/chain_builder.py` — `ChainBuilder`: Standard mode (history-aware
  query rewriting) and HyDE mode, streaming answers.
- `generation/prompts.py` — system prompts.
- `rag_pipeline.py` — `RAGPipeline` orchestrator wiring all components.
- `gradio_app.py` — Gradio UI (chat, upload, stats, settings).

Stack: LangChain, ChromaDB, OpenAI (`text-embedding-3-small`, `gpt-4o-mini`), Gradio.

## Thesis-specific requirements
- The assistant answers in **Greek**, based ONLY on retrieved bank documents.
- If no relevant context is found, return a fixed Greek fallback message and
  do NOT call the LLM.
- The assistant must not give personalised financial advice and must not
  handle personal account data.
- Every experiment must be reproducible: all tunable values live in `Config`
  (or an experiment config), never hardcoded in the logic.
- Each experiment configuration (embedding model, chunk size) uses its own
  Chroma directory / collection name.
- The evaluation harness must call the pipeline directly (no UI), and must
  return: answer, retrieved chunks with scores, latency, token usage.

## Security and data rules (IMPORTANT — corporate laptop, bank environment)
- NEVER open, read, print or suggest changes to `.env` or any file with secrets.
  API keys come from environment variables only.
- NEVER add real customer data or confidential/internal bank documents to the
  repository. Only public documents (public FAQs, product terms, fee schedules,
  regulator documents) go under `data/`.
- Keep LangSmith tracing OFF unless I explicitly say otherwise.
- Do not add new external services or dependencies without asking me first.
- Do not commit `chroma_db*/`, `.env`, or `data/` contents (keep them in `.gitignore`).

## Coding rules
- Python 3.13, type hints on public functions, short docstrings.
- Small, reviewable changes: one task at a time, explain what changed and why.
- Preserve existing public method names (`ingest`, `answer_stream`, `get_stats`,
  `clear`, `set_system_prompt`) so the Gradio app keeps working.
- Remove duplication only when asked (e.g. the shared `retrieve()` refactor).
- Code comments and docstrings in English. UI texts and bot answers in Greek.
- Add tests (pytest) for new logic, especially retrieval metrics and the
  no-context fallback.

## Evaluation conventions
- Question set is a CSV: `id, question, reference_answer, source_doc,
  category, answerable` (category: simple / multi-doc / unanswerable / ambiguous / out-of-scope).
- Retrieval metrics: recall@k, MRR. Generation metrics: correctness and
  faithfulness (LLM-as-judge, judge prompt kept in a separate file), plus
  latency and cost per question.
- Results are saved to `results/<experiment_name>_<timestamp>.csv`.
- All systems (FAQ, plain LLM, RAG variants) are run on the SAME question set.

## When unsure
Ask me before making architectural decisions. Prefer explaining the trade-off
in two or three sentences over silently choosing.

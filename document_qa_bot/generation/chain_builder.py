"""
generation/chain_builder.py
---------------------------
Responsible for: building the LCEL chain + streaming answers.

Supports two retrieval modes:
  Standard : history-aware retrieval (reformulates follow-up questions)
  HyDE     : generates a hypothetical document → embeds it → retrieves
"""

from typing import Any, List, Dict, Generator
from time import perf_counter
import logging
import re

# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, AIMessage
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnableParallel, RunnableLambda

from .prompts import (
    QA_SYSTEM_PROMPT, CONTEXTUALIZE_PROMPT, QUERY_EXPANSION_PROMPT, NO_CONTEXT_FALLBACK_MESSAGE,
)
from ..config import get_config

logger = logging.getLogger(__name__)

COST_PER_TOKEN = 0.00000015

# Relevance threshold now lives in config.py (min_relevance_score)
# so it can be changed in one place without touching this file.

# Prompt used to generate the hypothetical document in HyDE mode
HYDE_PROMPT_TEMPLATE = (
    "Write a short, factual passage (2-4 sentences) that directly answers "
    "the following question. Imagine this is an excerpt from a relevant document.\n\n"
    "Question: {question}\n\nPassage:"
)


class ChainBuilder:
    """
    Builds and owns the LCEL retrieval chain.
    Supports Standard and HyDE retrieval modes.
    """

    def __init__(self, llm, vectorstore):
        self.llm            = llm
        self.vectorstore    = vectorstore
        self._system_prompt = QA_SYSTEM_PROMPT
        self.chain          = None
        self.build()

    # ── Chain Construction ────────────────────────────────────────

    def build(self, system_prompt: str = None):
        """
        Update the system prompt.
        Called on startup and when the user changes persona in Settings.
        """
        if system_prompt is not None:
            self._system_prompt = system_prompt

        # History-aware reformulation prompt (used in _stream_standard)
        self._ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    # ── Public streaming entry point ──────────────────────────────

    def answer_stream(
        self,
        question: str,
        history: List[Dict] = None,
        use_hyde: bool = False,
    ) -> Generator:
        """
        Stream the answer token-by-token.

        use_hyde=False → Standard history-aware LCEL chain
        use_hyde=True  → HyDE: hypothetical doc → embed → retrieve → generate

        Yields:
          {"type": "token",  "content": "<text>"}
          {"type": "done",   "sources": [...], "token_info": "...",
                             "hypothetical_doc": str | None}
        """
        lc_history = self._to_lc_messages(history or [])

        if use_hyde:
            yield from self._stream_hyde(question, lc_history)
        else:
            yield from self._stream_standard(question, lc_history)

    def answer(
        self,
        question: str,
        history: List[Dict] = None,
        use_hyde: bool = False,
    ) -> Dict[str, Any]:
        """Return a non-streaming answer with sources, latency and token usage."""
        start = perf_counter()
        lc_history = self._to_lc_messages(history or [])
        total_prompt_tokens = 0
        total_completion_tokens = 0

        search_query = question

        if use_hyde:
            hypothetical_doc, prompt_tokens, completion_tokens = self._generate_hypothetical_doc_with_usage(question)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            search_query = hypothetical_doc
        elif lc_history:
            search_query, prompt_tokens, completion_tokens = self._rewrite_query_with_history(question, lc_history)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        if use_hyde:
            retrieved = self.retrieve(search_query)
        else:
            retrieved, prompt_tokens, completion_tokens = self._retrieve_standard(search_query)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        retrieved_docs = [doc for doc, _ in retrieved]
        doc_scores = [score for _, score in retrieved]
        sources = self._extract_sources(retrieved_docs, doc_scores)

        if not retrieved_docs:
            return {
                "answer": NO_CONTEXT_FALLBACK_MESSAGE,
                "sources": sources,
                "latency_seconds": round(perf_counter() - start, 3),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }

        context = self._format_docs(retrieved_docs)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt + "\n\nContext:\n" + context),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        response = (qa_prompt | self.llm).invoke({
            "input": question,
            "chat_history": lc_history,
        })
        answer_text = getattr(response, "content", "")
        prompt_tokens, completion_tokens = self._extract_token_usage(response)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        return {
            "answer": answer_text,
            "sources": sources,
            "latency_seconds": round(perf_counter() - start, 3),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

    # ── Standard RAG ──────────────────────────────────────────

    def _stream_standard(self, question: str, lc_history: list) -> Generator:
        """History-aware retrieval + generation with relevance scores."""
        # Step 1: Reformulate question if there is chat history
        search_query = question
        if lc_history:
            search_query = (
                self._ctx_prompt | self.llm | StrOutputParser()
            ).invoke({"input": question, "chat_history": lc_history})

        # Step 2: Retrieve with scores, filtered by threshold (with optional query expansion)
        retrieved, _, _ = self._retrieve_standard(search_query)
        retrieved_docs = [doc for doc, _ in retrieved]
        doc_scores = [score for _, score in retrieved]
        sources  = self._extract_sources(retrieved_docs, doc_scores)

        # No relevant chunks → fixed Greek fallback, do NOT call the LLM to answer.
        if not retrieved_docs:
            yield {"type": "token", "content": NO_CONTEXT_FALLBACK_MESSAGE}
            yield self._done_event(question, NO_CONTEXT_FALLBACK_MESSAGE, sources, hypothetical_doc=None)
            return

        context = self._format_docs(retrieved_docs)

        # Step 3: Build prompt and stream answer
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt + "\n\nContext:\n" + context),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        collected = ""
        for token in (qa_prompt | self.llm | StrOutputParser()).stream({
            "input":        question,
            "chat_history": lc_history,
        }):
            collected += token
            yield {"type": "token", "content": token}

        yield self._done_event(question, collected, sources, hypothetical_doc=None)

    # ── HyDE RAG ─────────────────────────────────────────────────

    def _stream_hyde(self, question: str, lc_history: list) -> Generator:
        """
        HyDE: generate hypothetical doc → embed → retrieve → generate.
        """
        # Signal the UI immediately so it shows a status (not a frozen screen)
        yield {"type": "status", "message": "⚗️ Generating hypothetical document…"}

        # Step 1 — Generate hypothetical document
        logger.info("HyDE: generating hypothetical document…")
        hypothetical_doc = self._generate_hypothetical_doc(question)
        logger.info(f"HyDE doc: {hypothetical_doc[:80]}…")

        # Step 2 — Retrieve using hypothetical doc, filtered by relevance score
        retrieved = self.retrieve(hypothetical_doc)
        retrieved_docs = [doc for doc, _ in retrieved]
        doc_scores = [score for _, score in retrieved]
        sources  = self._extract_sources(retrieved_docs, doc_scores)

        # No relevant chunks → fixed Greek fallback, do NOT call the LLM to answer.
        if not retrieved_docs:
            yield {"type": "token", "content": NO_CONTEXT_FALLBACK_MESSAGE}
            yield self._done_event(question, NO_CONTEXT_FALLBACK_MESSAGE, sources, hypothetical_doc)
            return

        context = self._format_docs(retrieved_docs)

        # Step 3 — Build a one-shot prompt (no chain needed, context is already ready)
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt + "\n\nContext:\n" + context),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Step 4 — Stream the answer
        collected = ""
        for token in (qa_prompt | self.llm | StrOutputParser()).stream({
            "input":        question,
            "chat_history": lc_history,
        }):
            collected += token
            yield {"type": "token", "content": token}

        yield self._done_event(question, collected, sources, hypothetical_doc)

    # ── Helpers ───────────────────────────────────────────────────

    def _generate_hypothetical_doc(self, question: str) -> str:
        """Ask the LLM to write a passage that would answer the question."""
        hypothetical_doc, _, _ = self._generate_hypothetical_doc_with_usage(question)
        return hypothetical_doc

    def _generate_hypothetical_doc_with_usage(self, question: str) -> tuple[str, int, int]:
        """Generate a HyDE passage and return it with token usage."""
        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT_TEMPLATE)
        response = (prompt | self.llm).invoke({"question": question})
        prompt_tokens, completion_tokens = self._extract_token_usage(response)
        return getattr(response, "content", ""), prompt_tokens, completion_tokens

    def _rewrite_query_with_history(self, question: str, lc_history: list) -> tuple[str, int, int]:
        """Rewrite follow-up questions into standalone queries and return usage."""
        response = (self._ctx_prompt | self.llm).invoke({
            "input": question,
            "chat_history": lc_history,
        })
        prompt_tokens, completion_tokens = self._extract_token_usage(response)
        return getattr(response, "content", ""), prompt_tokens, completion_tokens

    def retrieve(self, query: str) -> list[tuple[Any, float]]:
        """Retrieve docs for a query after applying top_k and relevance threshold."""
        config = get_config()
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=config.top_k,
        )
        return [(doc, score) for doc, score in results if score >= config.min_relevance_score]

    def retrieve_many(self, queries: list[str]) -> list[tuple[Any, float]]:
        """Retrieve for several queries, dedupe chunks (keep best score), cap at max_context_chunks."""
        best: dict[tuple, tuple[Any, float]] = {}
        for query in queries:
            for doc, score in self.retrieve(query):
                key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
                if key not in best or score > best[key][1]:
                    best[key] = (doc, score)
        merged = sorted(best.values(), key=lambda item: item[1], reverse=True)
        return merged[:get_config().max_context_chunks]

    def _retrieve_standard(self, search_query: str) -> tuple[list[tuple[Any, float]], int, int]:
        """Standard-mode retrieval; returns the results plus token usage of the expansion call."""
        if not get_config().query_expansion:
            return self.retrieve(search_query), 0, 0
        queries, prompt_tokens, completion_tokens = self._expand_query(search_query)
        return self.retrieve_many(queries), prompt_tokens, completion_tokens

    def _expand_query(self, query: str) -> tuple[list[str], int, int]:
        """Split a query into focused sub-queries; the original query is always kept first."""
        limit = get_config().num_subqueries
        prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
        response = (prompt | self.llm).invoke({"question": query, "n": limit})
        prompt_tokens, completion_tokens = self._extract_token_usage(response)

        subqueries = []
        for line in getattr(response, "content", "").splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line:
                subqueries.append(line)
        queries = list(dict.fromkeys([query, *subqueries[:limit]]))
        logger.info(f"Query expansion: {queries}")
        return queries, prompt_tokens, completion_tokens

    @staticmethod
    def _format_docs(docs) -> str:
        def label(d) -> str:
            source = d.metadata.get("source", "?")
            page = d.metadata.get("page")
            # Loader pages are 0-based; the label shows the human page number.
            return f"[Πηγή: {source}, σελ. {int(page) + 1}]" if isinstance(page, int) else f"[Πηγή: {source}]"

        return "\n\n---\n\n".join(f"{label(d)}\n{d.page_content}" for d in docs)

    @staticmethod
    def _extract_sources(docs, scores=None) -> list:
        """Extract source metadata. Scores are available only in HyDE mode."""
        _scores = scores if scores is not None else [None] * len(docs)
        return [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page":   doc.metadata.get("page", ""),
                "chunk":  doc.page_content,
                "score":  round(score, 4) if score is not None else None,
            }
            for doc, score in zip(docs, _scores)
        ]

    @staticmethod
    def _extract_token_usage(response: Any) -> tuple[int, int]:
        """Read token usage from the model response metadata."""
        usage_metadata = getattr(response, "usage_metadata", None) or {}
        if usage_metadata:
            prompt_tokens = int(usage_metadata.get("input_tokens", 0))
            completion_tokens = int(usage_metadata.get("output_tokens", 0))
            return prompt_tokens, completion_tokens

        response_metadata = getattr(response, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage", {})
        prompt_tokens = int(token_usage.get("prompt_tokens", 0))
        completion_tokens = int(token_usage.get("completion_tokens", 0))
        return prompt_tokens, completion_tokens

    @staticmethod
    def _done_event(question: str, collected: str, sources: list,
                    hypothetical_doc: str | None) -> dict:
        est_tokens = (len(question) + len(collected)) // 4
        return {
            "type":             "done",
            "sources":          sources,
            "token_info":       f"~{est_tokens} tokens | ~${est_tokens * COST_PER_TOKEN:.5f}",
            "hypothetical_doc": hypothetical_doc,
        }

    @staticmethod
    def _to_lc_messages(history: List[Dict]):
        """Convert Gradio {role, content} dicts → LangChain message objects."""
        messages = []
        for msg in history[-6:]:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if "---\n📚" in content:
                content = content.split("---\n📚")[0].strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

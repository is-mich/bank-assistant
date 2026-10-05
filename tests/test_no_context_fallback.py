"""The no-context fallback must be returned WITHOUT any LLM call."""

import pytest
from langchain_core.documents import Document

from document_qa_bot.config import get_config
from document_qa_bot.generation.chain_builder import ChainBuilder
from document_qa_bot.generation.prompts import NO_CONTEXT_FALLBACK_MESSAGE


class FakeStore:
    """Vector store stub returning fixed (doc, score) pairs and recording the requested k."""

    def __init__(self, results):
        self.results = results
        self.last_k = None

    def similarity_search_with_relevance_scores(self, query, k):
        self.last_k = k
        return self.results


class NoLLM:
    """Stands in for the LLM; any use (e.g. piping into a chain) fails the test."""


@pytest.fixture(autouse=True)
def plain_retrieval(monkeypatch):
    # Query expansion would call the LLM before retrieval; it is not under test here.
    monkeypatch.setattr(get_config(), "query_expansion", False)


def doc(text="chunk"):
    return Document(page_content=text, metadata={"source": "doc.pdf", "page": 0})


def test_retrieve_applies_top_k_and_threshold(monkeypatch):
    cfg = get_config()
    monkeypatch.setattr(cfg, "top_k", 5)
    monkeypatch.setattr(cfg, "min_relevance_score", 0.30)
    store = FakeStore([(doc("a"), 0.90), (doc("b"), 0.30), (doc("c"), 0.29)])
    builder = ChainBuilder(NoLLM(), store)

    kept = builder.retrieve("question")

    assert store.last_k == 5
    assert [d.page_content for d, _ in kept] == ["a", "b"]


@pytest.mark.parametrize("scores", [[], [0.05]])  # nothing retrieved, or a chunk below the threshold
def test_answer_returns_fallback_without_llm_when_nothing_found(monkeypatch, scores):
    monkeypatch.setattr(get_config(), "min_relevance_score", 0.15)
    builder = ChainBuilder(NoLLM(), FakeStore([(doc(), s) for s in scores]))

    out = builder.answer("ερώτηση")

    assert out["answer"] == NO_CONTEXT_FALLBACK_MESSAGE
    assert out["sources"] == []
    assert out["prompt_tokens"] == 0 and out["completion_tokens"] == 0


def test_answer_stream_returns_fallback_without_llm(monkeypatch):
    monkeypatch.setattr(get_config(), "min_relevance_score", 0.15)
    builder = ChainBuilder(NoLLM(), FakeStore([(doc(), 0.05)]))

    events = list(builder.answer_stream("ερώτηση"))

    assert [e["type"] for e in events] == ["token", "done"]
    assert events[0]["content"] == NO_CONTEXT_FALLBACK_MESSAGE
    assert events[1]["sources"] == []

import pytest

from document_qa_bot.evaluation.metrics import (
    GoldRef, chunk_matches, cost_usd, parse_gold, recall_at_k, reciprocal_rank,
)

DOC = "doc.pdf"


def chunk(page, source=DOC):
    """A retrieved chunk as returned by the pipeline (0-based page)."""
    return {"source": source, "page": page}


def test_parse_gold_document_only():
    assert parse_gold("doc.pdf") == [GoldRef("doc.pdf")]


def test_parse_gold_with_pages_and_several_refs():
    assert parse_gold("doc.pdf:39; doc.pdf:42") == [GoldRef("doc.pdf", 39), GoldRef("doc.pdf", 42)]


def test_parse_gold_empty():
    assert parse_gold("") == []
    assert parse_gold("  ;  ") == []


def test_chunk_matches_uses_one_based_page():
    assert chunk_matches(chunk(38), GoldRef(DOC, 39))
    assert not chunk_matches(chunk(39), GoldRef(DOC, 39))


def test_chunk_matches_document_level_and_other_document():
    assert chunk_matches(chunk(5), GoldRef(DOC))
    assert not chunk_matches(chunk(5, source="other.pdf"), GoldRef(DOC))


def test_chunk_matches_ignores_directory_in_source():
    assert chunk_matches(chunk(0, source="C:/tmp/doc.pdf"), GoldRef(DOC, 1))


def test_chunk_without_page_does_not_match_page_ref():
    assert not chunk_matches(chunk(""), GoldRef(DOC, 1))
    assert chunk_matches(chunk(""), GoldRef(DOC))


def test_recall_at_k_counts_gold_refs_hit_in_top_k():
    gold = [GoldRef(DOC, 1), GoldRef(DOC, 3)]
    retrieved = [chunk(0), chunk(7), chunk(2)]
    assert recall_at_k(retrieved, gold, k=3) == 1.0
    assert recall_at_k(retrieved, gold, k=1) == 0.5
    assert recall_at_k(retrieved, gold, k=2) == 0.5


def test_recall_at_k_none_without_gold():
    assert recall_at_k([chunk(0)], [], k=3) is None


def test_recall_at_k_zero_when_nothing_retrieved():
    assert recall_at_k([], [GoldRef(DOC)], k=3) == 0.0


def test_reciprocal_rank_first_relevant_position():
    gold = [GoldRef(DOC, 3)]
    assert reciprocal_rank([chunk(2), chunk(0)], gold, k=5) == 1.0
    assert reciprocal_rank([chunk(0), chunk(1), chunk(2)], gold, k=5) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_if_relevant_chunk_is_beyond_k():
    assert reciprocal_rank([chunk(0), chunk(1), chunk(2)], [GoldRef(DOC, 3)], k=2) == 0.0


def test_reciprocal_rank_none_without_gold():
    assert reciprocal_rank([chunk(0)], [], k=3) is None


def test_cost_usd_from_token_usage():
    assert cost_usd(1_000_000, 1_000_000, 0.15, 0.60) == pytest.approx(0.75)
    assert cost_usd(2000, 500, 0.15, 0.60) == pytest.approx(0.0006)

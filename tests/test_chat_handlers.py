import pytest

from document_qa_bot.chat_handlers import _cited_sources, answer_question, capture_and_clear
from document_qa_bot.config import get_config


class FakePipeline:
    """Replays a list of events; an Exception in the list is raised in the stream."""

    def __init__(self, events):
        self.events = events
        self.calls = []

    def answer_stream(self, question, history, use_hyde):
        self.calls.append((question, list(history), use_hyde))
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event


@pytest.fixture(autouse=True)
def no_stream_delay(monkeypatch):
    monkeypatch.setattr(get_config(), "stream_delay_seconds", 0.0)


def user_history(text="ερώτηση"):
    return [{"role": "user", "content": text}]


def source(page, score=0.5):
    return {"source": "doc.pdf", "page": page, "chunk": f"κείμενο {page}", "score": score}


def run(events, on_done=None, history=None):
    pipeline = FakePipeline(events)
    outputs = list(answer_question("ερώτηση", history or user_history(), False, on_done=on_done, pipeline=pipeline))
    return pipeline, outputs


def test_cited_sources_only_returns_cited_pages():
    sources = [source(6), source(38), source(41)]
    assert [s["page"] for s in _cited_sources("Α [Πηγή: doc.pdf, σελ. 7] Β [Πηγή: doc.pdf, σελ. 39]", sources)] == [6, 38]
    assert _cited_sources("Καλησπέρα!", sources) == []
    assert _cited_sources("Α [Πηγή: doc.pdf]", sources) == sources


def test_capture_and_clear_adds_question_unless_blank():
    assert capture_and_clear("Γεια", [])[2] == [{"role": "user", "content": "Γεια"}]
    assert capture_and_clear("   ", [])[2] == []


def test_answer_streams_tokens_and_appends_sources_when_cited():
    events = [
        {"type": "token", "content": "Η απάντηση "},
        {"type": "token", "content": "[Πηγή: doc.pdf, σελ. 7]"},
        {"type": "done", "sources": [source(6), source(9)], "token_info": "~10 tokens", "hypothetical_doc": None},
    ]
    pipeline, outputs = run(events)

    final_history = outputs[-1][0]
    assert final_history[0]["role"] == "user" and final_history[1]["role"] == "assistant"
    assert final_history[1]["content"].startswith("Η απάντηση [Πηγή: doc.pdf, σελ. 7]")
    assert "doc.pdf" in final_history[1]["content"].split("Sources:")[1]
    assert pipeline.calls == [("ερώτηση", [], False)]  # history passed to the pipeline excludes the question


def test_on_done_receives_answer_cited_sources_and_latency():
    seen = []
    events = [
        {"type": "token", "content": "Απάντηση [Πηγή: doc.pdf, σελ. 7]"},
        {"type": "done", "sources": [source(6), source(9)], "token_info": "", "hypothetical_doc": None},
    ]
    run(events, on_done=seen.append)

    [info] = seen
    assert info["question"] == "ερώτηση" and info["answer"].startswith("Απάντηση")
    assert [s["page"] for s in info["sources"]] == [6]  # cited only
    assert info["retrieved"] == 2 and info["error"] is None and info["latency_seconds"] >= 0


def test_failing_hook_does_not_break_the_answer():
    def broken(info):
        raise RuntimeError("disk full")

    events = [
        {"type": "token", "content": "Απάντηση"},
        {"type": "done", "sources": [], "token_info": "", "hypothetical_doc": None},
    ]
    _, outputs = run(events, on_done=broken)
    assert outputs[-1][0][-1]["content"] == "Απάντηση"


def test_pipeline_error_is_shown_and_reported_to_the_hook():
    seen = []
    _, outputs = run([RuntimeError("boom")], on_done=seen.append)

    assert outputs[-1][0][-1]["content"].startswith("❌ Error: boom")
    assert seen[0]["error"] == "RuntimeError: boom"


def test_blank_question_does_nothing():
    pipeline = FakePipeline([])
    outputs = list(answer_question("  ", user_history(), False, pipeline=pipeline))
    assert outputs == [(user_history(), "", "")] and pipeline.calls == []

"""
chat_handlers.py
----------------
Chat callbacks shared by the admin app (gradio_app.py) and the user app (user_app.py).
"""

import html
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from gradio.themes import Font, Soft

from .config import get_config

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[Πηγή:[^\]]*\]")
PAGE_RE     = re.compile(r"σελ\.\s*(\d+)")

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """Create the RAGPipeline once and share it between requests."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from .rag_pipeline import RAGPipeline  # heavy imports only when first needed
            _pipeline = RAGPipeline()
    return _pipeline


def greek_theme() -> Soft:
    """Soft theme with system fonts that render Greek well; no Google Fonts request."""
    # Font objects, not strings: launch() compares themes with the built-in ones.
    return Soft(
        font=(Font("Segoe UI"), Font("Arial"), Font("system-ui"), Font("sans-serif")),
        font_mono=(Font("Consolas"), Font("Courier New"), Font("monospace")),
    )


def capture_and_clear(question: str, history: list):
    """
    Step 1 (queue=False → instant): save question to State, clear the textbox
    and show the question in the chat. Runs BEFORE retrieval/LLM start.
    """
    history = list(history or [])
    if question.strip():
        history.append({"role": "user", "content": question})
    return question, "", history


def _cited_sources(answer: str, sources: list) -> list:
    """Keep only sources the answer cites; none if it cites nothing (greeting, refusal)."""
    citations = CITATION_RE.findall(answer)
    if not citations:
        return []
    pages = {int(n) for c in citations for n in PAGE_RE.findall(c)}
    if not pages:
        return sources
    return [s for s in sources if s.get("page") != "" and int(s["page"]) + 1 in pages]


def _sources_markdown(sources: list) -> str:
    """Sources block shown under an answer; the full chunk is the hover tooltip."""
    md = "\n\n---\n📚 **Sources:**\n"
    for i, s in enumerate(sources, 1):
        filename = Path(s["source"]).name
        page     = f" · page {int(s['page']) + 1}" if s.get("page") != "" else ""
        score    = f" `[{s['score']:.2f}]`" if s.get("score") is not None else ""
        chunk    = s.get("chunk", "")[:300].replace("\n", " ")
        tooltip  = html.escape(s.get("chunk", "")[:1500], quote=True).replace("\n", "&#10;")
        md += (
            f'\n<span title="{tooltip}">**[{i}] `{filename}`{page}{score}**</span>'
            f"\n> {chunk}…\n"
        )
    return md


def _notify(on_done: Optional[Callable[[dict], None]], info: dict) -> None:
    """Call the optional hook; a failing hook (e.g. logging) must never break the answer."""
    if on_done is None:
        return
    try:
        on_done(info)
    except Exception:
        logger.exception("on_done hook failed")


def answer_question(
    question: str,
    history: list,
    use_hyde: bool,
    on_done: Optional[Callable[[dict], None]] = None,
    pipeline=None,
):
    """
    Stream RAG answer. Routes to HyDE or standard retrieval based on checkbox.

    `on_done(info)` is called once per answer with: question, answer, sources (cited only),
    retrieved (number of chunks), hypothetical_doc, use_hyde, latency_seconds, error.
    """
    if not question.strip():
        yield history, "", ""
        return

    history = list(history or [])  # last message is the user's (added by capture_and_clear)
    reply   = None
    accumulated = ""
    hyde_md     = ""
    start  = time.perf_counter()
    paused = 0.0  # time spent in the cosmetic stream delay, excluded from latency

    def info(error=None, sources=(), retrieved=0, hypo_doc=None) -> dict:
        return {
            "question": question, "answer": accumulated, "sources": list(sources),
            "retrieved": retrieved, "hypothetical_doc": hypo_doc, "use_hyde": use_hyde,
            "latency_seconds": time.perf_counter() - start - paused, "error": error,
        }

    try:
        # No yield before the first token: Gradio shows its typing dots while the event is pending,
        # so "status" events are skipped (a yield would end the pending state).
        stream = (pipeline or get_pipeline()).answer_stream(question, history=history[:-1], use_hyde=use_hyde)
        for event in stream:
            if event["type"] == "token":
                if reply is None:
                    reply = {"role": "assistant", "content": ""}
                    history.append(reply)
                accumulated += event["content"]
                reply["content"] = accumulated
                yield history, "⏳ Generating…", hyde_md
                delay = get_config().stream_delay_seconds
                time.sleep(delay)
                paused += delay

            elif event["type"] == "done":
                all_sources = event.get("sources", [])
                sources     = _cited_sources(accumulated, all_sources)
                token_info  = event.get("token_info", "")
                hypo_doc    = event.get("hypothetical_doc")

                if sources and reply is not None:
                    reply["content"] = accumulated + _sources_markdown(sources)

                mode_label = "🧪 HyDE" if use_hyde else "🔍 Standard"
                if hypo_doc:
                    hyde_md = f"**💡 Hypothetical Document (HyDE):**\n\n> {hypo_doc}"

                _notify(on_done, info(sources=sources, retrieved=len(all_sources), hypo_doc=hypo_doc))
                yield history, f"{mode_label} | 🔢 {token_info}", hyde_md

    except Exception as e:
        if reply is None:
            reply = {"role": "assistant", "content": ""}
            history.append(reply)
        reply["content"] = f"❌ Error: {e}"
        _notify(on_done, info(error=f"{type(e).__name__}: {e}"))
        yield history, "", ""

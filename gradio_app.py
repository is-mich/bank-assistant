"""
Module 4: Document Q&A Bot — LangChain + LangSmith Edition
============================================================
Usage:
    python gradio_app.py
    → http://127.0.0.1:7860

LangSmith traces: https://smith.langchain.com
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

import gradio as gr
from document_qa_bot.chat_handlers import (
    answer_question, capture_and_clear, get_pipeline, greek_theme,
)
from document_qa_bot.generation.prompts import (
    QA_SYSTEM_PROMPT, QA_DETAILED_PROMPT, QA_CONCISE_PROMPT,
)

# ── Callbacks ──────────────────────────────────────────────────────────────────



def ingest_file(files):
    """Generator: shows spinner instantly, then ingests each file and reports."""
    if not files:
        yield "Please upload at least one file first."
        return

    if not isinstance(files, list):
        files = [files]

    yield f"⏳ Processing {len(files)} file(s)…"

    results = []
    for f in files:
        name = Path(f.name).name
        try:
            n = get_pipeline().ingest(f.name)
            results.append(f"✅ **{name}** → {n} chunks added")
        except Exception as e:
            results.append(f"⚠️ **{name}**: {e}")

    yield "\n\n".join(results)


def apply_system_prompt(prompt: str):
    get_pipeline().set_system_prompt(prompt)
    return "✅ System prompt applied!"


def reset_system_prompt():
    return QA_SYSTEM_PROMPT, "↩️ Reset to default."


def get_stats():
    try:
        stats = get_pipeline().get_stats()
        sources = "\n".join(f"- {s}" for s in stats["sources"][:15]) or "_No documents yet._"
        return f"""### 📊 Knowledge Base Stats
- **Total Chunks:** {stats['total_chunks']}
- **Unique Sources:** {stats['unique_sources']}

**Documents:**
{sources}"""
    except Exception as e:
        return f"Error: {e}"


def clear_database():
    try:
        get_pipeline().clear()
        return "✅ Database cleared."
    except Exception as e:
        return f"❌ Error: {e}"


def _docs_dropdown():
    """Dropdown listing the documents currently in the knowledge base."""
    try:
        sources = get_pipeline().get_stats()["sources"]
    except Exception:
        sources = []
    return gr.Dropdown(choices=sources, value=None)


def refresh_stats_and_docs():
    return get_stats(), _docs_dropdown()


def delete_document(name: str):
    """Delete one document (all its chunks) and refresh stats and the dropdown."""
    if not name:
        return "Select a document first.", get_stats(), _docs_dropdown()
    try:
        removed = get_pipeline().delete_document(name)
        status = f"✅ **{name}** deleted ({removed} chunks removed)."
    except Exception as e:
        status = f"❌ Error: {e}"
    return status, get_stats(), _docs_dropdown()


PAGES = ["chat", "upload", "stats", "settings"]


def _goto(page: str):
    """Build a handler that shows one page column and highlights its menu button."""
    def go():
        return (
            [gr.Column(visible=(p == page)) for p in PAGES]
            + [gr.Button(variant="primary" if p == page else "secondary") for p in PAGES]
        )
    return go


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="📚 Document Q&A Bot — LangChain Edition") as app:

    with gr.Sidebar():
        gr.Markdown("## 📚 Menu")
        nav_buttons = {
            "chat":     gr.Button("💬 Chat", variant="primary"),
            "upload":   gr.Button("📁 Upload Documents", variant="secondary"),
            "stats":    gr.Button("📊 Stats", variant="secondary"),
            "settings": gr.Button("⚙️ Settings", variant="secondary"),
        }

    gr.Markdown("""
    # 📚 Document Q&A Bot — LangChain Edition
    Powered by **LangChain** · **ChromaDB** · **GPT-4o-mini** · **LangSmith** observability
    """)

    with gr.Column():

        # ── Page 1: Chat ───────────────────────────────────────
        with gr.Column(visible=True) as page_chat:
            chatbot = gr.Chatbot(
                height=400,
                autoscroll=False,
                placeholder="Ανέβασε ένα έγγραφο από το μενού **📁 Upload Documents** και ρώτα με ό,τι θέλεις!",
                show_label=False,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    label="Question",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Ask ➤", variant="primary", scale=1)

            with gr.Row():
                clear_chat  = gr.Button("🗑️ Clear Chat", scale=1)
                hyde_toggle = gr.Checkbox(label="🧪 Use HyDE", value=False, scale=1,
                                          info="Generates a hypothetical answer doc for better retrieval")
                token_display = gr.Markdown("", label="")

            with gr.Accordion("💡 HyDE — Hypothetical Document", open=False):
                hyde_display = gr.Markdown(
                    "_Enable HyDE and ask a question to see the generated document._"
                )

            # ── State: holds the question after the textbox is cleared ─────────
            question_state = gr.State("")

            msg.submit(
                capture_and_clear, [msg, chatbot], [question_state, msg, chatbot], queue=False
            ).then(
                answer_question, [question_state, chatbot, hyde_toggle],
                [chatbot, token_display, hyde_display]
            )
            send_btn.click(
                capture_and_clear, [msg, chatbot], [question_state, msg, chatbot], queue=False
            ).then(
                answer_question, [question_state, chatbot, hyde_toggle],
                [chatbot, token_display, hyde_display]
            )
            clear_chat.click(lambda: ([], "", ""), outputs=[chatbot, token_display, hyde_display])

        # ── Page 2: Upload ─────────────────────────────────────
        with gr.Column(visible=False) as page_upload:
            gr.Markdown("### Add documents to the knowledge base")

            file_upload = gr.File(
                label="Upload Document(s)",
                file_types=[".pdf", ".txt", ".md", ".docx"],
                file_count="multiple",
            )
            ingest_btn = gr.Button("⬆️ Ingest Document", variant="primary")
            ingest_status = gr.Markdown()

            ingest_event = ingest_btn.click(ingest_file, file_upload, ingest_status)

            gr.Markdown("""
            **Supported:** PDF, TXT, Markdown, DOCX

            Each document is:
            1. Loaded via **LangChain** Document Loaders
            2. Split with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
            3. Embedded with `OpenAIEmbeddings` (`text-embedding-3-small`)
            4. Stored in **ChromaDB** (local persistence)
            """)

        # ── Page 3: Stats ──────────────────────────────────────
        with gr.Column(visible=False) as page_stats:
            stats_output = gr.Markdown()
            with gr.Row():
                doc_dropdown   = gr.Dropdown(choices=[], value=None, label="Document", scale=3)
                delete_doc_btn = gr.Button("🗑️ Delete Document", variant="stop", scale=1)
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh")
                clear_db_btn = gr.Button("🗑️ Clear Database", variant="stop")
            clear_status = gr.Markdown()

            refresh_btn.click(refresh_stats_and_docs, outputs=[stats_output, doc_dropdown])
            delete_doc_btn.click(
                delete_document, doc_dropdown, [clear_status, stats_output, doc_dropdown]
            )
            clear_db_btn.click(clear_database, outputs=clear_status).then(
                refresh_stats_and_docs, outputs=[stats_output, doc_dropdown]
            )
            ingest_event.then(refresh_stats_and_docs, outputs=[stats_output, doc_dropdown])
            app.load(refresh_stats_and_docs, outputs=[stats_output, doc_dropdown])

        # ── Page 4: Settings ───────────────────────────────────
        with gr.Column(visible=False) as page_settings:
            gr.Markdown("### Customize the bot's persona")

            with gr.Row():
                preset_btns = [
                    gr.Button("🎓 Academic", scale=1),
                    gr.Button("✂️ Concise", scale=1),
                    gr.Button("↩️ Default", scale=1),
                ]

            system_prompt_box = gr.Textbox(
                value=QA_SYSTEM_PROMPT,
                label="System Prompt",
                lines=8,
                info="Changes apply immediately to the next question.",
            )

            apply_btn = gr.Button("✅ Apply", variant="primary")
            prompt_status = gr.Markdown()

            apply_btn.click(apply_system_prompt, system_prompt_box, prompt_status)
            preset_btns[0].click(lambda: QA_DETAILED_PROMPT, outputs=system_prompt_box)
            preset_btns[1].click(lambda: QA_CONCISE_PROMPT,  outputs=system_prompt_box)
            preset_btns[2].click(reset_system_prompt, outputs=[system_prompt_box, prompt_status])

    pages = [page_chat, page_upload, page_stats, page_settings]
    for name, btn in nav_buttons.items():
        btn.click(_goto(name), outputs=pages + list(nav_buttons.values()), queue=False)

    gr.Markdown("""
    ---
    **LangChain pipeline:** Loader → Splitter → Embeddings → ChromaDB → History-Aware Retriever → GPT-4o-mini
    | **LangSmith:** Enable tracing via `LANGCHAIN_TRACING_V2=true` in `.env`
    """)


if __name__ == "__main__":
    app.launch(share=False, theme=greek_theme())

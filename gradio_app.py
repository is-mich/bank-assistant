"""
Module 4: RAG-based bank assistant (Admin + User)
==================================================

Usage:
    python gradio_app.py
    python gradio_app.py --share
"""

import argparse
import hmac
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

import gradio as gr

from document_qa_bot.chat_handlers import (
    answer_question,
    capture_and_clear,
    get_pipeline,
    greek_theme,
)
from document_qa_bot.config import get_config
from document_qa_bot.generation.prompts import (
    QA_CONCISE_PROMPT,
    QA_DETAILED_PROMPT,
    QA_SYSTEM_PROMPT,
)
from document_qa_bot.interaction_log import InteractionLogger

# ── Shared text ───────────────────────────────────────────────────────────────

UNAUTHORIZED_MSG = "⛔ Δεν έχετε δικαίωμα για αυτή την ενέργεια."
TITLE = "Βοηθός Γενικών Όρων Τραπεζικών Συναλλαγών"
NOTICE = """### Πριν ξεκινήσετε
Αυτός ο βοηθός απαντά **μόνο** με βάση τα δημόσια έγγραφα της Τράπεζας και χρησιμοποιείται σε πειραματική \
μελέτη. Δεν παρέχει εξατομικευμένες οικονομικές συμβουλές και δεν έχει πρόσβαση σε προσωπικούς λογαριασμούς.

- **Μην γράφετε προσωπικά στοιχεία** (ονόματα, αριθμούς λογαριασμού ή κάρτας, κωδικούς).
- Οι ερωτήσεις, οι απαντήσεις και οι αξιολογήσεις σας **καταγράφονται** για την ανάλυση της μελέτης. \
Συνδέεστε με ψευδώνυμο. Αριθμοί κάρτας, IBAN και email αντικαθίστανται αυτόματα πριν την αποθήκευση.
- Η συμμετοχή είναι προαιρετική· μπορείτε να σταματήσετε οποιαδήποτε στιγμή."""
CONSENT_LABEL = "Διάβασα τα παραπάνω και συμφωνώ με την καταγραφή της συνομιλίας μου."
CONSENT_REFUSAL = "Για να χρησιμοποιήσετε τον βοηθό, αποδεχτείτε πρώτα την καταγραφή στο πλαίσιο παραπάνω."


# ── Access control ────────────────────────────────────────────────────────────

def parse_users(raw: str | None) -> dict[str, str]:
    """Parse 'name:password;name2:password2' into a dict."""
    users: dict[str, str] = {}
    for part in (raw or "").split(";"):
        name, sep, password = part.strip().partition(":")
        if sep and name and password:
            users[name] = password
    return users


def parse_admins(raw: str | None) -> set[str]:
    """Parse semicolon-separated admin usernames."""
    return {u.strip() for u in (raw or "").split(";") if u.strip()}


def make_auth(users: dict[str, str]):
    """Gradio auth callback with constant-time password comparison."""

    def check(username: str, password: str) -> bool:
        return username in users and hmac.compare_digest(users[username].encode(), password.encode())

    return check


AUTH_USERS = parse_users(os.environ.get("APP_USERS", ""))
ADMIN_USERS = parse_admins(os.environ.get("APP_ADMINS", ""))
AUTH_ENABLED = bool(AUTH_USERS)


def role_from_username(username: str | None) -> str:
    """Resolve role for a username. Unknown users default to least privilege."""
    if not AUTH_ENABLED:
        return "admin"
    if username and username in ADMIN_USERS:
        return "admin"
    return "user"


def request_role(request: gr.Request | None) -> str:
    """Resolve role from request context."""
    if request is None:
        return "admin" if not AUTH_ENABLED else "user"
    return role_from_username(getattr(request, "username", None))


def is_admin_request(request: gr.Request | None) -> bool:
    """Permission check for admin-only operations."""
    return request_role(request) == "admin"


# ── Logging for user-mode sessions ────────────────────────────────────────────

_user_logger: InteractionLogger | None = None


def get_user_logger() -> InteractionLogger:
    """Lazy singleton logger for user-mode interactions."""
    global _user_logger
    if _user_logger is None:
        config = get_config()
        _user_logger = InteractionLogger(config.logs_path / "interactions.jsonl", experiment=config.experiment_name)
    return _user_logger


def _who(request: gr.Request) -> dict:
    return {"session": request.session_hash, "user": getattr(request, "username", None)}


# ── Admin callbacks ───────────────────────────────────────────────────────────

def ingest_file(files, request: gr.Request):
    """Generator: shows spinner instantly, then ingests each file and reports."""
    if not is_admin_request(request):
        yield UNAUTHORIZED_MSG
        return

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


def apply_system_prompt(prompt: str, request: gr.Request):
    if not is_admin_request(request):
        return UNAUTHORIZED_MSG
    get_pipeline().set_system_prompt(prompt)
    return "✅ System prompt applied!"


def reset_system_prompt(request: gr.Request):
    if not is_admin_request(request):
        return gr.skip(), UNAUTHORIZED_MSG
    return QA_SYSTEM_PROMPT, "↩️ Reset to default."


def get_stats(request: gr.Request):
    if not is_admin_request(request):
        return UNAUTHORIZED_MSG
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


def clear_database(request: gr.Request):
    if not is_admin_request(request):
        return UNAUTHORIZED_MSG
    try:
        get_pipeline().clear()
        return "✅ Database cleared."
    except Exception as e:
        return f"❌ Error: {e}"


def _docs_dropdown(request: gr.Request):
    """Dropdown listing the documents currently in the knowledge base."""
    if not is_admin_request(request):
        return gr.Dropdown(choices=[], value=None)
    try:
        sources = get_pipeline().get_stats()["sources"]
    except Exception:
        sources = []
    return gr.Dropdown(choices=sources, value=None)


def refresh_stats_and_docs(request: gr.Request):
    return get_stats(request), _docs_dropdown(request)


def delete_document(name: str, request: gr.Request):
    """Delete one document (all its chunks) and refresh stats and the dropdown."""
    if not is_admin_request(request):
        return UNAUTHORIZED_MSG, get_stats(request), _docs_dropdown(request)

    if not name:
        return "Select a document first.", get_stats(request), _docs_dropdown(request)
    try:
        removed = get_pipeline().delete_document(name)
        status = f"✅ **{name}** deleted ({removed} chunks removed)."
    except Exception as e:
        status = f"❌ Error: {e}"
    return status, get_stats(request), _docs_dropdown(request)


# ── User callbacks ────────────────────────────────────────────────────────────

def log_user_consent(consented: bool, request: gr.Request):
    get_user_logger().log("consent", consented=bool(consented), **_who(request))


def toggle_user_inputs(consented: bool):
    return gr.Textbox(interactive=bool(consented)), gr.Button(interactive=bool(consented))


def user_answer(question: str, history: list, consented: bool, scenario: str, request: gr.Request):
    if not consented:
        yield history + [{"role": "assistant", "content": CONSENT_REFUSAL}]
        return

    def on_done(info: dict) -> None:
        get_user_logger().log(
            "answer",
            scenario=(scenario or "").strip() or None,
            question=info["question"],
            answer=info["answer"],
            sources=[
                {"source": s["source"], "page": s["page"], "score": s["score"]}
                for s in info["sources"]
            ],
            retrieved=info["retrieved"],
            latency_seconds=round(info["latency_seconds"], 2),
            error=info["error"],
            **_who(request),
        )

    for chat_history, _status, _hyde in answer_question(question, history, use_hyde=False, on_done=on_done):
        yield chat_history


def user_on_like(data: gr.LikeData, request: gr.Request):
    get_user_logger().log(
        "feedback",
        liked=data.liked,
        message=str(data.value),
        message_index=data.index,
        **_who(request),
    )


# ── Page routing ──────────────────────────────────────────────────────────────

PAGES = ["chat", "upload", "stats", "settings"]


def _goto(page: str):
    """Show one admin page and highlight its nav button. Users stay on user page."""

    def go(request: gr.Request):
        can_admin = is_admin_request(request)
        if not can_admin:
            return (
                [gr.Column(visible=False)]
                + [gr.Column(visible=True)]
                + [gr.Column(visible=False) for _ in PAGES]
                + [gr.Button(visible=False, variant="secondary") for _ in PAGES]
            )

        return (
            [gr.Column(visible=True)]
            + [gr.Column(visible=False)]
            + [gr.Column(visible=(p == page)) for p in PAGES]
            + [gr.Button(visible=True, variant="primary" if p == page else "secondary") for p in PAGES]
        )

    return go


def initialize_access(request: gr.Request):
    """Role-based UI gating when the app loads."""
    is_admin = is_admin_request(request)
    role = request_role(request)
    username = getattr(request, "username", None)
    user_display = username or "local"
    role_badge = f"👤 Χρήστης: **{user_display}** · Ρόλος: **{role}**"

    if is_admin:
        return (
            [role_badge]
            + [gr.Column(visible=True)]
            + [gr.Column(visible=False)]
            + [gr.Column(visible=(p == "chat")) for p in PAGES]
            + [gr.Button(visible=True, variant="primary" if p == "chat" else "secondary") for p in PAGES]
        )

    return (
        [role_badge]
        + [gr.Column(visible=False)]
        + [gr.Column(visible=True)]
        + [gr.Column(visible=False) for _ in PAGES]
        + [gr.Button(visible=False, variant="secondary") for _ in PAGES]
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="RAG-based bank assistant") as app:
    with gr.Sidebar():
        with gr.Column(visible=True) as admin_sidebar:
            gr.Markdown("## 📚 Admin Menu")
            nav_buttons = {
                "chat": gr.Button("💬 Chat", variant="primary"),
                "upload": gr.Button("📁 Upload Documents", variant="secondary"),
                "stats": gr.Button("📊 Stats", variant="secondary"),
                "settings": gr.Button("⚙️ Settings", variant="secondary"),
            }

    gr.Markdown("""
    # RAG-based bank assistant
    """)
    role_badge = gr.Markdown("👤 Χρήστης: **local** · Ρόλος: **admin**")

    # ── User-only interface (from user_app) ────────────────────────────────
    with gr.Column(visible=False) as page_user:
        gr.Markdown(f"## {TITLE}")
        gr.Markdown(NOTICE)
        consent = gr.Checkbox(label=CONSENT_LABEL, value=False)
        scenario = gr.Textbox(label="Κωδικός σεναρίου (αν σας δόθηκε)", max_lines=1)

        user_chatbot = gr.Chatbot(
            height=450,
            autoscroll=False,
            show_label=False,
            placeholder="Αποδεχτείτε την καταγραφή και γράψτε την ερώτησή σας για τους Γενικούς Όρους.",
        )
        with gr.Row():
            user_msg = gr.Textbox(
                placeholder="Γράψτε την ερώτησή σας…",
                show_label=False,
                scale=4,
                interactive=False,
            )
            user_send = gr.Button("Αποστολή", variant="primary", scale=1, interactive=False)

        user_clear = gr.Button("Καθαρισμός συνομιλίας")

        user_status = gr.Markdown("")
        user_question_state = gr.State("")

        consent.change(toggle_user_inputs, consent, [user_msg, user_send], queue=False)
        consent.change(log_user_consent, consent, None, queue=False)

        for trigger in (user_msg.submit, user_send.click):
            trigger(
                capture_and_clear,
                [user_msg, user_chatbot],
                [user_question_state, user_msg, user_chatbot],
                queue=False,
            ).then(user_answer, [user_question_state, user_chatbot, consent, scenario], user_chatbot)

        user_clear.click(lambda: [], outputs=user_chatbot)
        user_chatbot.like(user_on_like, None, None)

    with gr.Column():
        # ── Admin Page 1: Chat ─────────────────────────────────────────────
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
                clear_chat = gr.Button("🗑️ Clear Chat", scale=1)
                hyde_toggle = gr.Checkbox(
                    label="🧪 Use HyDE",
                    value=False,
                    scale=1,
                    info="Generates a hypothetical answer doc for better retrieval",
                )
                token_display = gr.Markdown("", label="")

            with gr.Accordion("💡 HyDE — Hypothetical Document", open=False):
                hyde_display = gr.Markdown("_Enable HyDE and ask a question to see the generated document._")

            question_state = gr.State("")

            msg.submit(
                capture_and_clear,
                [msg, chatbot],
                [question_state, msg, chatbot],
                queue=False,
            ).then(answer_question, [question_state, chatbot, hyde_toggle], [chatbot, token_display, hyde_display])
            send_btn.click(
                capture_and_clear,
                [msg, chatbot],
                [question_state, msg, chatbot],
                queue=False,
            ).then(answer_question, [question_state, chatbot, hyde_toggle], [chatbot, token_display, hyde_display])
            clear_chat.click(lambda: ([], "", ""), outputs=[chatbot, token_display, hyde_display])

        # ── Admin Page 2: Upload ───────────────────────────────────────────
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

        # ── Admin Page 3: Stats ────────────────────────────────────────────
        with gr.Column(visible=False) as page_stats:
            stats_output = gr.Markdown()
            with gr.Row():
                doc_dropdown = gr.Dropdown(choices=[], value=None, label="Document", scale=3)
                delete_doc_btn = gr.Button("🗑️ Delete Document", variant="stop", scale=1)
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh")
                clear_db_btn = gr.Button("🗑️ Clear Database", variant="stop")
            clear_status = gr.Markdown()

            refresh_btn.click(refresh_stats_and_docs, outputs=[stats_output, doc_dropdown])
            delete_doc_btn.click(delete_document, doc_dropdown, [clear_status, stats_output, doc_dropdown])
            clear_db_btn.click(clear_database, outputs=clear_status).then(
                refresh_stats_and_docs,
                outputs=[stats_output, doc_dropdown],
            )
            ingest_event.then(refresh_stats_and_docs, outputs=[stats_output, doc_dropdown])
            app.load(refresh_stats_and_docs, outputs=[stats_output, doc_dropdown])

        # ── Admin Page 4: Settings ─────────────────────────────────────────
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
            preset_btns[1].click(lambda: QA_CONCISE_PROMPT, outputs=system_prompt_box)
            preset_btns[2].click(reset_system_prompt, outputs=[system_prompt_box, prompt_status])

    pages = [page_chat, page_upload, page_stats, page_settings]
    for name, btn in nav_buttons.items():
        btn.click(_goto(name), outputs=[admin_sidebar, page_user] + pages + list(nav_buttons.values()), queue=False)

    app.load(
        initialize_access,
        outputs=[role_badge, admin_sidebar, page_user] + pages + list(nav_buttons.values()),
    )

app.queue(default_concurrency_limit=get_config().ui_concurrency)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified role-based app (admin + user).")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve the local network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create temporary public gradio.live link")
    parser.add_argument("--no-auth", action="store_true", help="disable login (local testing only)")
    args = parser.parse_args(argv)

    if args.share and (args.no_auth or not AUTH_ENABLED):
        sys.exit("--share requires login: set APP_USERS and do not use --no-auth.")
    if args.no_auth and args.host != "127.0.0.1":
        sys.exit("--no-auth is only allowed on 127.0.0.1.")

    if args.share:
        print(
            "\nPUBLIC SHARE LINK: one link serves both roles. Login credentials decide the UI (admin/user).\n",
            flush=True,
        )

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=greek_theme(),
        auth=None if args.no_auth or not AUTH_ENABLED else make_auth(AUTH_USERS),
        auth_message=(
            "Συνδεθείτε με τα στοιχεία που σας δόθηκαν. Το UI προσαρμόζεται αυτόματα σε admin ή user."
        ) if AUTH_ENABLED and not args.no_auth else None,
    )


if __name__ == "__main__":
    main()

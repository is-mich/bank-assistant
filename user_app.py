"""
User app for scenario sessions: chat only, password login, consent, logged interactions.

    set OPENAI_API_KEY=...                       (or put it in .env; real environment variables win)
    set APP_USERS=P01:pass1;P02:pass2            (login accounts; use pseudonyms, not real names)
    python user_app.py --experiment structure_chunks --host 0.0.0.0 --port 7860
    python user_app.py --experiment structure_chunks --share      (temporary public https link)

There are no upload, delete, clear or settings handlers here: they are not just hidden,
they do not exist in this app. Interactions go to logs/interactions.jsonl.
"""

import argparse
import hmac
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Questions from users must never be sent to a tracing service.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import gradio as gr

from document_qa_bot.chat_handlers import answer_question, capture_and_clear, get_pipeline, greek_theme
from document_qa_bot.config import get_config
from document_qa_bot.evaluation.runner import apply_overrides
from document_qa_bot.interaction_log import InteractionLogger

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


def parse_users(raw: str) -> dict[str, str]:
    """Parse 'name:password;name2:password2' into a dict; passwords may contain ':'."""
    users = {}
    for part in (raw or "").split(";"):
        name, sep, password = part.strip().partition(":")
        if sep and name and password:
            users[name] = password
    return users


def make_auth(users: dict[str, str]):
    """Gradio auth callback with constant-time password comparison."""
    def check(username: str, password: str) -> bool:
        return username in users and hmac.compare_digest(users[username].encode(), password.encode())
    return check


def build_app(logger: InteractionLogger) -> gr.Blocks:
    """The chat UI. Everything the user can trigger is defined in this function."""
    config = get_config()

    def who(request: gr.Request) -> dict:
        return {"session": request.session_hash, "user": getattr(request, "username", None)}

    def log_consent(consented: bool, request: gr.Request):
        logger.log("consent", consented=bool(consented), **who(request))

    def toggle_inputs(consented: bool):
        return gr.Textbox(interactive=bool(consented)), gr.Button(interactive=bool(consented))

    def answer(question: str, history: list, consented: bool, scenario: str, request: gr.Request):
        if not consented:
            yield history + [{"role": "assistant", "content": CONSENT_REFUSAL}]
            return

        def on_done(info: dict) -> None:
            logger.log(
                "answer", scenario=(scenario or "").strip() or None,
                question=info["question"], answer=info["answer"], sources=[
                    {"source": s["source"], "page": s["page"], "score": s["score"]} for s in info["sources"]
                ],
                retrieved=info["retrieved"], latency_seconds=round(info["latency_seconds"], 2),
                error=info["error"], **who(request),
            )

        for chat_history, _status, _hyde in answer_question(question, history, use_hyde=False, on_done=on_done):
            yield chat_history

    def on_like(data: gr.LikeData, request: gr.Request):
        logger.log("feedback", liked=data.liked, message=str(data.value), message_index=data.index, **who(request))

    with gr.Blocks(title=TITLE) as app:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(NOTICE)
        consent = gr.Checkbox(label=CONSENT_LABEL, value=False)
        scenario = gr.Textbox(label="Κωδικός σεναρίου (αν σας δόθηκε)", max_lines=1)

        chatbot = gr.Chatbot(
            height=450, autoscroll=False, show_label=False,
            placeholder="Αποδεχτείτε την καταγραφή και γράψτε την ερώτησή σας για τους Γενικούς Όρους.",
        )
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Γράψτε την ερώτησή σας…", show_label=False, scale=4, interactive=False,
            )
            send = gr.Button("Αποστολή", variant="primary", scale=1, interactive=False)
        clear = gr.Button("Καθαρισμός συνομιλίας")

        question_state = gr.State("")
        consent.change(toggle_inputs, consent, [msg, send], queue=False)
        consent.change(log_consent, consent, None, queue=False)

        for trigger in (msg.submit, send.click):
            trigger(
                capture_and_clear, [msg, chatbot], [question_state, msg, chatbot], queue=False
            ).then(answer, [question_state, chatbot, consent, scenario], chatbot)
        clear.click(lambda: [], outputs=chatbot)
        chatbot.like(on_like, None, None)

    app.queue(default_concurrency_limit=config.ui_concurrency)
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="User app (chat only, login, logged).")
    parser.add_argument("--experiment", help="experiment whose Chroma collection is served")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a Config value (query-time settings), e.g. --set top_k=8")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve the local network")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-auth", action="store_true", help="disable login (local testing only)")
    parser.add_argument("--share", action="store_true",
                        help="create a temporary public gradio.live link (requires login)")
    args = parser.parse_args(argv)

    if args.share and args.no_auth:
        sys.exit("--share requires login: remove --no-auth and set APP_USERS.")

    # Variables already set in the environment (e.g. on a server) take precedence over .env.
    load_dotenv(Path(__file__).parent / ".env", override=False)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    users = parse_users(os.environ.get("APP_USERS", ""))
    if not users and not args.no_auth:
        sys.exit("No login accounts: set APP_USERS='P01:password;P02:password' (or --no-auth for local tests).")
    if args.no_auth and args.host != "127.0.0.1":
        sys.exit("--no-auth is only allowed on 127.0.0.1.")
    if args.share:
        print(
            "\nPUBLIC SHARE LINK: anyone who has the link can reach the login page, and the traffic passes "
            "through Gradio's servers.\nGive it only to participants who have an account, and stop the app "
            "(Ctrl+C) when the session ends.\n", flush=True,
        )

    config = get_config()
    if args.experiment:
        config.use_experiment(args.experiment)
    apply_overrides(config, args.overrides)

    if get_pipeline().get_stats()["total_chunks"] == 0:
        sys.exit(f"The collection of experiment {config.experiment_name!r} is empty; ingest documents first.")

    logger = InteractionLogger(config.logs_path / "interactions.jsonl", experiment=config.experiment_name)
    logger.log(
        "app_start", top_k=config.top_k, query_expansion=config.query_expansion,
        chunking_strategy=config.chunking_strategy, llm_model=config.llm_model,
        min_relevance_score=config.min_relevance_score,
    )

    app = build_app(logger)
    app.launch(
        server_name=args.host, server_port=args.port, share=args.share, theme=greek_theme(),
        auth=None if args.no_auth else make_auth(users), auth_message="Συνδεθείτε με τα στοιχεία που σας δόθηκαν.",
    )


if __name__ == "__main__":
    main()

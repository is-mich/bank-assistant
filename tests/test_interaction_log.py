import json
import threading

from document_qa_bot.interaction_log import InteractionLogger, redact


def test_redact_masks_iban_card_number_and_email():
    assert redact("Ο IBAN μου είναι GR16 0110 1250 0000 0001 2300 695") == "Ο IBAN μου είναι [IBAN]"
    assert redact("κάρτα 4111 1111 1111 1111 λήγει") == "κάρτα [ΑΡΙΘΜΟΣ] λήγει"
    assert redact("4111-1111-1111-1111") == "[ΑΡΙΘΜΟΣ]"
    assert redact("γράψτε στο nikos.p@example.gr") == "γράψτε στο [EMAIL]"


def test_redact_keeps_normal_text_and_short_numbers():
    text = "Σύμφωνα με τον όρο 2.2 στη σελ. 39 το όριο είναι 500 ευρώ"
    assert redact(text) == text


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_appends_json_lines_with_static_fields_and_greek_text(tmp_path):
    path = tmp_path / "logs" / "interactions.jsonl"
    logger = InteractionLogger(path, experiment="exp1")

    logger.log("answer", question="Τι είναι η κάρτα;", user="P01")
    logger.log("feedback", liked=True)

    first, second = read(path)
    assert first["event"] == "answer" and first["experiment"] == "exp1" and first["user"] == "P01"
    assert first["question"] == "Τι είναι η κάρτα;"  # not escaped to \u sequences
    assert "ts" in first
    assert second["event"] == "feedback" and second["liked"] is True
    assert "Τι είναι" in path.read_text(encoding="utf-8")


def test_log_masks_personal_data_in_nested_fields(tmp_path):
    path = tmp_path / "log.jsonl"
    InteractionLogger(path).log(
        "answer", question="IBAN GR16 0110 1250 0000 0001 2300 695",
        sources=[{"source": "doc.pdf", "note": "mail me at a@b.gr"}],
    )
    [record] = read(path)
    assert record["question"] == "IBAN [IBAN]"
    assert record["sources"][0]["note"] == "mail me at [EMAIL]"


def test_log_is_safe_across_threads(tmp_path):
    path = tmp_path / "log.jsonl"
    logger = InteractionLogger(path)

    def work(n):
        for i in range(25):
            logger.log("answer", question=f"q{n}-{i}")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(read(path)) == 200  # every line parses and none is lost or interleaved

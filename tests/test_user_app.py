import pytest

import user_app
from document_qa_bot.interaction_log import InteractionLogger

ADMIN_HANDLERS = {"clear_database", "delete_document", "ingest_file", "apply_system_prompt", "reset_system_prompt"}


def test_parse_users_accepts_colons_in_passwords_and_skips_bad_entries():
    users = user_app.parse_users("P01:abc;P02:pa:ss;;bad;:nopass;P03:")
    assert users == {"P01": "abc", "P02": "pa:ss"}


def test_parse_users_empty():
    assert user_app.parse_users("") == {}
    assert user_app.parse_users(None) == {}


def test_auth_accepts_only_exact_credentials():
    check = user_app.make_auth({"P01": "secret"})
    assert check("P01", "secret")
    assert not check("P01", "Secret")
    assert not check("P01", "")
    assert not check("P02", "secret")
    assert not check("", "")


def test_share_link_requires_login():
    with pytest.raises(SystemExit, match="--share requires login"):
        user_app.main(["--share", "--no-auth"])


def test_user_app_has_chat_handlers_only(tmp_path):
    app = user_app.build_app(InteractionLogger(tmp_path / "log.jsonl"))
    names = {getattr(f, "name", "") for f in app.fns.values()}

    assert "answer" in names and "capture_and_clear" in names
    assert not names & ADMIN_HANDLERS

from types import SimpleNamespace

import gradio_app
import pytest


def test_parse_users_accepts_valid_entries_only():
    users = gradio_app.parse_users("admin:secret;u1:pass;;bad;:nopass;u2:")
    assert users == {"admin": "secret", "u1": "pass"}


def test_parse_admins_reads_semicolon_list():
    admins = gradio_app.parse_admins("admin; manager ; ;supervisor")
    assert admins == {"admin", "manager", "supervisor"}


def test_make_auth_accepts_only_exact_credentials():
    check = gradio_app.make_auth({"P01": "123a", "P02": "987b"})
    assert check("P01", "123a")
    assert check("P02", "987b")
    assert not check("P01", "wrong")
    assert not check("P03", "123a")


def test_role_resolution_defaults_to_user_in_auth_mode():
    old_auth_enabled = gradio_app.AUTH_ENABLED
    old_admin_users = gradio_app.ADMIN_USERS
    try:
        gradio_app.AUTH_ENABLED = True
        gradio_app.ADMIN_USERS = {"admin"}

        assert gradio_app.role_from_username("admin") == "admin"
        assert gradio_app.role_from_username("u1") == "user"
        assert gradio_app.role_from_username(None) == "user"
    finally:
        gradio_app.AUTH_ENABLED = old_auth_enabled
        gradio_app.ADMIN_USERS = old_admin_users


def test_no_auth_mode_defaults_to_admin():
    old_auth_enabled = gradio_app.AUTH_ENABLED
    old_admin_users = gradio_app.ADMIN_USERS
    try:
        gradio_app.AUTH_ENABLED = False
        gradio_app.ADMIN_USERS = set()

        assert gradio_app.request_role(None) == "admin"
        assert gradio_app.request_role(SimpleNamespace(username=None)) == "admin"
        assert gradio_app.is_admin_request(SimpleNamespace(username="anyone"))
    finally:
        gradio_app.AUTH_ENABLED = old_auth_enabled
        gradio_app.ADMIN_USERS = old_admin_users


@pytest.mark.parametrize("action", ["clear_database", "apply_system_prompt"])
def test_user_cannot_run_admin_actions(action, monkeypatch):
    old_auth_enabled = gradio_app.AUTH_ENABLED
    old_admin_users = gradio_app.ADMIN_USERS
    try:
        gradio_app.AUTH_ENABLED = True
        gradio_app.ADMIN_USERS = {"admin"}

        def _fail_pipeline_access():
            raise AssertionError("pipeline should not be called for unauthorized users")

        monkeypatch.setattr(gradio_app, "get_pipeline", _fail_pipeline_access)

        request = SimpleNamespace(username="user1")
        if action == "clear_database":
            result = gradio_app.clear_database(request)
        else:
            result = gradio_app.apply_system_prompt("new prompt", request)

        assert result == gradio_app.UNAUTHORIZED_MSG
    finally:
        gradio_app.AUTH_ENABLED = old_auth_enabled
        gradio_app.ADMIN_USERS = old_admin_users

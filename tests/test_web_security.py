"""Tests for the dashboard's access-control layer."""

from __future__ import annotations

import base64

import pytest
from flask import Flask

import web_security
from web_security import InsecureConfigError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test from a blank DASHBOARD_* environment."""
    for var in (
        "DASHBOARD_HOST", "DASHBOARD_PORT", "DASHBOARD_USER",
        "DASHBOARD_PASSWORD", "DASHBOARD_DEBUG",
        "DASHBOARD_ALLOW_INSECURE_BIND",
    ):
        monkeypatch.delenv(var, raising=False)


def _app(**routes):
    app = Flask(__name__)
    web_security.install_auth(app)

    @app.route("/api/thing", methods=["GET"])
    def _get():
        return {"ok": True}

    @app.route("/api/thing", methods=["POST"])
    def _post():
        return {"ok": True}

    return app.test_client()


def _basic(user, password):
    raw = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


# ── is_loopback ──────────────────────────────────────────────────────

@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "", "127.5.1.9"])
def test_loopback_hosts(host):
    assert web_security.is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "::"])
def test_non_loopback_hosts(host):
    assert not web_security.is_loopback(host)


# ── resolve_bind ─────────────────────────────────────────────────────

def test_default_bind_is_loopback_and_undebugged():
    host, port, debug = web_security.resolve_bind()
    assert host == "127.0.0.1"
    assert port == 5000
    assert debug is False


def test_public_bind_without_password_is_refused(monkeypatch):
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
    with pytest.raises(InsecureConfigError, match="no password"):
        web_security.resolve_bind()


def test_public_bind_with_password_is_allowed(monkeypatch):
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    host, _, debug = web_security.resolve_bind()
    assert host == "0.0.0.0"
    assert debug is False


def test_public_bind_override_escape_hatch(monkeypatch):
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.setenv("DASHBOARD_ALLOW_INSECURE_BIND", "true")
    host, _, _ = web_security.resolve_bind()
    assert host == "0.0.0.0"


def test_debugger_refused_off_loopback(monkeypatch):
    """The Werkzeug debugger is an RCE surface — never off loopback."""
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    monkeypatch.setenv("DASHBOARD_DEBUG", "true")
    with pytest.raises(InsecureConfigError, match="debugger"):
        web_security.resolve_bind()


def test_debugger_allowed_on_loopback(monkeypatch):
    monkeypatch.setenv("DASHBOARD_DEBUG", "true")
    _, _, debug = web_security.resolve_bind()
    assert debug is True


def test_bad_port_is_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PORT", "not-a-port")
    with pytest.raises(InsecureConfigError, match="whole number"):
        web_security.resolve_bind()


# ── Authentication ───────────────────────────────────────────────────

def test_no_password_means_no_auth_challenge():
    client = _app()
    assert client.get("/api/thing").status_code == 200


def test_password_set_requires_credentials(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    client = _app()
    resp = client.get("/api/thing")
    assert resp.status_code == 401
    assert "Basic" in resp.headers["WWW-Authenticate"]


def test_correct_credentials_accepted(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    client = _app()
    resp = client.get("/api/thing", headers=_basic("rider", "hunter2"))
    assert resp.status_code == 200


def test_wrong_password_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    client = _app()
    resp = client.get("/api/thing", headers=_basic("rider", "wrong"))
    assert resp.status_code == 401


def test_wrong_user_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    monkeypatch.setenv("DASHBOARD_USER", "alice")
    client = _app()
    resp = client.get("/api/thing", headers=_basic("rider", "hunter2"))
    assert resp.status_code == 401


def test_auth_also_guards_the_page_route(monkeypatch):
    """Auth is global, not per-route — a new endpoint cannot forget it."""
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    app = Flask(__name__)
    web_security.install_auth(app)

    @app.route("/")
    def index():
        return "secret"

    assert app.test_client().get("/").status_code == 401


# ── CSRF / same-origin ───────────────────────────────────────────────

def test_cross_origin_post_blocked(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    client = _app()
    resp = client.post(
        "/api/thing",
        headers={**_basic("rider", "hunter2"), "Origin": "http://evil.test"},
    )
    assert resp.status_code == 403


def test_same_origin_post_allowed(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    client = _app()
    resp = client.post(
        "/api/thing",
        headers={
            **_basic("rider", "hunter2"),
            "Origin": "http://localhost",
            "Host": "localhost",
        },
    )
    assert resp.status_code == 200


def test_cross_origin_get_allowed():
    """GETs are not state-changing, so the origin check does not apply."""
    client = _app()
    resp = client.get("/api/thing", headers={"Origin": "http://evil.test"})
    assert resp.status_code == 200


def test_originless_post_allowed():
    """curl/scripts send no Origin and carry no ambient credential."""
    client = _app()
    assert client.post("/api/thing").status_code == 200


def test_cross_origin_blocked_before_auth_is_checked(monkeypatch):
    """A cross-origin probe must not be able to test credentials."""
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    client = _app()
    resp = client.post("/api/thing", headers={"Origin": "http://evil.test"})
    assert resp.status_code == 403


@pytest.mark.parametrize("referer,host,expected", [
    ("http://localhost:5000/", "localhost:5000", 200),
    ("http://localhost:5000/", "localhost:6000", 403),
    ("http://evil.test/page", "localhost:5000", 403),
])
def test_referer_fallback(monkeypatch, referer, host, expected):
    """Origin-less browsers fall back to Referer."""
    client = _app()
    resp = client.post(
        "/api/thing", headers={"Referer": referer, "Host": host},
    )
    assert resp.status_code == expected

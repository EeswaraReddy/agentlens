"""
Server integration tests — full ingest -> list -> detail -> stats round-trip,
using a fresh SQLite DB and FastAPI's TestClient.
"""

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def client(monkeypatch):
    # fresh DB per test
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    monkeypatch.setenv("AGENTLENS_DATABASE_URL", "sqlite:///" + tmp.name)

    # purge any cached server modules so they re-read the env var
    for mod in [m for m in list(sys.modules) if m.startswith("agentlens.server")]:
        sys.modules.pop(mod, None)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fastapi.testclient import TestClient
    from agentlens.server.app import create_app
    from agentlens.server.db import SessionLocal, Project

    app = create_app()
    cli = TestClient(app)
    cli.__enter__()                              # trigger startup -> init_db + default project
    with SessionLocal() as s:
        key = s.query(Project).first().api_key
    cli.headers.update({"X-API-Key": key})
    try:
        yield cli, key
    finally:
        cli.__exit__(None, None, None)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _trace_payload(name="support", error=False):
    return {
        "trace": {
            "trace_id": "tr_" + ("err" if error else "ok"),
            "name": name,
            "status": "error" if error else "ok",
            "duration_ms": 250.5,
            "metadata": {"user": "alice"},
            "summary": {"spans": 3, "total_tokens": 220,
                        "total_cost_usd": 0.00198, "errors": 1 if error else 0},
            "spans": [
                {"span_id": "a1", "parent_id": None, "name": "router",
                 "kind": "agent", "status": "ok", "duration_ms": 200.0},
                {"span_id": "b2", "parent_id": "a1", "name": "classify",
                 "kind": "llm", "status": "ok", "duration_ms": 80.0,
                 "model": "gpt-4o-mini", "prompt_tokens": 120,
                 "completion_tokens": 20, "cost_usd": 0.00003},
                {"span_id": "c3", "parent_id": "a1", "name": "lookup",
                 "kind": "tool", "status": "error" if error else "ok",
                 "duration_ms": 100.0,
                 "inputs": {"q": "A1"}, "outputs": {"r": "shipped"},
                 "error": "boom" if error else None},
            ],
        }
    }


def test_health(client):
    cli, _ = client
    r = cli.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_auth_required(client):
    cli, _ = client
    cli.headers.pop("X-API-Key", None)
    r = cli.get("/api/v1/traces")
    assert r.status_code == 401


def test_ingest_and_list(client):
    cli, _ = client
    r = cli.post("/api/v1/traces", json=_trace_payload())
    assert r.status_code == 201
    assert r.json()["spans_stored"] == 3

    lst = cli.get("/api/v1/traces").json()
    assert lst["total"] == 1
    item = lst["items"][0]
    assert item["status"] == "ok"
    assert item["span_count"] == 3
    assert item["total_tokens"] == 220


def test_filter_and_search(client):
    cli, _ = client
    cli.post("/api/v1/traces", json=_trace_payload(name="alpha"))
    cli.post("/api/v1/traces", json=_trace_payload(name="beta", error=True))
    assert cli.get("/api/v1/traces?status=error").json()["total"] == 1
    assert cli.get("/api/v1/traces?q=bet").json()["total"] == 1


def test_trace_detail(client):
    cli, _ = client
    cli.post("/api/v1/traces", json=_trace_payload())
    r = cli.get("/api/v1/traces/tr_ok")
    assert r.status_code == 200
    body = r.json()
    assert body["trace_id"] == "tr_ok"
    assert len(body["spans"]) == 3
    assert body["metadata"]["user"] == "alice"
    llm = next(s for s in body["spans"] if s["kind"] == "llm")
    assert llm["model"] == "gpt-4o-mini"
    assert llm["prompt_tokens"] == 120


def test_stats(client):
    cli, _ = client
    cli.post("/api/v1/traces", json=_trace_payload(name="ok-trace"))
    cli.post("/api/v1/traces", json=_trace_payload(name="bad-trace", error=True))
    s = cli.get("/api/v1/stats").json()
    assert s["total_traces"] == 2
    assert s["error_rate"] == 0.5
    assert s["total_tokens"] == 440
    assert any(m["model"] == "gpt-4o-mini" for m in s["top_models"])


def test_idempotent_reingest(client):
    cli, _ = client
    cli.post("/api/v1/traces", json=_trace_payload())
    cli.post("/api/v1/traces", json=_trace_payload())
    assert cli.get("/api/v1/traces").json()["total"] == 1


def test_serves_dashboard_ui(client):
    cli, _ = client
    r = cli.get("/")
    assert r.status_code == 200
    assert "AgentLens" in r.text
    assert "agent-lens" not in r.text.lower() or "agentlens" in r.text.lower()

"""API tests — run with: python -m pytest -q  (or: python tests/test_api.py)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from app import create_app  # noqa: E402
from app.voice import route, speakable  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_reports_every_backend(client):
    body = client.get("/api/health").get_json()
    assert body["status"] == "ok"
    assert set(body["backends"]) == {"llm", "embeddings", "timeseries", "documents", "relational"}
    assert body["indexed_chunks"] > 0


def test_ingest_and_read_back(client):
    res = client.post(
        "/api/metrics/ingest",
        json={"points": [{"device": "TEST-01", "metric": "temperature", "value": 42.5}]},
    )
    assert res.status_code == 201
    assert res.get_json()["written"] == 1

    rows = client.get("/api/metrics?device=TEST-01&hours=1").get_json()["points"]
    assert rows and rows[-1]["value"] == 42.5


def test_ingest_rejects_bad_payload(client):
    assert client.post("/api/metrics/ingest", json={"points": [{"device": "X"}]}).status_code == 400
    assert client.post("/api/metrics/ingest", json={}).status_code == 400


def test_metrics_validates_window(client):
    assert client.get("/api/metrics?hours=0").status_code == 400
    assert client.get("/api/metrics?hours=9999").status_code == 400


def test_ask_returns_grounded_answer_with_citations(client):
    body = client.post(
        "/api/ask", json={"question": "What is the vibration warning limit for PUMP-07?"}
    ).get_json()
    assert body["citations"], "an answer must cite its sources"
    assert body["retrieved"] > 0
    assert "3.5" in body["answer"] or "SOP-204" in str(body["citations"])


def test_ask_requires_a_question(client):
    assert client.post("/api/ask", json={}).status_code == 400
    assert client.post("/api/ask", json={"question": "x" * 600}).status_code == 400


def test_new_document_is_searchable_immediately(client):
    client.post(
        "/api/documents",
        json={
            "doc_id": "TEST-DOC",
            "title": "Test Procedure",
            "text": "The calibration token for the spectrometer is ZX-4417 and expires every 90 days.",
        },
    )
    hits = client.post("/api/search", json={"query": "spectrometer calibration token"}).get_json()["hits"]
    assert any(h["doc_id"] == "TEST-DOC" for h in hits)


def test_anomaly_detection_finds_planted_faults(client):
    body = client.get("/api/anomalies?hours=72").get_json()
    assert body["count"] > 0
    flagged = {(a["device"], a["metric"]) for a in body["anomalies"]}
    assert ("LINE-B", "temperature") in flagged or ("LINE-B", "vibration") in flagged


def test_report_matches_schema(client):
    report = client.post("/api/reports", json={"hours": 48}).get_json()
    for key in ("headline", "period_summary", "overall_status", "findings", "kpis", "meta"):
        assert key in report
    assert report["overall_status"] in {"healthy", "watch", "degraded", "critical"}
    for finding in report["findings"]:
        assert finding["severity"] in {"low", "medium", "high"}
        assert finding["evidence"]
    assert report["markdown"].startswith("#")


def test_voice_command_routes_and_answers(client):
    body = client.post("/api/voice/command", json={"transcript": "status of line b"}).get_json()
    assert body["intent"]["intent"] == "get_status"
    assert body["intent"]["device"] == "LINE-B"
    assert body["speech"]


@pytest.mark.parametrize(
    "transcript,expected",
    [
        ("generate a shift report", "generate_report"),
        ("any anomalies today", "get_anomalies"),
        ("temperature of line a", "get_metric"),
        ("what is the vibration limit for pump 7", "ask_question"),
        ("how is line a doing", "get_status"),
    ],
)
def test_intent_routing(transcript, expected):
    assert route(transcript)["intent"] == expected


def test_speakable_strips_markup():
    assert "[1]" not in speakable("Stop the line **now** [1]")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

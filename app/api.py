"""REST API.

Every endpoint returns JSON, validates its input, and reports which backend
and which model produced the answer — so the response is auditable, not magic.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from .config import Config
from .llm import LLMError, extractive_answer
from .rag import build_context
from .reports import build_report, to_markdown
from .timeseries import Point
from .voice import route as route_intent, speakable

api_bp = Blueprint("api", __name__)

ANSWER_SYSTEM = """You are an operations assistant for a production plant.
Answer only from the numbered CONTEXT passages and the LIVE DATA provided.
Cite every claim with its passage marker, like [1] or [2].
If the context does not contain the answer, say so plainly instead of guessing.
Keep the answer under 120 words and lead with the direct answer."""


def services():
    return current_app.extensions["services"]


def _bad(message: str, code: int = 400):
    return jsonify({"error": message}), code


def _limits() -> dict[tuple[str, str], dict]:
    """Engineering limits from the relational store, keyed for the detector."""
    return {
        (d["device"], t["metric"]): t
        for d in services().sql.devices()
        for t in d.get("thresholds", [])
    }


# --------------------------------------------------------------------------
# health & metadata
# --------------------------------------------------------------------------

@api_bp.get("/health")
def health():
    s = services()
    return jsonify(
        {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "backends": Config.describe(),
            "indexed_chunks": len(s.index.chunks),
            "documents": s.docs.count("documents"),
            "devices": s.ts.devices(),
            "metrics": s.ts.metric_names(),
        }
    )


@api_bp.get("/devices")
def devices():
    return jsonify({"devices": services().sql.devices()})


# --------------------------------------------------------------------------
# time-series
# --------------------------------------------------------------------------

@api_bp.post("/metrics/ingest")
def ingest():
    payload = request.get_json(silent=True) or {}
    raw = payload.get("points") or ([payload] if payload.get("device") else [])
    if not raw:
        return _bad("expected a 'points' array, or a single point object")

    points = []
    for i, item in enumerate(raw):
        try:
            ts = item.get("ts")
            ts = (
                datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if ts
                else datetime.now(timezone.utc)
            )
            points.append(
                Point(
                    device=str(item["device"]),
                    metric=str(item["metric"]),
                    value=float(item["value"]),
                    ts=ts,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _bad(f"invalid point at index {i}: {exc}")

    written = services().ts.write(points)
    return jsonify({"written": written, "backend": services().ts.backend}), 201


@api_bp.get("/metrics")
def metrics():
    hours = request.args.get("hours", default=24, type=int)
    if not 1 <= hours <= 720:
        return _bad("hours must be between 1 and 720")
    rows = services().ts.query(
        device=request.args.get("device"),
        metric=request.args.get("metric"),
        hours=hours,
        limit=request.args.get("limit", default=5000, type=int),
    )
    return jsonify({"count": len(rows), "hours": hours, "points": rows})


@api_bp.get("/metrics/summary")
def metrics_summary():
    hours = request.args.get("hours", default=24, type=int)
    return jsonify({"hours": hours, "summary": services().ts.summary(hours=hours)})


@api_bp.get("/anomalies")
def anomalies():
    hours = request.args.get("hours", default=24, type=int)
    threshold = request.args.get("threshold", type=float)
    found = services().ts.detect_anomalies(hours=hours, threshold=threshold, limits=_limits())
    return jsonify(
        {
            "hours": hours,
            "threshold": threshold or Config.ZSCORE_THRESHOLD,
            "count": len(found),
            "anomalies": found,
        }
    )


# --------------------------------------------------------------------------
# documents & RAG
# --------------------------------------------------------------------------

@api_bp.get("/documents")
def list_documents():
    docs = services().docs.list("documents", limit=200)
    return jsonify(
        {
            "count": len(docs),
            "documents": [
                {"doc_id": d["doc_id"], "title": d["title"], "chars": len(d["text"])} for d in docs
            ],
        }
    )


@api_bp.post("/documents")
def add_document():
    payload = request.get_json(silent=True) or {}
    for field in ("doc_id", "title", "text"):
        if not str(payload.get(field, "")).strip():
            return _bad(f"'{field}' is required")

    s = services()
    s.docs.insert(
        "documents",
        {
            "doc_id": str(payload["doc_id"]).strip(),
            "title": str(payload["title"]).strip(),
            "text": str(payload["text"]).strip(),
            "source": "api",
        },
    )
    chunks = s.load_index()  # re-index so the new document is searchable now
    return jsonify({"indexed_chunks": chunks, "documents": s.docs.count("documents")}), 201


@api_bp.post("/search")
def search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return _bad("'query' is required")
    hits = services().index.search(query, top_k=int(payload.get("top_k") or Config.TOP_K))
    return jsonify({"query": query, "hits": hits})


@api_bp.post("/ask")
def ask():
    """RAG endpoint: retrieve, ground, answer, cite."""
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return _bad("'question' is required")
    if len(question) > 500:
        return _bad("question too long (max 500 characters)")

    s = services()
    started = time.perf_counter()

    hits = s.index.search(question, top_k=int(payload.get("top_k") or Config.TOP_K))
    context, citations = build_context(hits)

    # Live numbers go in alongside the documents, so the model can answer
    # "is LINE-B running hot right now?" and not only "what does the SOP say?".
    live = ""
    if payload.get("include_live", True):
        summary = s.ts.summary(hours=int(payload.get("hours") or 24))
        if summary:
            live = "\n".join(
                f"{r['device']} {r['metric']}: last {r['last']}, avg {r['avg']}, trend {r['trend_label']}"
                for r in summary
            )

    user = f"CONTEXT\n{context or '(no matching documents)'}\n\nLIVE DATA\n{live or '(not requested)'}\n\nQUESTION\n{question}"

    mode = s.llm.mode
    try:
        answer = s.llm.chat(ANSWER_SYSTEM, user) if s.llm.enabled else extractive_answer(question, hits)
    except LLMError as exc:
        answer = extractive_answer(question, hits)
        mode = f"fallback ({str(exc)[:80]})"

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    result = {
        "question": question,
        "answer": answer.strip(),
        "citations": citations,
        "retrieved": len(hits),
        "mode": mode,
        "model": s.llm.model if s.llm.enabled else None,
        "latency_ms": elapsed_ms,
    }
    s.docs.insert("qa_history", {**result, "answer_chars": len(answer)})
    return jsonify(result)


@api_bp.get("/history")
def history():
    return jsonify({"history": services().docs.list("qa_history", limit=25)})


# --------------------------------------------------------------------------
# voice
# --------------------------------------------------------------------------

@api_bp.post("/voice/command")
def voice_command():
    """Transcript in, action taken, spoken answer out."""
    payload = request.get_json(silent=True) or {}
    transcript = str(payload.get("transcript", "")).strip()
    if not transcript:
        return _bad("'transcript' is required")

    s = services()
    intent = route_intent(transcript)
    data: dict = {}

    if intent["intent"] == "get_anomalies":
        found = s.ts.detect_anomalies(hours=intent["hours"], limits=_limits())
        data = {"anomalies": found[:5], "count": len(found)}
        if found:
            top = found[0]
            answer = (
                f"I found {len(found)} anomalies in the last {intent['hours']} hours. "
                f"The most significant is {top['device']} {top['metric']} at {top['value']}, "
                f"against a baseline of {top['baseline_mean']}."
            )
        else:
            answer = f"No anomalies in the last {intent['hours']} hours. Everything is within baseline."

    elif intent["intent"] == "get_metric" or intent["intent"] == "get_status":
        summary = [
            r
            for r in s.ts.summary(hours=intent["hours"])
            if (not intent["device"] or r["device"] == intent["device"])
            and (not intent["metric"] or r["metric"] == intent["metric"])
        ]
        data = {"summary": summary}
        if summary:
            answer = " ".join(
                f"{r['device']} {r['metric']} is at {r['last']}, averaging {r['avg']} and {r['trend_label']}."
                for r in summary[:3]
            )
        else:
            answer = "I don't have data for that device or metric."

    elif intent["intent"] == "generate_report":
        report = _generate_report(intent["hours"])
        data = {"report": report}
        answer = f"{report['headline']}. Overall status: {report['overall_status']}. {report['period_summary']}"

    else:  # ask_question -> RAG
        hits = s.index.search(transcript)
        context, citations = build_context(hits)
        user = f"CONTEXT\n{context}\n\nQUESTION\n{transcript}"
        try:
            answer = s.llm.chat(ANSWER_SYSTEM, user) if s.llm.enabled else extractive_answer(transcript, hits)
        except LLMError:
            answer = extractive_answer(transcript, hits)
        data = {"citations": citations}

    response = {
        "transcript": transcript,
        "intent": intent,
        "answer": answer.strip(),
        "speech": speakable(answer),
        **data,
    }
    s.docs.insert("qa_history", {"question": transcript, "answer": answer, "via": "voice"})
    return jsonify(response)


@api_bp.post("/voice/transcribe")
def voice_transcribe():
    """Server-side speech-to-text hook.

    The deployed demo uses the browser's Web Speech API, which needs no model
    download. To run STT on the server instead, install faster-whisper and
    implement the transcription here — /api/voice/command is unchanged.
    """
    return (
        jsonify(
            {
                "error": "server-side STT not enabled in this deployment",
                "hint": "the browser performs speech-to-text and posts the transcript to /api/voice/command",
            }
        ),
        501,
    )


# --------------------------------------------------------------------------
# AI reports
# --------------------------------------------------------------------------

def _generate_report(hours: int, audience: str = "production manager") -> dict:
    s = services()
    summary = s.ts.summary(hours=hours)
    found = s.ts.detect_anomalies(hours=hours, limits=_limits())

    # Pull the procedures relevant to what actually went wrong, so the report
    # can name a likely cause instead of only describing the number.
    kb_context = ""
    if found:
        query = " ".join(f"{a['device']} {a['metric']}" for a in found[:4])
        hits = s.index.search(query, top_k=4)
        kb_context, _ = build_context(hits, max_chars=3000)

    report = build_report(s.llm, hours, summary, found, kb_context, audience)
    report["markdown"] = to_markdown(report)
    s.docs.insert("reports", {"hours": hours, "report": report})
    return report


@api_bp.post("/reports")
def create_report():
    payload = request.get_json(silent=True) or {}
    hours = int(payload.get("hours") or 24)
    if not 1 <= hours <= 720:
        return _bad("hours must be between 1 and 720")
    audience = str(payload.get("audience") or "production manager")
    return jsonify(_generate_report(hours, audience)), 201


@api_bp.get("/reports")
def list_reports():
    return jsonify({"reports": services().docs.list("reports", limit=10)})

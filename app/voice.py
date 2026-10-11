"""Voice command layer.

The browser does speech-to-text and text-to-speech (Web Speech API); this
module turns the resulting transcript into an intent plus arguments, so a
spoken command can trigger a real API action instead of only a chat reply.

Swapping in server-side STT (Whisper / faster-whisper) means replacing the
transcript source only — the routing below is unchanged. See /api/voice/transcribe.
"""

from __future__ import annotations

import re

INTENTS = {
    "get_status": [
        r"\b(status|health|how is|how's|what'?s (?:going on|happening) (?:with|on)|check)\b",
    ],
    "get_anomalies": [r"\b(anomal|unusual|abnormal|weird|spike|problem|issue|alarm)\w*\b"],
    "generate_report": [r"\b(report|summary|summarise|summarize|brief|shift report)\b"],
    "get_metric": [r"\b(temperature|vibration|pressure|throughput)\b"],
}

DEVICE_PATTERNS = {
    "LINE-A": [r"\bline\s*a\b", r"\bline\s*(?:one|1)\b", r"\bbottling\s*a\b"],
    "LINE-B": [r"\bline\s*b\b", r"\bline\s*(?:two|2)\b", r"\bbottling\s*b\b"],
    "PUMP-07": [r"\bpump\s*(?:07|7|seven)?\b"],
}

METRICS = ["temperature", "vibration", "pressure", "throughput"]

_HOURS = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s*(hour|hr|h|day)s?\b")

# "What is the vibration limit for PUMP-07?" names a device and a metric but is
# a question about the procedure, not a request for the current reading. These
# markers send it to the RAG pipeline instead of the telemetry lookup.
_KNOWLEDGE = re.compile(
    r"\b(limit|threshold|procedure|sop|policy|rule|why|cause|reason|should (?:i|we|the)|"
    r"how (?:do|does|long|often|should)|what to do|interval|replace|maintenance|mean[s]?\b|"
    r"allowed|permitted|recommend)\b"
)


def route(transcript: str) -> dict:
    """Return {intent, device, metric, hours, confidence} for a spoken command."""
    text = (transcript or "").lower().strip()
    if not text:
        return {"intent": "unknown", "device": None, "metric": None, "hours": 24, "confidence": 0.0}

    device = next(
        (d for d, pats in DEVICE_PATTERNS.items() if any(re.search(p, text) for p in pats)), None
    )
    metric = next((m for m in METRICS if m in text), None)

    hours = 24
    match = _HOURS.search(text)
    if match:
        n = int(match.group(1))
        hours = min(n * 24, 720) if match.group(2).startswith("day") else min(n, 720)

    scores: dict[str, int] = {}
    for intent, patterns in INTENTS.items():
        scores[intent] = sum(1 for p in patterns if re.search(p, text))

    knowledge_question = bool(_KNOWLEDGE.search(text))

    if scores.get("generate_report"):
        intent = "generate_report"
    elif scores.get("get_anomalies") and not knowledge_question:
        intent = "get_anomalies"
    elif knowledge_question:
        intent = "ask_question"
    elif metric and (device or scores.get("get_metric")):
        intent = "get_metric"
    elif scores.get("get_status") and device:
        intent = "get_status"
    else:
        intent = "ask_question"  # falls through to the RAG pipeline

    matched = sum(1 for v in scores.values() if v) + bool(device) + bool(metric)
    confidence = min(0.4 + 0.2 * matched, 0.98) if intent != "ask_question" else 0.35

    return {
        "intent": intent,
        "device": device,
        "metric": metric,
        "hours": hours,
        "confidence": round(confidence, 2),
    }


def speakable(text: str, max_chars: int = 420) -> str:
    """Strip markdown and citation markers so text-to-speech sounds natural."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[*_`#>|]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: stop + 1] if stop > 120 else cut).strip()

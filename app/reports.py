"""AI-generated analytical reports.

Statistics do the measuring; the LLM does the explaining, and only inside a
strict JSON schema that the API validates before returning it. If the model is
unavailable or returns something off-schema, a deterministic generator produces
the same shape, so downstream consumers never have to handle two formats.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .llm import LLMClient, LLMError

REPORT_SCHEMA = {
    "headline": "str",
    "period_summary": "str",
    "overall_status": "one of: healthy | watch | degraded | critical",
    "findings": [
        {
            "device": "str",
            "metric": "str",
            "observation": "str (plain language, includes the number)",
            "suspected_cause": "str",
            "severity": "one of: low | medium | high",
            "recommended_action": "str",
            "evidence": "str (the statistic this is based on)",
        }
    ],
    "kpis": [{"name": "str", "value": "str", "comment": "str"}],
    "open_questions": ["str"],
}

SYSTEM = """You are an operations analyst writing a shift report for a production manager
who is not an engineer. Work only from the statistics and context you are given:
never invent a number, a device, or a cause that is not supported by the data.
Where the knowledge base explains a likely cause, use it. Where the cause is not
knowable from the data, say so and list it as an open question. Be concise and concrete."""


def _schema_hint() -> str:
    import json
    return "Match this schema exactly: " + json.dumps(REPORT_SCHEMA)


def build_report(
    llm: LLMClient,
    hours: int,
    summary: list[dict],
    anomalies: list[dict],
    kb_context: str = "",
    audience: str = "production manager",
) -> dict:
    stats_block = _format_stats(summary, anomalies)
    user = (
        f"Reporting window: last {hours} hours.\n"
        f"Audience: {audience}.\n\n"
        f"MEASURED DATA\n{stats_block}\n\n"
        f"KNOWLEDGE BASE EXTRACTS (procedures and maintenance history)\n"
        f"{kb_context or '(none retrieved)'}\n"
    )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": hours,
        "anomaly_count": len(anomalies),
        "series_count": len(summary),
    }

    if llm.enabled:
        try:
            data = llm.chat_json(SYSTEM, user, _schema_hint())
            report = _coerce(data)
            report["meta"] = {**meta, "generator": "llm", "model": llm.model}
            return report
        except (LLMError, ValueError, KeyError) as exc:
            meta["llm_error"] = str(exc)[:200]

    report = _deterministic(hours, summary, anomalies)
    report["meta"] = {**meta, "generator": "rule-based-fallback"}
    return report


# ---------------- helpers ----------------

def _format_stats(summary: list[dict], anomalies: list[dict]) -> str:
    lines = ["Per-series aggregates:"]
    for s in summary:
        lines.append(
            f"- {s['device']} / {s['metric']}: avg {s['avg']}, min {s['min']}, max {s['max']}, "
            f"last {s['last']}, trend {s['trend_label']} ({s['trend']:+})"
        )
    lines.append("")
    lines.append(f"Statistical anomalies detected: {len(anomalies)}")
    for a in anomalies[:12]:
        lines.append(
            f"- [{a.get('kind', 'statistical')}] {a['device']} / {a['metric']} = {a['value']} "
            f"at {a['ts']} — {a.get('detail', '')} (severity {a['severity']})"
        )
    return "\n".join(lines)


def _coerce(data: dict) -> dict:
    """Validate the model's JSON against the schema and normalise it."""
    allowed_status = {"healthy", "watch", "degraded", "critical"}
    allowed_sev = {"low", "medium", "high"}

    findings = []
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        sev = str(f.get("severity", "medium")).lower()
        findings.append(
            {
                "device": str(f.get("device", "unknown")),
                "metric": str(f.get("metric", "unknown")),
                "observation": str(f.get("observation", "")).strip(),
                "suspected_cause": str(f.get("suspected_cause", "not established")).strip(),
                "severity": sev if sev in allowed_sev else "medium",
                "recommended_action": str(f.get("recommended_action", "")).strip(),
                "evidence": str(f.get("evidence", "")).strip(),
            }
        )

    status = str(data.get("overall_status", "watch")).lower()
    return {
        "headline": str(data.get("headline", "Operations report")).strip(),
        "period_summary": str(data.get("period_summary", "")).strip(),
        "overall_status": status if status in allowed_status else "watch",
        "findings": findings,
        "kpis": [
            {
                "name": str(k.get("name", "")),
                "value": str(k.get("value", "")),
                "comment": str(k.get("comment", "")),
            }
            for k in (data.get("kpis") or [])
            if isinstance(k, dict)
        ],
        "open_questions": [str(q) for q in (data.get("open_questions") or [])],
    }


def _deterministic(hours: int, summary: list[dict], anomalies: list[dict]) -> dict:
    """Same schema, no model. Used when no API key is set."""
    findings = []
    for a in anomalies[:6]:
        direction = "above" if a["zscore"] > 0 else "below"
        cause = (
            "value is outside the documented operating limit — check the matching SOP"
            if a.get("kind") == "threshold"
            else "not established from data alone — see linked procedure"
        )
        findings.append(
            {
                "device": a["device"],
                "metric": a["metric"],
                "observation": (
                    f"{a['metric'].title()} reached {a['value']}, well {direction} its recent "
                    f"baseline of {a['baseline_mean']}."
                ),
                "suspected_cause": cause,
                "severity": a["severity"],
                "recommended_action": "Inspect per the relevant SOP and record the cause in the shift log.",
                "evidence": a.get("detail") or f"z-score {a['zscore']} vs trailing baseline",
            }
        )

    worst = max((abs(a["zscore"]) for a in anomalies), default=0)
    status = "critical" if worst >= 5 else "degraded" if worst >= 3.5 else "watch" if anomalies else "healthy"

    return {
        "headline": (
            f"{len(anomalies)} anomal{'y' if len(anomalies) == 1 else 'ies'} across "
            f"{len(summary)} monitored series in the last {hours}h"
        ),
        "period_summary": (
            f"Monitored {len(summary)} device/metric series over {hours} hours. "
            f"Statistical screening flagged {len(anomalies)} deviation(s) from the trailing baseline."
        ),
        "overall_status": status,
        "findings": findings,
        "kpis": [
            {
                "name": f"{s['device']} {s['metric']} (avg)",
                "value": str(s["avg"]),
                "comment": f"trend {s['trend_label']}",
            }
            for s in summary
        ],
        "open_questions": [
            f"Confirm the cause of the {a['device']} {a['metric']} deviation at {a['ts']}."
            for a in anomalies[:3]
        ],
    }


def to_markdown(report: dict) -> str:
    """Render the structured report for humans — email, Slack, or print."""
    out = [f"# {report['headline']}", "", f"**Status:** {report['overall_status'].upper()}", ""]
    if report.get("period_summary"):
        out += [report["period_summary"], ""]
    if report.get("findings"):
        out += ["## Findings", ""]
        for i, f in enumerate(report["findings"], 1):
            out += [
                f"**{i}. {f['device']} — {f['metric']}** ({f['severity']})",
                f"- Observation: {f['observation']}",
                f"- Suspected cause: {f['suspected_cause']}",
                f"- Action: {f['recommended_action']}",
                f"- Evidence: {f['evidence']}",
                "",
            ]
    if report.get("kpis"):
        out += ["## KPIs", "", "| Metric | Value | Comment |", "|---|---|---|"]
        out += [f"| {k['name']} | {k['value']} | {k['comment']} |" for k in report["kpis"]]
        out += [""]
    if report.get("open_questions"):
        out += ["## Open questions", ""] + [f"- {q}" for q in report["open_questions"]]
    return "\n".join(out)

/* OpsMind dashboard — talks to the Flask API, handles voice in the browser. */

import { drawChart } from "./chart.js";

const $ = (id) => document.getElementById(id);
const api = async (path, options = {}) => {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({ error: "bad response" }));
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
};

let health = null;
let limits = {};   // "DEVICE|metric" -> {warn_high, crit_high, unit}

/* ------------------------------------------------------------------ boot */

async function boot() {
  try {
    health = await api("/api/health");
  } catch (e) {
    $("backends").innerHTML = `<span class="badge">API unreachable</span>`;
    return;
  }

  const b = health.backends;
  $("backends").innerHTML = [
    ["time-series", b.timeseries],
    ["documents", b.documents],
    ["relational", b.relational],
    ["retrieval", b.embeddings],
    ["llm", b.llm],
  ]
    .map(
      ([k, v]) =>
        `<span class="badge ${v.includes("fallback") || v.includes("local") ? "" : "live"}">${k} <b>${v}</b></span>`
    )
    .join("");

  $("footnote").textContent =
    `${health.indexed_chunks} indexed chunks from ${health.documents} documents · ` +
    `${health.devices.length} devices`;

  $("device").innerHTML = health.devices.map((d) => `<option>${d}</option>`).join("");
  $("metric").innerHTML = health.metrics.map((m) => `<option>${m}</option>`).join("");
  if (health.devices.includes("LINE-B")) $("device").value = "LINE-B";
  if (health.metrics.includes("temperature")) $("metric").value = "temperature";

  try {
    const reg = await api("/api/devices");
    reg.devices.forEach((d) =>
      (d.thresholds || []).forEach((t) => (limits[`${d.device}|${t.metric}`] = t))
    );
  } catch { /* thresholds are optional decoration on the chart */ }

  await Promise.all([loadChart(), loadAnomalies()]);
}

/* --------------------------------------------------------------- metrics */

async function loadChart() {
  const device = $("device").value;
  const metric = $("metric").value;
  const hours = $("hours").value;
  if (!device || !metric) return;

  const data = await api(`/api/metrics?device=${device}&metric=${metric}&hours=${hours}`);
  const limit = limits[`${device}|${metric}`] || {};

  drawChart($("chart"), data.points, {
    warn: limit.warn_high,
    crit: limit.crit_high,
    unit: limit.unit && limit.unit !== "units/min" ? ` ${limit.unit}` : "",
  });

  const summary = await api(`/api/metrics/summary?hours=${hours}`);
  const row = summary.summary.find((s) => s.device === device && s.metric === metric);
  $("stats").innerHTML = row
    ? [
        ["last", row.last],
        ["avg", row.avg],
        ["min", row.min],
        ["max", row.max],
        ["σ", row.stdev],
        ["trend", `${row.trend > 0 ? "+" : ""}${row.trend}`],
      ]
        .map(
          ([k, v], i) =>
            `<div class="stat"><div class="k">${k}</div><div class="v ${
              i === 5 ? (row.trend > 0 ? "up" : "down") : ""
            }">${v}</div></div>`
        )
        .join("")
    : "";
}

async function loadAnomalies() {
  const hours = $("hours").value;
  const data = await api(`/api/anomalies?hours=${hours}`);
  $("anomalies").innerHTML = data.anomalies.length
    ? data.anomalies
        .slice(0, 12)
        .map(
          (a) => `
      <div class="anom ${a.severity}">
        <div class="t">
          <span>${a.device} · ${a.metric}</span>
          <span class="z">${a.kind === "threshold" ? "limit" : "z " + a.zscore}</span>
        </div>
        <div class="m">
          <span class="kind">${a.kind}</span>${escapeHtml(a.detail || "")}
        </div>
        <div class="m">${new Date(a.ts).toLocaleString()}</div>
      </div>`
        )
        .join("")
    : `<div class="empty">No anomalies in the last ${hours}h.</div>`;
}

/* ------------------------------------------------------------------- ask */

async function ask(question) {
  if (!question.trim()) return;
  $("answer").classList.remove("hidden");
  $("answer").innerHTML = `<div class="text">Thinking…</div>`;
  $("askBtn").disabled = true;

  try {
    const r = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question, hours: Number($("hours").value) }),
    });
    renderAnswer(r);
    return r;
  } catch (e) {
    $("answer").innerHTML = `<div class="text">Error: ${e.message}</div>`;
  } finally {
    $("askBtn").disabled = false;
  }
}

function renderAnswer(r) {
  const cites = (r.citations || [])
    .map((c) => `<div class="cite"><b>${c.marker}</b> ${c.title} <span>· score ${c.score}</span></div>`)
    .join("");
  $("answer").innerHTML = `
    <div class="text">${escapeHtml(r.answer)}</div>
    ${cites ? `<div class="cites">${cites}</div>` : ""}
    <div class="meta">
      <span>mode: ${r.mode}</span>
      <span>retrieved: ${r.retrieved} chunks</span>
      <span>latency: ${r.latency_ms} ms</span>
      ${r.model ? `<span>model: ${r.model}</span>` : ""}
    </div>`;
}

/* ----------------------------------------------------------------- voice */

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

function setupVoice() {
  if (!SpeechRecognition) {
    $("mic").title = "Speech recognition not supported in this browser — try Chrome";
    $("mic").disabled = true;
    return;
  }
  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = async (event) => {
    const transcript = event.results[0][0].transcript;
    $("question").value = transcript;
    $("answer").classList.remove("hidden");
    $("answer").innerHTML = `<div class="text">Heard: “${escapeHtml(transcript)}” — routing…</div>`;

    const r = await api("/api/voice/command", {
      method: "POST",
      body: JSON.stringify({ transcript }),
    });

    $("answer").innerHTML = `
      <div class="text">${escapeHtml(r.answer)}</div>
      <div class="meta">
        <span>intent: ${r.intent.intent}</span>
        <span>device: ${r.intent.device || "—"}</span>
        <span>metric: ${r.intent.metric || "—"}</span>
        <span>confidence: ${r.intent.confidence}</span>
      </div>`;

    speak(r.speech || r.answer);
    if (r.report) renderReport(r.report);
    if (r.anomalies) loadAnomalies();
  };

  recognition.onend = () => $("mic").classList.remove("recording");
  recognition.onerror = () => $("mic").classList.remove("recording");

  $("mic").onclick = () => {
    if ($("mic").classList.contains("recording")) {
      recognition.stop();
      return;
    }
    $("mic").classList.add("recording");
    recognition.start();
  };
}

function speak(text) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.02;
  window.speechSynthesis.speak(utterance);
}

/* ---------------------------------------------------------------- report */

async function generateReport() {
  $("reportBtn").disabled = true;
  $("report").innerHTML = `<div class="empty">Analysing telemetry and drafting…</div>`;
  try {
    const report = await api("/api/reports", {
      method: "POST",
      body: JSON.stringify({ hours: Number($("reportHours").value) }),
    });
    renderReport(report);
  } catch (e) {
    $("report").innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  } finally {
    $("reportBtn").disabled = false;
  }
}

function renderReport(r) {
  const findings = (r.findings || [])
    .map(
      (f) => `
      <div class="finding">
        <span class="sev ${f.severity}">${f.severity}</span>
        <h4>${f.device} · ${f.metric}</h4>
        <p>${escapeHtml(f.observation)}</p>
        <p><span class="lbl">Cause:</span> ${escapeHtml(f.suspected_cause)}</p>
        <p><span class="lbl">Action:</span> ${escapeHtml(f.recommended_action)}</p>
        <p><span class="lbl">Evidence:</span> ${escapeHtml(f.evidence)}</p>
      </div>`
    )
    .join("");

  const kpis = (r.kpis || [])
    .map((k) => `<tr><td>${escapeHtml(k.name)}</td><td>${escapeHtml(k.value)}</td><td>${escapeHtml(k.comment)}</td></tr>`)
    .join("");

  $("report").innerHTML = `
    <span class="status ${r.overall_status}">${r.overall_status}</span>
    <h3>${escapeHtml(r.headline)}</h3>
    <p>${escapeHtml(r.period_summary)}</p>
    ${findings ? `<div class="findings">${findings}</div>` : ""}
    ${kpis ? `<table class="kpis"><thead><tr><th>KPI</th><th>Value</th><th>Comment</th></tr></thead><tbody>${kpis}</tbody></table>` : ""}
    ${
      (r.open_questions || []).length
        ? `<div class="openq"><b>Open questions:</b><ul>${r.open_questions
            .map((q) => `<li>${escapeHtml(q)}</li>`)
            .join("")}</ul></div>`
        : ""
    }
    <div class="meta" style="margin-top:14px;color:#8b9bb0;font-size:11.5px">
      generated by ${r.meta?.generator || "?"}${r.meta?.model ? ` (${r.meta.model})` : ""} ·
      ${r.meta?.anomaly_count ?? 0} anomalies · ${r.meta?.series_count ?? 0} series
    </div>`;
}

/* ----------------------------------------------------------------- wire */

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

$("askBtn").onclick = () => ask($("question").value);
$("question").onkeydown = (e) => { if (e.key === "Enter") ask($("question").value); };
$("reportBtn").onclick = generateReport;
["device", "metric", "hours"].forEach((id) => {
  $(id).onchange = () => { loadChart(); loadAnomalies(); };
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.onclick = () => { $("question").value = chip.textContent; ask(chip.textContent); };
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(loadChart, 180);
});

setupVoice();
boot();

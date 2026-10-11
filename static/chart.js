/* Minimal canvas line chart — no external library, so the dashboard has no
   CDN dependency and renders identically offline. Handles HiDPI, axes,
   warning/critical limit lines and hover read-out. */

export function drawChart(canvas, points, options = {}) {
  const { warn = null, crit = null, unit = "" } = options;
  const dpr = window.devicePixelRatio || 1;
  const box = canvas.parentElement.getBoundingClientRect();
  const W = Math.max(320, box.width);
  const H = Math.max(180, box.height);

  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  if (!points.length) {
    ctx.fillStyle = "#8b9bb0";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText("No data in this window.", 14, H / 2);
    return () => {};
  }

  const pad = { top: 14, right: 14, bottom: 26, left: 46 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const values = points.map((p) => p.value);
  const candidates = [...values, ...(warn ? [warn] : []), ...(crit ? [crit] : [])];
  let lo = Math.min(...candidates);
  let hi = Math.max(...candidates);
  const span = hi - lo || 1;
  lo -= span * 0.12;
  hi += span * 0.12;

  const x = (i) => pad.left + (i / Math.max(1, points.length - 1)) * plotW;
  const y = (v) => pad.top + plotH - ((v - lo) / (hi - lo)) * plotH;

  // grid + y labels
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = lo + ((hi - lo) * i) / 4;
    const py = Math.round(y(value)) + 0.5;
    ctx.strokeStyle = "#1f2733";
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(W - pad.right, py);
    ctx.stroke();
    ctx.fillStyle = "#8b9bb0";
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(1), pad.left - 8, py);
  }

  // x labels
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const ticks = Math.min(6, points.length);
  for (let t = 0; t < ticks; t++) {
    const i = Math.round((t / (ticks - 1 || 1)) * (points.length - 1));
    const label = new Date(points[i].ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    ctx.fillStyle = "#8b9bb0";
    ctx.fillText(label, x(i), H - pad.bottom + 7);
  }

  // limit lines
  const limitLine = (value, color, label) => {
    if (!value || value <= 0) return;
    const py = Math.round(y(value)) + 0.5;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(W - pad.right, py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, pad.left + 6, py - 3);
    ctx.restore();
  };
  limitLine(warn, "#d9a441", `warning ${warn}${unit}`);
  limitLine(crit, "#f06a6a", `critical ${crit}${unit}`);

  // area + line
  const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
  gradient.addColorStop(0, "rgba(77,163,255,.28)");
  gradient.addColorStop(1, "rgba(77,163,255,0)");

  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.value)) : ctx.moveTo(x(i), y(p.value))));
  ctx.lineTo(x(points.length - 1), pad.top + plotH);
  ctx.lineTo(x(0), pad.top + plotH);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(x(i), y(p.value)) : ctx.moveTo(x(i), y(p.value))));
  ctx.strokeStyle = "#4da3ff";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // hover read-out
  const overlay = canvas.nextElementSibling;
  const onMove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const i = Math.round(((mx - pad.left) / plotW) * (points.length - 1));
    if (i < 0 || i >= points.length || !overlay) {
      if (overlay) overlay.style.display = "none";
      return;
    }
    const p = points[i];
    overlay.style.display = "block";
    overlay.style.left = Math.min(Math.max(x(i), 60), W - 60) + "px";
    overlay.style.top = Math.max(y(p.value) - 42, 0) + "px";
    overlay.innerHTML =
      `<b>${p.value}${unit}</b><br>${new Date(p.ts).toLocaleString([], {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      })}`;
  };
  const onLeave = () => { if (overlay) overlay.style.display = "none"; };
  canvas.onmousemove = onMove;
  canvas.onmouseleave = onLeave;

  return () => { canvas.onmousemove = null; canvas.onmouseleave = null; };
}

"""Time-series layer.

Writes and queries operational metrics. Uses InfluxDB when INFLUX_URL is set,
otherwise an SQLite table with the same interface, so the demo runs anywhere.
Also owns the statistical anomaly detection that feeds the AI reports.
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .config import Config


@dataclass
class Point:
    device: str
    metric: str
    value: float
    ts: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.astimezone(timezone.utc).isoformat()
        return d


@dataclass
class Anomaly:
    device: str
    metric: str
    value: float
    ts: str
    zscore: float
    baseline_mean: float
    severity: str
    kind: str = "statistical"      # statistical | threshold
    detail: str = ""               # human-readable reason

    def to_dict(self) -> dict:
        return asdict(self)


class TimeSeriesStore:
    """One interface, two implementations."""

    def __init__(self) -> None:
        self.backend = "influxdb" if Config.INFLUX_URL else "sqlite"
        self._influx = None
        if self.backend == "influxdb":
            try:
                from influxdb_client import InfluxDBClient  # type: ignore
                self._influx = InfluxDBClient(
                    url=Config.INFLUX_URL, token=Config.INFLUX_TOKEN, org=Config.INFLUX_ORG
                )
            except Exception:
                # Never let a missing driver take the API down — degrade to SQLite.
                self.backend = "sqlite"
        if self.backend == "sqlite":
            self._init_sqlite()

    # ---------------- SQLite fallback ----------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(Config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS metrics (
                       id      INTEGER PRIMARY KEY AUTOINCREMENT,
                       device  TEXT NOT NULL,
                       metric  TEXT NOT NULL,
                       value   REAL NOT NULL,
                       ts      TEXT NOT NULL
                   )"""
            )
            # The index that makes the range query fast — the same shape of
            # decision you make when choosing an InfluxDB tag set.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics(device, metric, ts)"
            )

    # ---------------- writes ----------------

    def write(self, points: Iterable[Point]) -> int:
        points = list(points)
        if not points:
            return 0
        if self.backend == "influxdb":
            from influxdb_client import Point as IPoint  # type: ignore
            from influxdb_client.client.write_api import SYNCHRONOUS  # type: ignore

            writer = self._influx.write_api(write_options=SYNCHRONOUS)
            writer.write(
                bucket=Config.INFLUX_BUCKET,
                org=Config.INFLUX_ORG,
                record=[
                    IPoint("operations")
                    .tag("device", p.device)
                    .field(p.metric, float(p.value))
                    .time(p.ts)
                    for p in points
                ],
            )
            return len(points)

        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO metrics (device, metric, value, ts) VALUES (?,?,?,?)",
                [
                    (p.device, p.metric, float(p.value), p.ts.astimezone(timezone.utc).isoformat())
                    for p in points
                ],
            )
        return len(points)

    # ---------------- reads ----------------

    def query(
        self, device: str | None = None, metric: str | None = None, hours: int = 24, limit: int = 5000
    ) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        if self.backend == "influxdb":
            flux = f'''
                from(bucket: "{Config.INFLUX_BUCKET}")
                  |> range(start: -{hours}h)
                  |> filter(fn: (r) => r._measurement == "operations")
            '''
            if device:
                flux += f'  |> filter(fn: (r) => r.device == "{device}")\n'
            if metric:
                flux += f'  |> filter(fn: (r) => r._field == "{metric}")\n'
            flux += f"  |> limit(n: {limit})"
            tables = self._influx.query_api().query(flux, org=Config.INFLUX_ORG)
            return [
                {
                    "device": rec.values.get("device"),
                    "metric": rec.get_field(),
                    "value": rec.get_value(),
                    "ts": rec.get_time().isoformat(),
                }
                for table in tables
                for rec in table.records
            ]

        sql = "SELECT device, metric, value, ts FROM metrics WHERE ts >= ?"
        params: list = [since.isoformat()]
        if device:
            sql += " AND device = ?"
            params.append(device)
        if metric:
            sql += " AND metric = ?"
            params.append(metric)
        sql += " ORDER BY ts ASC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def devices(self) -> list[str]:
        if self.backend == "influxdb":
            rows = self.query(hours=24 * 30, limit=20000)
            return sorted({r["device"] for r in rows if r["device"]})
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT device FROM metrics ORDER BY device")]

    def metric_names(self) -> list[str]:
        if self.backend == "influxdb":
            rows = self.query(hours=24 * 30, limit=20000)
            return sorted({r["metric"] for r in rows if r["metric"]})
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT metric FROM metrics ORDER BY metric")]

    # ---------------- analytics ----------------

    def summary(self, hours: int = 24) -> list[dict]:
        """Per device+metric aggregates — the numeric spine of every AI report."""
        rows = self.query(hours=hours, limit=200000)
        buckets: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            buckets.setdefault((r["device"], r["metric"]), []).append(float(r["value"]))

        out = []
        for (device, metric), values in sorted(buckets.items()):
            if not values:
                continue
            # Trend = second half mean minus first half mean. A series with a
            # single sample has no trend rather than a divide-by-zero.
            half = len(values) // 2
            if half and len(values) - half:
                first, second = values[:half], values[half:]
                trend = (sum(second) / len(second)) - (sum(first) / len(first))
            else:
                trend = 0.0
            out.append(
                {
                    "device": device,
                    "metric": metric,
                    "count": len(values),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "avg": round(sum(values) / len(values), 2),
                    "stdev": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
                    "last": round(values[-1], 2),
                    "trend": round(trend, 2),
                    "trend_label": "rising" if trend > 0.5 else "falling" if trend < -0.5 else "stable",
                }
            )
        return out

    def detect_anomalies(
        self,
        hours: int = 24,
        threshold: float | None = None,
        limits: dict[tuple[str, str], dict] | None = None,
    ) -> list[dict]:
        """Two detectors, because one is not enough.

        1. Rolling z-score catches sudden spikes against a trailing baseline.
        2. Engineering limits (from the SQL thresholds table) catch the slow
           drifts a rolling baseline hides — the baseline moves with the drift,
           so a line can climb 16 degrees without ever scoring a spike.

        Deliberately explainable: the LLM interprets anomalies, it never finds
        them. Statistics and documented limits find them.
        """
        threshold = threshold or Config.ZSCORE_THRESHOLD
        rows = self.query(hours=hours, limit=200000)
        series: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            series.setdefault((r["device"], r["metric"]), []).append(r)

        anomalies: list[Anomaly] = []
        window = 30
        for (device, metric), points in series.items():
            points.sort(key=lambda p: p["ts"])
            values = [float(p["value"]) for p in points]
            for i in range(window, len(values)):
                baseline = values[max(0, i - window) : i]
                mean = sum(baseline) / len(baseline)
                sd = statistics.pstdev(baseline)
                if sd < 1e-6:
                    continue
                z = (values[i] - mean) / sd
                if abs(z) >= threshold:
                    anomalies.append(
                        Anomaly(
                            device=device,
                            metric=metric,
                            value=round(values[i], 2),
                            ts=points[i]["ts"],
                            zscore=round(z, 2),
                            baseline_mean=round(mean, 2),
                            severity="high" if abs(z) >= threshold + 1.5 else "medium",
                            kind="statistical",
                            detail=(
                                f"{round(values[i], 2)} against a 30-sample baseline of "
                                f"{round(mean, 2)} (z {round(z, 2)})"
                            ),
                        )
                    )

        # Collapse consecutive spikes in the same series into one event.
        deduped: list[Anomaly] = []
        for a in sorted(anomalies, key=lambda x: (x.device, x.metric, x.ts)):
            prev = deduped[-1] if deduped else None
            if prev and prev.device == a.device and prev.metric == a.metric:
                gap = _parse(a.ts) - _parse(prev.ts)
                if gap < timedelta(minutes=30):
                    if abs(a.zscore) > abs(prev.zscore):
                        deduped[-1] = a
                    continue
            deduped.append(a)

        deduped += self._limit_breaches(series, limits or {})
        deduped.sort(key=lambda a: (a.severity != "high", -abs(a.zscore)))
        return [a.to_dict() for a in deduped]

    @staticmethod
    def _limit_breaches(
        series: dict[tuple[str, str], list[dict]], limits: dict[tuple[str, str], dict]
    ) -> list[Anomaly]:
        """Flag a series that is sitting above its documented warning or
        critical limit, independently of how it got there."""
        sustained = 3  # consecutive samples, so one bad reading is not an alarm
        out: list[Anomaly] = []

        for key, points in series.items():
            limit = limits.get(key)
            if not limit:
                continue
            warn, crit = limit.get("warn_high"), limit.get("crit_high")
            if not warn or warn <= 0:
                continue

            points = sorted(points, key=lambda p: p["ts"])
            tail = points[-sustained:]
            if len(tail) < sustained:
                continue
            values = [float(p["value"]) for p in tail]

            level = None
            if crit and all(v >= crit for v in values):
                level = ("high", crit, "critical")
            elif all(v >= warn for v in values):
                level = ("medium", warn, "warning")
            if not level:
                continue

            severity, bound, label = level
            unit = limit.get("unit") or ""
            baseline = sum(float(p["value"]) for p in points) / len(points)
            out.append(
                Anomaly(
                    device=key[0],
                    metric=key[1],
                    value=round(values[-1], 2),
                    ts=tail[-1]["ts"],
                    zscore=round((values[-1] - bound) / bound * 10, 2),
                    baseline_mean=round(baseline, 2),
                    severity=severity,
                    kind="threshold",
                    detail=(
                        f"{round(values[-1], 2)}{unit} has been above the {label} limit of "
                        f"{bound}{unit} for {sustained} consecutive samples"
                    ),
                )
            )
        return out


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

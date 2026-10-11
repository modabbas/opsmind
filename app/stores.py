"""Document store (MongoDB / JSON fallback) and relational store (PostgreSQL / SQLite).

Three databases, three jobs:
  * InfluxDB  -> high-volume numeric metrics  (see timeseries.py)
  * MongoDB   -> schema-less documents, Q&A history, generated reports
  * SQL       -> the relational metadata that needs joins and constraints
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import Config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentStore:
    """MongoDB when MONGO_URI is set, otherwise a JSON file with the same API."""

    def __init__(self) -> None:
        self.backend = "mongodb" if Config.MONGO_URI else "json-store"
        self._db = None
        if self.backend == "mongodb":
            try:
                from pymongo import MongoClient  # type: ignore
                self._db = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=3000)[Config.MONGO_DB]
                self._db.command("ping")
            except Exception:
                self.backend = "json-store"
        if self.backend == "json-store":
            self._path = Path(Config.SQLITE_PATH).with_name("documents.json")
            if not self._path.exists():
                self._write({"documents": [], "qa_history": [], "reports": []})

    # -- json helpers --
    def _read(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- generic collection ops --
    def insert(self, collection: str, doc: dict) -> str:
        doc = {**doc}
        doc.setdefault("_id", uuid.uuid4().hex[:12])
        doc.setdefault("created_at", _now())
        if self.backend == "mongodb":
            self._db[collection].insert_one(doc)
        else:
            data = self._read()
            data.setdefault(collection, []).append(doc)
            self._write(data)
        return doc["_id"]

    def list(self, collection: str, limit: int = 100) -> list[dict]:
        if self.backend == "mongodb":
            cursor = self._db[collection].find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
            return list(cursor)
        data = self._read()
        rows = sorted(data.get(collection, []), key=lambda d: d.get("created_at", ""), reverse=True)
        return [{k: v for k, v in r.items() if k != "_id"} for r in rows[:limit]]

    def get(self, collection: str, doc_id: str) -> dict | None:
        if self.backend == "mongodb":
            return self._db[collection].find_one({"_id": doc_id}, {"_id": 0})
        for row in self._read().get(collection, []):
            if row.get("_id") == doc_id:
                return row
        return None

    def count(self, collection: str) -> int:
        if self.backend == "mongodb":
            return self._db[collection].count_documents({})
        return len(self._read().get(collection, []))


class RelationalStore:
    """Device registry and thresholds — relational because this is where you
    want constraints, joins and a single source of truth."""

    def __init__(self) -> None:
        self.backend = "postgresql" if Config.DATABASE_URL else "sqlite"
        self._pg = None
        if self.backend == "postgresql":
            try:
                import psycopg2  # type: ignore
                self._pg = psycopg2.connect(Config.DATABASE_URL)
            except Exception:
                self.backend = "sqlite"
        self._init_schema()

    def _conn(self):
        if self.backend == "postgresql":
            return self._pg
        conn = sqlite3.connect(Config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        ddl = """
            CREATE TABLE IF NOT EXISTS devices (
                device       TEXT PRIMARY KEY,
                line         TEXT NOT NULL,
                location     TEXT NOT NULL,
                criticality  TEXT NOT NULL
            )"""
        ddl2 = """
            CREATE TABLE IF NOT EXISTS thresholds (
                device    TEXT NOT NULL,
                metric    TEXT NOT NULL,
                warn_high REAL,
                crit_high REAL,
                unit      TEXT,
                PRIMARY KEY (device, metric)
            )"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(ddl)
        cur.execute(ddl2)
        conn.commit()
        if self.backend == "sqlite":
            conn.close()

    def upsert_device(self, device: str, line: str, location: str, criticality: str) -> None:
        conn = self._conn()
        cur = conn.cursor()
        if self.backend == "postgresql":
            cur.execute(
                """INSERT INTO devices (device, line, location, criticality) VALUES (%s,%s,%s,%s)
                   ON CONFLICT (device) DO UPDATE SET line=EXCLUDED.line,
                   location=EXCLUDED.location, criticality=EXCLUDED.criticality""",
                (device, line, location, criticality),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO devices (device, line, location, criticality) VALUES (?,?,?,?)",
                (device, line, location, criticality),
            )
        conn.commit()
        if self.backend == "sqlite":
            conn.close()

    def upsert_threshold(self, device: str, metric: str, warn: float, crit: float, unit: str) -> None:
        conn = self._conn()
        cur = conn.cursor()
        if self.backend == "postgresql":
            cur.execute(
                """INSERT INTO thresholds (device, metric, warn_high, crit_high, unit)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (device, metric) DO UPDATE SET warn_high=EXCLUDED.warn_high,
                   crit_high=EXCLUDED.crit_high, unit=EXCLUDED.unit""",
                (device, metric, warn, crit, unit),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO thresholds (device, metric, warn_high, crit_high, unit) VALUES (?,?,?,?,?)",
                (device, metric, warn, crit, unit),
            )
        conn.commit()
        if self.backend == "sqlite":
            conn.close()

    def devices(self) -> list[dict]:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT d.device, d.line, d.location, d.criticality,
                      t.metric, t.warn_high, t.crit_high, t.unit
               FROM devices d LEFT JOIN thresholds t ON t.device = d.device
               ORDER BY d.device, t.metric"""
        )
        rows = cur.fetchall()
        cols = ["device", "line", "location", "criticality", "metric", "warn_high", "crit_high", "unit"]
        out: dict[str, dict] = {}
        for row in rows:
            r = dict(zip(cols, row))
            entry = out.setdefault(
                r["device"],
                {"device": r["device"], "line": r["line"], "location": r["location"],
                 "criticality": r["criticality"], "thresholds": []},
            )
            if r["metric"]:
                entry["thresholds"].append(
                    {"metric": r["metric"], "warn_high": r["warn_high"],
                     "crit_high": r["crit_high"], "unit": r["unit"]}
                )
        if self.backend == "sqlite":
            conn.close()
        return list(out.values())

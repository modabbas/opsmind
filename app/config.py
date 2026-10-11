"""Central configuration. Everything is driven by environment variables so the
same code runs on Replit (no databases, zero setup) and in production
(InfluxDB + MongoDB + PostgreSQL)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


class Config:
    # --- LLM (any OpenAI-compatible endpoint: OpenAI, Groq, Together, OpenRouter, local vLLM) ---
    LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_API_KEY = _env("LLM_API_KEY")
    LLM_MODEL = _env("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TIMEOUT = int(_env("LLM_TIMEOUT", "45"))

    # Optional embeddings endpoint. Without it the RAG layer uses a local
    # TF-IDF vectorizer, so retrieval still works offline.
    EMBEDDINGS_URL = _env("EMBEDDINGS_URL")
    EMBEDDINGS_MODEL = _env("EMBEDDINGS_MODEL", "text-embedding-3-small")

    # --- Time-series: InfluxDB, else local SQLite ---
    INFLUX_URL = _env("INFLUX_URL")
    INFLUX_TOKEN = _env("INFLUX_TOKEN")
    INFLUX_ORG = _env("INFLUX_ORG", "opsmind")
    INFLUX_BUCKET = _env("INFLUX_BUCKET", "metrics")

    # --- Documents / history: MongoDB, else local JSON store ---
    MONGO_URI = _env("MONGO_URI")
    MONGO_DB = _env("MONGO_DB", "opsmind")

    # --- Relational metadata: PostgreSQL, else local SQLite ---
    DATABASE_URL = _env("DATABASE_URL")
    SQLITE_PATH = str(DATA_DIR / "opsmind.db")

    # --- RAG tuning (values picked with eval/run_eval.py) ---
    CHUNK_SIZE = int(_env("CHUNK_SIZE", "110"))        # words per chunk
    CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "25"))   # words of overlap
    TOP_K = int(_env("TOP_K", "4"))                    # chunks sent to the LLM
    CANDIDATE_K = int(_env("CANDIDATE_K", "12"))       # chunks before re-ranking

    # --- Anomaly detection ---
    ZSCORE_THRESHOLD = float(_env("ZSCORE_THRESHOLD", "3.4"))

    @classmethod
    def describe(cls) -> dict:
        """Which backend each layer actually resolved to — shown in /api/health."""
        return {
            "llm": "remote" if cls.LLM_API_KEY else "offline-fallback",
            "embeddings": "remote" if (cls.EMBEDDINGS_URL and cls.LLM_API_KEY) else "local-tfidf",
            "timeseries": "influxdb" if cls.INFLUX_URL else "sqlite",
            "documents": "mongodb" if cls.MONGO_URI else "json-store",
            "relational": "postgresql" if cls.DATABASE_URL else "sqlite",
        }

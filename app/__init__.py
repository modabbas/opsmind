"""Application factory and shared service container."""

from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS

from .config import Config
from .llm import LLMClient
from .rag import VectorIndex
from .stores import DocumentStore, RelationalStore
from .timeseries import TimeSeriesStore


class Services:
    """Built once at startup and shared by every request."""

    def __init__(self) -> None:
        self.ts = TimeSeriesStore()
        self.docs = DocumentStore()
        self.sql = RelationalStore()
        self.index = VectorIndex()
        self.llm = LLMClient()

    def load_index(self) -> int:
        """Rebuild the vector index from whatever is in the document store."""
        self.index = VectorIndex()
        total = 0
        for doc in self.docs.list("documents", limit=1000):
            total += self.index.add_document(doc["doc_id"], doc["title"], doc["text"])
        self.index.build()
        return total


def create_app(seed_if_empty: bool = True) -> Flask:
    app = Flask(__name__, static_folder="../static", static_url_path="")
    CORS(app)
    app.config["JSON_SORT_KEYS"] = False

    services = Services()
    app.extensions["services"] = services

    if seed_if_empty:
        bootstrap(services)
    services.load_index()

    from .api import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def home():
        return app.send_static_file("index.html")

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def server_error(exc):
        return jsonify({"error": "internal error", "detail": str(exc)[:200]}), 500

    return app


def bootstrap(services: Services) -> None:
    """Load demo data the first time the app starts."""
    from .seed import DEVICES, KNOWLEDGE_BASE, THRESHOLDS, generate_metrics

    if services.docs.count("documents") == 0:
        for doc in KNOWLEDGE_BASE:
            services.docs.insert(
                "documents",
                {"doc_id": doc["doc_id"], "title": doc["title"], "text": doc["text"].strip(),
                 "source": "seed"},
            )

    if not services.sql.devices():
        for d in DEVICES:
            services.sql.upsert_device(d["device"], d["line"], d["location"], d["criticality"])
        for device, metric, warn, crit, unit in THRESHOLDS:
            services.sql.upsert_threshold(device, metric, warn, crit, unit)

    if not services.ts.query(hours=96, limit=1):
        services.ts.write(generate_metrics())

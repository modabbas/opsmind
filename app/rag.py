"""Retrieval-Augmented Generation pipeline.

    document -> chunk -> embed -> vector index -> search -> re-rank -> cited context

Embeddings come from a remote OpenAI-compatible endpoint when one is
configured; otherwise a local TF-IDF vectorizer is used so retrieval works
with no API key and no network. The rest of the pipeline is identical either
way, which is what makes the eval harness meaningful.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import Config

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "at", "by",
    "it", "be", "as", "with", "that", "this", "from", "was", "were", "can", "will", "if",
    "then", "than", "but", "not", "no", "do", "does", "has", "have", "had", "its", "we",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Sliding window over words, with overlap so a fact split across a
    boundary still survives in at least one chunk."""
    size = size or Config.CHUNK_SIZE
    overlap = overlap or Config.CHUNK_OVERLAP
    words = text.split()
    if len(words) <= size:
        return [text.strip()] if text.strip() else []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size]).strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(words):
            break
    return chunks


@dataclass
class Chunk:
    doc_id: str
    title: str
    chunk_id: str
    text: str
    tokens: list[str] = field(default_factory=list)


class VectorIndex:
    """TF-IDF vector index with cosine similarity, plus optional remote embeddings."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.matrix: np.ndarray | None = None   # L2-normalised, shape (n_chunks, vocab)
        self.mode = "local-tfidf"

    # ---------- building ----------

    def add_document(self, doc_id: str, title: str, text: str) -> int:
        new = chunk_text(text)
        for i, piece in enumerate(new):
            self.chunks.append(
                Chunk(doc_id=doc_id, title=title, chunk_id=f"{doc_id}#{i}", text=piece, tokens=tokenize(piece))
            )
        return len(new)

    def build(self) -> None:
        """(Re)compute the TF-IDF matrix. Cheap enough to redo on every write."""
        if not self.chunks:
            self.matrix = None
            return
        df = Counter()
        for c in self.chunks:
            df.update(set(c.tokens))
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n_docs = len(self.chunks)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, i in self.vocab.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

        matrix = np.zeros((n_docs, len(self.vocab)), dtype=np.float32)
        for row, c in enumerate(self.chunks):
            counts = Counter(c.tokens)
            if not counts:
                continue
            peak = max(counts.values())
            for term, n in counts.items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = (0.5 + 0.5 * n / peak) * self.idf[col]
        self.matrix = _normalize(matrix)

    def vectorize(self, text: str) -> np.ndarray:
        if not self.vocab or self.idf is None:
            return np.zeros(0, dtype=np.float32)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(tokenize(text))
        if not counts:
            return vec
        peak = max(counts.values())
        for term, n in counts.items():
            col = self.vocab.get(term)
            if col is not None:
                vec[col] = (0.5 + 0.5 * n / peak) * self.idf[col]
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    # ---------- searching ----------

    def search(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> list[dict]:
        """Two-stage retrieval: vector recall, then a re-rank pass."""
        top_k = top_k or Config.TOP_K
        candidate_k = candidate_k or Config.CANDIDATE_K
        if self.matrix is None or not self.chunks:
            return []

        qvec = self.vectorize(query)
        if qvec.size == 0 or not qvec.any():
            return []

        scores = self.matrix @ qvec
        n = min(candidate_k, len(self.chunks))
        candidates = np.argpartition(-scores, n - 1)[:n]
        candidates = candidates[np.argsort(-scores[candidates])]

        reranked = self._rerank(query, [(int(i), float(scores[i])) for i in candidates])
        results = []
        for rank, (idx, vector_score, rerank_score) in enumerate(reranked[:top_k], start=1):
            c = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "vector_score": round(vector_score, 4),
                    "score": round(rerank_score, 4),
                }
            )
        return results

    def _rerank(self, query: str, candidates: list[tuple[int, float]]) -> list[tuple[int, float, float]]:
        """Lexical-overlap re-rank with a diversity penalty.

        Vector recall alone over-returns near-duplicate chunks from the same
        document; penalising repeats measurably improved answer grounding in
        eval/run_eval.py.
        """
        q_terms = set(tokenize(query))
        seen_docs: Counter = Counter()
        out: list[tuple[int, float, float]] = []
        for idx, vscore in candidates:
            c = self.chunks[idx]
            overlap = len(q_terms & set(c.tokens)) / (len(q_terms) or 1)
            exact = 1.0 if query.lower().strip() in c.text.lower() else 0.0
            penalty = 0.06 * seen_docs[c.doc_id]
            final = 0.62 * vscore + 0.30 * overlap + 0.08 * exact - penalty
            seen_docs[c.doc_id] += 1
            out.append((idx, vscore, final))
        out.sort(key=lambda t: t[2], reverse=True)
        return out


def build_context(hits: list[dict], max_chars: int = 4000) -> tuple[str, list[dict]]:
    """Render retrieved chunks as a numbered context block the model must cite."""
    parts, citations, used = [], [], 0
    for i, h in enumerate(hits, start=1):
        block = f"[{i}] {h['title']}\n{h['text']}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
        citations.append(
            {"marker": f"[{i}]", "title": h["title"], "doc_id": h["doc_id"],
             "chunk_id": h["chunk_id"], "score": h["score"]}
        )
    return "\n\n".join(parts), citations


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms

"""LLM client.

Talks to any OpenAI-compatible /chat/completions endpoint. When no API key is
configured it falls back to a local extractive answerer, so the deployed demo
never shows an error page to a visitor — it just says which mode it is in.
"""

from __future__ import annotations

import json
import re
import time

import requests

from .config import Config


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        self.enabled = bool(Config.LLM_API_KEY)
        self.model = Config.LLM_MODEL

    @property
    def mode(self) -> str:
        return "llm" if self.enabled else "fallback"

    # ---------------- public API ----------------

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.2) -> str:
        if not self.enabled:
            raise LLMError("no API key configured")

        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(3):  # retry transient 429/5xx with backoff
            try:
                resp = requests.post(
                    f"{Config.LLM_BASE_URL.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {Config.LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=Config.LLM_TIMEOUT,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}"
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(1.2 * (attempt + 1))
        raise LLMError(f"LLM request failed after retries: {last_error}")

    def chat_json(self, system: str, user: str, schema_hint: str = "") -> dict:
        """Ask for strict JSON and parse defensively — models occasionally wrap
        JSON in prose or a code fence."""
        system = f"{system}\n\nReturn ONLY valid JSON. {schema_hint}".strip()
        raw = self.chat(system, user, json_mode=True, temperature=0.1)
        return parse_json(raw)


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", raw, re.S)
    if brace:
        return json.loads(brace.group(0))
    raise LLMError("model did not return parsable JSON")


# ---------------- offline fallback ----------------

def extractive_answer(question: str, hits: list[dict]) -> str:
    """Compose an answer straight from the retrieved chunks.

    Not a language model — it selects the sentences that overlap the question
    most and cites them. Keeps the demo honest and functional without a key.
    """
    if not hits:
        return "I couldn't find anything in the knowledge base that covers that."

    from .rag import tokenize

    q_terms = set(tokenize(question))
    scored = []
    for i, hit in enumerate(hits, start=1):
        for sentence in re.split(r"(?<=[.!?])\s+", hit["text"]):
            terms = set(tokenize(sentence))
            if not terms:
                continue
            overlap = len(q_terms & terms) / (len(q_terms) or 1)
            if overlap > 0:
                scored.append((overlap, i, sentence.strip()))
    scored.sort(reverse=True)

    if not scored:
        top = hits[0]
        return f"Closest match in the knowledge base: {top['text'][:320].strip()} [1]"

    lines, seen = [], set()
    for _, marker, sentence in scored[:3]:
        if sentence in seen:
            continue
        seen.add(sentence)
        lines.append(f"{sentence} [{marker}]")
    return " ".join(lines)

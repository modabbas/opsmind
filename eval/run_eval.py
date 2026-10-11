"""RAG evaluation harness.

A fixed question set with the document each answer must come from. Reports
retrieval hit-rate at k, mean reciprocal rank, grounding (did the answer cite
anything) and latency. This is what turns "I tuned the prompt" into a number.

    python eval/run_eval.py
    python eval/run_eval.py --chunk-size 160 --top-k 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_SET = [
    # (question, document the answer lives in, fact that must survive chunking)
    ("At what temperature is a bottling line considered to be overheating?", "SOP-101", "78 degC"),
    ("Should we stop the line at the warning temperature level?", "SOP-101", "Do not stop the line"),
    ("What are the most common causes of line overheating?", "SOP-101", "clogged coolant filter"),
    ("How long does a hot stop add to recovery?", "SOP-101", "forty minutes"),
    ("What is the vibration warning limit for PUMP-07?", "SOP-204", "3.5 mm/s"),
    ("What does a slow rise in vibration over several days indicate?", "SOP-204", "bearing wear"),
    ("How long does a bearing replacement take on a bottling line?", "SOP-204", "three hours"),
    ("What is the replacement interval for the main drive bearing?", "SOP-204", "8000 running hours"),
    ("What is the normal discharge pressure for PUMP-07?", "SOP-318", "5.2 to 5.6 bar"),
    ("Can we switch to the standby pump without supervisor approval?", "SOP-318", "does not require supervisor approval"),
    ("What happens if pump pressure drops below 4.5 bar?", "SOP-318", "lose cooling"),
    ("Why was the LINE-B cooling fan bearing replaced?", "MAINT-LOG", "three consecutive shifts"),
    ("What was found during the LINE-A bearing inspection?", "MAINT-LOG", "Wear within limits"),
    ("What is the throughput target for LINE-A?", "KPI-DEF", "110 units per minute"),
    ("What must a shift report contain?", "KPI-DEF", "alarms raised by severity"),
    ("Can a finding be closed without a stated cause?", "KPI-DEF", "Never close a finding"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    # Config reads the environment, so overrides go in before the import.
    if args.chunk_size:
        os.environ["CHUNK_SIZE"] = str(args.chunk_size)
    if args.chunk_overlap:
        os.environ["CHUNK_OVERLAP"] = str(args.chunk_overlap)
    if args.top_k:
        os.environ["TOP_K"] = str(args.top_k)

    from app import create_app
    from app.config import Config

    app = create_app()
    client = app.test_client()

    hits_at_1 = hits_at_k = grounded = span_hits = 0
    reciprocal_total = 0.0
    latencies: list[float] = []

    print(f"\nchunk_size={Config.CHUNK_SIZE} overlap={Config.CHUNK_OVERLAP} top_k={Config.TOP_K}")
    print("-" * 78)

    for question, expected_doc, gold_fact in EVAL_SET:
        started = time.perf_counter()
        result = client.post(
            "/api/search", json={"question": question, "query": question, "top_k": Config.TOP_K}
        ).get_json()
        latencies.append((time.perf_counter() - started) * 1000)

        hits = result["hits"]
        retrieved_docs = [h["doc_id"] for h in hits]
        rank = retrieved_docs.index(expected_doc) + 1 if expected_doc in retrieved_docs else 0

        # The hard metric: is the sentence that actually answers the question
        # inside the retrieved text, or did chunking cut it out?
        context = " ".join(h["text"] for h in hits).lower()
        span_ok = gold_fact.lower() in context

        if rank == 1:
            hits_at_1 += 1
        if rank:
            hits_at_k += 1
            reciprocal_total += 1 / rank
        if span_ok:
            span_hits += 1
        if hits:
            grounded += 1

        flag = "ok  " if (rank == 1 and span_ok) else ("part" if rank else "MISS")
        print(f"{flag} rank={rank or '-'} span={'y' if span_ok else 'n'}  {question[:56]}")

    n = len(EVAL_SET)
    print("-" * 78)
    print(f"hit@1           {hits_at_1}/{n}  ({hits_at_1 / n:.0%})")
    print(f"hit@{Config.TOP_K}           {hits_at_k}/{n}  ({hits_at_k / n:.0%})")
    print(f"MRR             {reciprocal_total / n:.3f}")
    print(f"answer span@{Config.TOP_K}   {span_hits}/{n}  ({span_hits / n:.0%})   <- the metric that moves")
    print(f"retrieval p50   {sorted(latencies)[n // 2]:.0f} ms")
    print()
    return 0 if span_hits / n >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import httpx


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def summarize(samples: list[dict]) -> dict:
    relevant = [sample for sample in samples if sample["expected_filename"]]
    unsupported = [sample for sample in samples if sample["expected_supported"] is False]
    costs = [sample["usage"]["estimated_cost_usd"] for sample in samples]
    return {
        "questions": len(samples),
        "recall_at_k": (sum(any(c["filename"] == s["expected_filename"] for c in s["retrieved"])
                            for s in relevant) / len(relevant)) if relevant else None,
        "unanswerable_abstention_rate": (sum(not s["supported"] for s in unsupported) / len(unsupported))
                                      if unsupported else None,
        "latency_ms_p50": median(s["latency_ms"] for s in samples) if samples else None,
        "latency_ms_p95": percentile([s["latency_ms"] for s in samples], 0.95) if samples else None,
        "calls": sum(s["usage"]["calls"] for s in samples),
        "input_tokens": sum(s["usage"]["input_tokens"] for s in samples),
        "output_tokens": sum(s["usage"]["output_tokens"] for s in samples),
        "embedding_tokens": sum(s["usage"]["embedding_tokens"] for s in samples),
        "estimated_cost_usd": round(sum(costs), 6) if all(cost is not None for cost in costs) else None,
    }


def run(base_url: str, fixture_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    uploaded = []
    samples = []
    with httpx.Client(base_url=base_url, timeout=90) as client:
        try:
            for document in fixture["documents"]:
                response = client.post("/api/documents", files={"file": (document["filename"], document["text"].encode())})
                response.raise_for_status()
                uploaded.append(response.json()["id"])
            for question in fixture["questions"]:
                started = perf_counter()
                response = client.post("/api/questions", json={"question": question["question"], "top_k": 1})
                response.raise_for_status()
                data = response.json()
                samples.append({"expected_filename": question.get("expected_filename"),
                                "expected_supported": question["expected_supported"],
                                "retrieved": data["retrieved"], "supported": data["validation"]["supported"],
                                "latency_ms": round((perf_counter() - started) * 1000, 2), "usage": data["usage"]})
        finally:
            for document_id in uploaded:
                client.delete(f"/api/documents/{document_id}")
    return summarize(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("fixtures.json"))
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.base_url, arguments.fixture), ensure_ascii=False, indent=2))

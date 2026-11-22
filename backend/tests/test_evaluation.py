from eval.evaluate import summarize


def test_summarize_reports_recall_abstention_latency_and_tokens():
    samples = [
        {"expected_filename": "a.txt", "expected_supported": True, "retrieved": [{"filename": "a.txt"}],
         "supported": True, "latency_ms": 100, "usage": {"calls": 3, "input_tokens": 10,
         "output_tokens": 5, "embedding_tokens": 2, "estimated_cost_usd": 0.001}},
        {"expected_filename": None, "expected_supported": False, "retrieved": [],
         "supported": False, "latency_ms": 200, "usage": {"calls": 3, "input_tokens": 20,
         "output_tokens": 5, "embedding_tokens": 2, "estimated_cost_usd": 0.002}},
    ]
    result = summarize(samples)
    assert result["recall_at_k"] == 1.0
    assert result["unanswerable_abstention_rate"] == 1.0
    assert result["latency_ms_p50"] == 150
    assert result["calls"] == 6
    assert result["estimated_cost_usd"] == 0.003

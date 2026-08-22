from crupier.costs import (
    actual_cost_from_calls,
    estimate_model_cost,
    estimate_tokens,
    usage_estimated_cost_from_calls,
)


def test_actual_cost_returns_none_when_metered_call_lacks_cost():
    calls = [
        {"usage": {"input_tokens": 10, "output_tokens": 2}},
        {"model": "openai:test"},
    ]

    assert actual_cost_from_calls(calls, []) is None


def test_actual_cost_skips_unmetered_usage_and_sums_reported_cost():
    calls = [
        {"usage": {"cache_status": "hit"}},
        {"usage": {"prompt_tokens": 1, "total_cost_usd": 0.125}},
    ]

    assert actual_cost_from_calls(calls, []) == 0.125


def test_usage_estimated_cost_skips_incomplete_calls():
    calls = [
        {"usage": {"input_tokens": 9_000_000}},
        {"model": "missing:usage", "usage": {}},
        {"model": "missing:tokens", "usage": {"cache_status": "hit"}},
        {
            "model": "unknown:model",
            "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        },
    ]

    assert usage_estimated_cost_from_calls(calls, []) == 6.0


def test_estimate_tokens_empty_and_pricing_without_card():
    assert estimate_tokens("") == 0
    assert estimate_model_cost(None, input_tokens=1_000_000, output_tokens=1_000_000) == 6.0

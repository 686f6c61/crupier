from crupier import Crupier
from crupier.adapters import AdapterResponse
from crupier.config import CrupierConfig
from crupier.models import RequestEnvelope
from crupier.registry import ModelRegistry
from crupier.runtime_policy import apply_runtime_policy


class CapturingNaNAdapter:
    provider = "nan"

    def __init__(self):
        self.requests = []

    def generate(self, *, model, prompt, request):
        self.requests.append(request)
        return AdapterResponse(text="image-ok", metadata={"provider": "nan", "model": model})


def test_runtime_policy_disables_default_thinking_for_tight_routine_request(tmp_path):
    card = ModelRegistry.builtin_cards()["nan:qwen3.6"]
    request = RequestEnvelope(
        task="Classify this as billing or technical",
        mode="agentic",
        constraints={"max_output_tokens": 128},
    )

    updated, policy = apply_runtime_policy("nan:qwen3.6", request, card)

    assert updated.constraints["enable_thinking"] is False
    assert policy["reason"] == "tight_output_budget"
    assert "enable_thinking" not in request.constraints


def test_runtime_policy_enables_thinking_for_complex_reasoning_request():
    card = ModelRegistry.builtin_cards()["nan:qwen3.6"]
    request = RequestEnvelope(
        task="Prove this algebra theorem and critique every assumption",
        constraints={"max_output_tokens": 2000},
    )

    updated, policy = apply_runtime_policy("nan:qwen3.6", request, card)

    assert updated.constraints["enable_thinking"] is True
    assert policy["reason"] == "complex_request"


def test_runtime_policy_never_overrides_explicit_reasoning_control():
    card = ModelRegistry.builtin_cards()["nan:qwen3.6"]
    request = RequestEnvelope(task="Simple answer", constraints={"enable_thinking": True, "max_output_tokens": 32})

    updated, policy = apply_runtime_policy("nan:qwen3.6", request, card)

    assert updated is request
    assert policy["source"] == "request"


def test_runtime_policy_derives_deepseek_effort_from_complexity():
    card = ModelRegistry.builtin_cards()["nan:deepseek-v4-flash"]
    request = RequestEnvelope(task="Audit this complex codebase and reason about concurrency failures")

    updated, policy = apply_runtime_policy("nan:deepseek-v4-flash", request, card)

    assert updated.constraints["reasoning_effort"] == "high"
    assert policy["reasoning_effort"] == "high"


def test_executor_applies_runtime_policy_before_provider_call(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"nan": {"enabled": True, "env_key": "NAN_API_KEY"}},
            "models": {"allow": ["nan:qwen3.6"]},
            "routing": {"default_strategy": "single", "require_operational_providers": False},
            "profiles": {"agentic": {"strategy": "single"}},
        }
    )
    config.root = tmp_path
    adapter = CapturingNaNAdapter()
    client = Crupier(config, adapters={"nan": adapter})

    result = client.deal(
        "Classify this ticket",
        constraints={"max_output_tokens": 128},
        dry_run=False,
        trace="summary",
    )

    assert adapter.requests[0].constraints["enable_thinking"] is False
    runtime = result.provider_metadata["calls"][0]["runtime_policy"]
    assert runtime["thinking_enabled"] is False

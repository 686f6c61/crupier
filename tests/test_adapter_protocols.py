import pytest

from crupier import Crupier
from crupier.adapters import AdapterResponse, ProviderAdapter, ProviderModel
from crupier.config import CrupierConfig
from crupier.errors import CrupierError


class GenerateOnlyAdapter:
    provider = "openai"

    def generate(self, *, model, prompt, request):
        return AdapterResponse(text="generate-only: ok")


class DiscoveryAdapter(GenerateOnlyAdapter):
    provider = "anthropic"

    def list_models(self):
        return [ProviderModel(id="claude-test", provider=self.provider)]


class NonCallableDiscoveryAdapter(GenerateOnlyAdapter):
    list_models = "disabled"


class NonCallableProbeAdapter(GenerateOnlyAdapter):
    probe_capability = "disabled"


def _execution_adapter(adapter: ProviderAdapter) -> ProviderAdapter:
    """Keep the generate-only structural contract covered by mypy."""
    return adapter


def _client(tmp_path, adapters: dict[str, ProviderAdapter] | None = None) -> Crupier:
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    selected_adapters = adapters or {"openai": _execution_adapter(GenerateOnlyAdapter())}
    return Crupier(config, adapters=selected_adapters)


def test_generate_only_adapter_satisfies_execution_contract(tmp_path):
    result = _client(tmp_path).deal("Say hi", dry_run=False)

    assert result.output_text == "generate-only: ok"


@pytest.mark.parametrize("adapter", [GenerateOnlyAdapter(), NonCallableDiscoveryAdapter()])
def test_discovery_reports_missing_or_non_callable_capability(tmp_path, adapter):
    client = _client(tmp_path, {"openai": _execution_adapter(adapter)})

    with pytest.raises(CrupierError, match="list_models"):
        client.models.discover(provider="openai")


def test_discovery_skip_unavailable_preserves_healthy_providers(tmp_path):
    client = _client(
        tmp_path,
        {
            "openai": _execution_adapter(GenerateOnlyAdapter()),
            "anthropic": _execution_adapter(DiscoveryAdapter()),
        },
    )
    warnings = []

    models = client.models.discover(skip_unavailable=True, warnings=warnings)

    assert [model.model_ref for model in models] == ["anthropic:claude-test"]
    assert len(warnings) == 1
    assert "openai" in warnings[0]
    assert "list_models" in warnings[0]


@pytest.mark.parametrize("adapter", [GenerateOnlyAdapter(), NonCallableProbeAdapter()])
def test_missing_probe_capability_degrades_without_aborting(tmp_path, adapter):
    client = _client(tmp_path, {"openai": _execution_adapter(adapter)})

    report = client.capabilities.probe(
        ["openai:gpt-5.5"],
        probes=["structured_output"],
    )

    assert report.results[0].status in {"inferred", "unknown"}
    assert "No native provider probe" in report.results[0].metadata["note"]

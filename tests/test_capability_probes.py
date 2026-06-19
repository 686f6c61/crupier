import json

from crupier import Crupier
from crupier.adapters import AdapterResponse, EmbeddingResponse, ProviderModel
from crupier.cli import main
from crupier.config import CrupierConfig


class FakeProbeAdapter:
    provider = "openai"

    def __init__(self):
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append({"model": model, "prompt": prompt, "constraints": request.constraints})
        if "JSON object" in prompt:
            return AdapterResponse(
                text='{"ok": true, "probe": "crupier"}',
                usage={"output_tokens": 8},
                metadata={"fake": True},
            )
        return AdapterResponse(
            text="crupier-probe-ok",
            usage={"output_tokens": 3},
            metadata={"fake": True},
        )

    def list_models(self):
        return [ProviderModel(id="gpt-5.5", provider="openai")]


class NativeProbeAdapter(FakeProbeAdapter):
    def probe_capability(self, *, model, probe, request):
        return AdapterResponse(
            text="",
            usage={"output_tokens": 1},
            metadata={
                "native_probe": True,
                "probe_status": "verified",
                "ok": True,
                "capability": probe,
            },
        )


class FakeEmbeddingAdapter(FakeProbeAdapter):
    def embed(self, *, model, input):
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            usage={"prompt_tokens": 3, "total_tokens": 3},
            metadata={"provider": "openai", "model": model, "api": "embeddings.create"},
        )


def test_capability_probe_apply_updates_card(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    adapter = FakeProbeAdapter()
    client = Crupier(config, adapters={"openai": adapter})

    report = client.capabilities.probe(
        ["openai:gpt-5.5"],
        probes=["text_basic", "json_instruction", "tool_call", "streaming"],
        apply=True,
    )

    assert report.summary() == {"verified": 2, "inferred": 2}
    assert len(adapter.calls) == 2
    assert len(report.written_files) == 1

    card_path = tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.5.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["capability_status"]["text_generation"]["status"] == "verified"
    assert card["capability_status"]["json_instruction"]["status"] == "verified"
    assert card["capability_status"]["tool_call"]["status"] == "inferred"
    assert card["capability_status"]["streaming"]["status"] == "inferred"
    assert card["local_eval_scores"]["probe_text_basic"] == 1.0
    assert "crupier-probe-ok" not in json.dumps(card)


def test_capability_probe_native_results_update_support_flags(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": NativeProbeAdapter()})

    report = client.capabilities.probe(
        ["openai:gpt-5.5"],
        probes=["max_output_param", "structured_output", "tool_call", "streaming"],
        apply=True,
    )

    assert report.summary() == {"verified": 4}
    card_path = tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.5.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["supports_structured_output"] is True
    assert card["supports_tools"] is True
    assert card["supports_streaming"] is True
    assert card["capability_status"]["structured_output"]["status"] == "verified"
    assert card["capability_status"]["tool_call"]["status"] == "verified"
    assert card["capability_status"]["streaming"]["status"] == "verified"


def test_capability_readiness_reports_needs_probes_for_unverified_card(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": NativeProbeAdapter()})

    report = client.capabilities.readiness(["openai:gpt-5.5"])

    assert report.summary() == {"needs_probes": 1}
    item = report.items[0]
    assert item.missing_probes
    assert "text_basic" in item.missing_probes


def test_embedding_model_readiness_uses_embedding_probe_only(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:text-embedding-3-small"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeEmbeddingAdapter()})

    report = client.capabilities.readiness(["openai:text-embedding-3-small"])

    assert report.summary() == {"needs_probes": 1}
    assert report.items[0].missing_probes == []
    assert report.items[0].inferred_probes == ["embeddings"]
    assert [item["probe"] for item in report.items[0].required_probes] == ["embeddings"]


def test_embedding_probe_apply_updates_embedding_card(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:text-embedding-3-small"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeEmbeddingAdapter()})

    report = client.capabilities.probe(["openai:text-embedding-3-small"], probes=["embeddings"], apply=True)

    assert report.summary() == {"verified": 1}
    readiness = client.capabilities.readiness(["openai:text-embedding-3-small"])
    assert readiness.summary() == {"ready": 1}
    card_path = tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__text-embedding-3-small.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["model_kind"] == "embedding"
    assert card["supports_embeddings"] is True
    assert card["embedding_dimensions"] == 4
    assert card["modalities_output"] == ["embedding"]
    assert card["capability_status"]["embeddings"]["status"] == "verified"


def test_capability_readiness_ready_after_verified_probes(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": NativeProbeAdapter()})
    client.capabilities.probe(["openai:gpt-5.5"], apply=True)

    report = client.capabilities.readiness(["openai:gpt-5.5"])

    assert report.summary() == {"ready": 1}
    assert report.items[0].missing_probes == []
    assert report.items[0].failed_probes == []
    assert report.items[0].inferred_probes == []


def test_capability_probe_without_apply_does_not_write(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True}},
            "models": {"allow": ["openai:gpt-5.5"]},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": FakeProbeAdapter()})

    report = client.capabilities.probe(["openai:gpt-5.5"], probes=["text_basic"], apply=False)

    assert report.summary() == {"verified": 1}
    assert report.written_files == []
    assert not (tmp_path / ".crupier" / "registry" / "capability-cards" / "openai__gpt-5.5.json").exists()


def test_cli_capabilities_probe_dry_run(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "capabilities", "probe", "--dry-run", "--probe", "text_basic"]) == 0

    output = capsys.readouterr().out
    assert "capability_probe: dry-run not-applied" in output
    assert "planned" in output


def test_cli_capabilities_readiness(tmp_path, capsys):
    assert main(["--project", str(tmp_path), "init"]) == 0
    assert main(["--project", str(tmp_path), "capabilities", "readiness"]) == 0

    output = capsys.readouterr().out
    assert "capability_readiness: standard" in output
    assert "needs_probes" in output

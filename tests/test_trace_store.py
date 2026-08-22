import json
import os
import stat
from pathlib import Path

import pytest

from crupier import Crupier
from crupier.cli import main
from crupier.config import CrupierConfig, write_default_project
from crupier.errors import CrupierProviderUnavailableError
from crupier.redaction import redact_text, redact_value


def make_config(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "trace-test", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.4-mini"]},
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    return config


def test_trace_store_is_opt_in(tmp_path):
    client = Crupier(make_config(tmp_path))

    client.deal("Plan this", dry_run=True, trace=False)

    assert client.traces.list() == []


def test_sensitive_artifacts_are_private_under_permissive_umask(tmp_path):
    client = Crupier(make_config(tmp_path))
    previous_umask = os.umask(0o022)
    try:
        trace = client.deal(
            "Store private evidence",
            constraints={"store_trace": True},
            dry_run=True,
            trace="summary",
        )
        client.feedback.record(
            project=client.config.project.name,
            models=["openai:gpt-5.4-mini"],
            rating=4,
        )
        eval_report = client.evals.compare(task="Private eval", write_report=True, dry_run=True)
    finally:
        os.umask(previous_umask)

    assert trace.trace is not None
    assert eval_report.written_path is not None
    files = [
        client.config.traces_dir / f"{trace.trace.trace_id}.json",
        client.config.feedback_dir / "feedback.jsonl",
        Path(eval_report.written_path),
    ]
    directories = [client.config.traces_dir, client.config.feedback_dir, client.config.evals_dir]
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in directories)


def test_trace_store_metadata_does_not_store_prompt_response_or_secret(tmp_path):
    client = Crupier(make_config(tmp_path))
    fake_secret = "s" + "k-test-secret-value"

    result = client.deal(
        f"Plan this with {fake_secret}",
        input={"token": fake_secret},
        constraints={"store_trace": True},
        dry_run=True,
        trace=False,
    )

    refs = client.traces.list()
    assert len(refs) == 1
    record = client.traces.read(refs[0].trace_id)
    serialized = json.dumps(record)
    assert result.trace is None
    assert "task" not in record["request"]
    assert "input" not in record["request"]
    assert "output_text" not in record["result"]
    assert fake_secret not in serialized
    assert "Plan this with" not in serialized
    assert record["replayable"] is False


def test_metadata_trace_contains_no_task_substring_when_store_prompt_is_false(tmp_path):
    client = Crupier(make_config(tmp_path))
    task = (
        "Revisa el expediente de Marta Pérez con NIF 12345678Z "
        "en Calle Falsa 123, teléfono 600111222."
    )
    fragments = ["Marta Pérez", "12345678Z", "Calle Falsa", "600111222"]

    client.deal(
        task,
        input={"customer": "Marta Pérez", "nif": "12345678Z"},
        constraints={"store_trace": True, "store_prompt": False},
        dry_run=True,
        trace="summary",
    )

    refs = client.traces.list()
    assert len(refs) == 1
    record = client.traces.read(refs[0].trace_id)
    serialized = json.dumps(record)
    for fragment in fragments:
        assert fragment not in serialized
        assert fragment not in record["request"].get("summary", "")
        assert fragment not in str(record.get("trace", {}).get("request_summary", ""))
    assert "task" not in record["request"]


def test_metadata_trace_keeps_non_content_metrics(tmp_path):
    client = Crupier(make_config(tmp_path))
    task = "Tarea con datos personales de Marta Pérez para medir la longitud."

    result = client.deal(
        task,
        constraints={"store_trace": True, "store_prompt": False},
        dry_run=True,
        trace="summary",
    )

    record = client.traces.read(result.trace.trace_id)
    route = record["result"]["route"]
    summary = record["request"]["summary"]
    trace_summary = record["trace"]["request_summary"]
    assert route["strategy"] == "single"
    assert "openai:gpt-5.4-mini" in json.dumps(route)
    assert "cost" in record["result"]
    assert record["result"]["cost"] is not None
    assert f"chars={len(task)}" in summary
    assert f"chars={len(task)}" in trace_summary
    assert "prompt omitted" in summary
    assert "sha256=" in summary


def test_trace_store_replay_requires_prompt_storage(tmp_path):
    client = Crupier(make_config(tmp_path))
    original = client.deal(
        "Replay this exact route",
        constraints={"store_trace": True, "store_prompt": True, "store_response": True},
        dry_run=True,
        trace="summary",
    )
    trace_id = original.trace.trace_id

    record = client.traces.read(trace_id)
    replay = client.traces.replay(trace_id, client, dry_run=True, trace="summary")

    assert record["replayable"] is True
    assert record["request"]["task"] == "Replay this exact route"
    assert record["result"]["output_text"].startswith("Crupier dry-run planned")
    assert replay.route.strategy == original.route.strategy


def test_cli_trace_commands(tmp_path, capsys):
    write_default_project(tmp_path)
    assert main(["--project", str(tmp_path), "deal", "Trace me", "--store-prompt", "--store-response"]) == 0
    capsys.readouterr()

    assert main(["--project", str(tmp_path), "trace", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    trace_id = listed[0]["trace_id"]
    assert listed[0]["replayable"] is True

    assert main(["--project", str(tmp_path), "trace", "show", trace_id, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["request"]["task"] == "Trace me"

    assert main(["--project", str(tmp_path), "trace", "replay", trace_id, "--json"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["route"]["strategy"] == shown["result"]["route"]["strategy"]

    assert main(["--project", str(tmp_path), "trace", "delete", trace_id]) == 0
    capsys.readouterr()
    assert main(["--project", str(tmp_path), "trace", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


PROVIDER_SECRETS = [
    ("openai", "sk-proj-abcdefghijklmnopqrstuvwxyz012345"),
    ("anthropic", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"),
    ("google", "AIzaSyDaGmWKa4JsXZHjGw7ISLn_3namBGewQe"),
    ("ollama", "ollama_live_abcdefghijklmnopqrstuvwxyz012345"),
    ("openrouter", "sk-or-v1-" + "ab" * 32),
    ("nan", "nan_live_abcdefghijklmnopqrstuvwxyz012345"),
    ("aws", "AKIAIOSFODNN7EXAMPLE"),
    ("bearer", "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signaturelong"),
]


@pytest.mark.parametrize(("provider", "secret"), PROVIDER_SECRETS)
def test_central_redactor_covers_supported_provider_key_formats(provider: str, secret: str) -> None:
    redacted = redact_text(f"{provider} credential {secret} stays hidden")
    assert secret not in redacted
    assert "[redacted]" in redacted


def test_structural_redactor_blocks_secret_field_names_case_insensitively() -> None:
    payload = {
        "Authorization": "secret-header-value",
        "nested": {
            "X-Api-Key": "nested-api-key-value",
            "Password": "hunter2-secret",
            "headers": {"cookie": "session=abc123secret"},
        },
    }
    redacted = redact_value(payload)
    serialized = json.dumps(redacted)
    for secret in ("secret-header-value", "nested-api-key-value", "hunter2-secret", "session=abc123secret"):
        assert secret not in serialized
    assert redacted["Authorization"] == "[redacted]"
    assert redacted["nested"]["X-Api-Key"] == "[redacted]"
    assert redacted["nested"]["Password"] == "[redacted]"
    assert redacted["nested"]["headers"]["cookie"] == "[redacted]"


class _SecretFailAdapter:
    provider = "openai"

    def generate(self, *, model, prompt, request):
        del model, prompt, request
        raise CrupierProviderUnavailableError(
            "upstream rejected AIzaSyDaGmWKa4JsXZHjGw7ISLn_3namBGewQe"
        )


def test_trace_feedback_eval_and_experiment_use_same_redactor(tmp_path) -> None:
    secret = "AIzaSyDaGmWKa4JsXZHjGw7ISLn_3namBGewQe"
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "redact-artifacts", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.4-mini", "openai:gpt-5.5"]},
            "routing": {
                "default_strategy": "single",
                "require_operational_providers": False,
                "max_provider_retries": 0,
            },
            "experiments": {
                "shadow-sync": {
                    "traffic": "shadow",
                    "sample_rate": 1.0,
                    "execution": "sync",
                    "candidate_models": ["openai:gpt-5.4-mini"],
                }
            },
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={"openai": _SecretFailAdapter()})

    client.deal(
        f"Plan with {secret}",
        constraints={"store_trace": True, "store_prompt": True},
        dry_run=True,
        trace="summary",
    )
    trace_blob = json.dumps(client.traces.read(client.traces.list()[0].trace_id))

    feedback = client.feedback.record(
        project=client.config.project.name,
        models=["openai:gpt-5.4-mini"],
        rating=3,
        note=f"reviewer pasted {secret}",
    )
    feedback_blob = json.dumps(feedback.to_dict())

    eval_report = client.evals.compare(task=f"Compare {secret}", write_report=True, dry_run=True)
    eval_blob = Path(eval_report.written_path).read_text(encoding="utf-8")

    try:
        client.deal(
            "Run experiment",
            constraints={"force_model": "openai:gpt-5.5"},
            dry_run=False,
            experiment="shadow-sync",
        )
    except CrupierProviderUnavailableError:
        pass
    experiment_blob = json.dumps(
        [record.to_dict() for record in client.experiments.store.list("experiment_observation", limit=100)]
    )

    for name, blob in (
        ("trace", trace_blob),
        ("feedback", feedback_blob),
        ("eval", eval_blob),
        ("experiment", experiment_blob),
    ):
        assert secret not in blob, name
        assert "[redacted]" in blob, name

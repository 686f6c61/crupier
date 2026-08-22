import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _synthetic_secrets import (
    SYNTHETIC_ANTHROPIC_API_KEY,
    SYNTHETIC_BEARER_TOKEN,
    SYNTHETIC_GOOGLE_API_KEY,
    SYNTHETIC_OPENAI_API_KEY,
)

from crupier import Crupier
from crupier.cli import main
from crupier.config import CrupierConfig, write_default_project
from crupier.errors import CrupierProviderUnavailableError
from crupier.models import PlanningContext, RequestEnvelope
from crupier.orchestrator import ModelOrchestrator
from crupier.project_audit import _canary_error
from crupier.redaction import redact_text, redact_value
from crupier.trace_store import _jsonable


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


def _timestamp(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).replace(microsecond=0).isoformat()


def _trace_record(trace_id: str, *, days_ago: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "created_at": _timestamp(days_ago),
        "replayable": False,
        "request": {"summary": trace_id},
        "result": {"route": {"strategy": "single", "steps": []}},
    }


def test_trace_store_prunes_entries_older_than_ttl_days(tmp_path):
    config = make_config(tmp_path)
    config.logging.ttl_days = 7
    config.traces_dir.mkdir(parents=True)
    (config.traces_dir / "old.json").write_text(
        json.dumps(_trace_record("old", days_ago=8)), encoding="utf-8"
    )
    (config.traces_dir / "recent.json").write_text(
        json.dumps(_trace_record("recent", days_ago=6)), encoding="utf-8"
    )

    client = Crupier(config)

    assert [item.trace_id for item in client.traces.list()] == ["recent"]
    assert not (config.traces_dir / "old.json").exists()


def test_feedback_and_evals_respect_ttl_days(tmp_path):
    config = make_config(tmp_path)
    config.logging.ttl_days = 7
    config.feedback_dir.mkdir(parents=True)
    config.evals_dir.joinpath("history").mkdir(parents=True)
    feedback_rows = [
        {
            "schema_version": 1,
            "feedback_id": "old",
            "created_at": _timestamp(8),
            "project": "trace-test",
            "rating": 1,
            "models": ["openai:gpt-5.4-mini"],
        },
        {
            "schema_version": 1,
            "feedback_id": "recent",
            "created_at": _timestamp(6),
            "project": "trace-test",
            "rating": 5,
            "models": ["openai:gpt-5.4-mini"],
        },
    ]
    history_rows = [
        {"schema_version": 1, "run_id": "old", "created_at": _timestamp(8), "model_scores": []},
        {"schema_version": 1, "run_id": "recent", "created_at": _timestamp(6), "model_scores": []},
    ]
    config.feedback_dir.joinpath("feedback.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in feedback_rows), encoding="utf-8"
    )
    config.evals_dir.joinpath("history", "compare_runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history_rows), encoding="utf-8"
    )

    client = Crupier(config)

    assert [item.feedback_id for item in client.feedback.list()] == ["recent"]
    assert client.evals.history().total_runs == 1


def test_purge_command_reports_removed_artifacts(tmp_path, capsys):
    write_default_project(tmp_path)
    config_path = tmp_path / "crupier.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("ttl_days = 30", "ttl_days = 7"),
        encoding="utf-8",
    )
    traces_dir = tmp_path / ".crupier" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.joinpath("old.json").write_text(
        json.dumps(_trace_record("old", days_ago=8)), encoding="utf-8"
    )

    assert main(["--project", str(tmp_path), "purge", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"evals": 0, "feedback": 0, "total": 1, "traces": 1}


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


def test_trace_list_reports_corrupt_trace_path(tmp_path):
    client = Crupier(make_config(tmp_path))
    valid = client.deal(
        "Keep valid trace",
        constraints={"store_trace": True},
        dry_run=True,
        trace="summary",
    )
    corrupt_path = client.config.traces_dir / "truncated.json"
    corrupt_path.write_text('{"trace_id":', encoding="utf-8")

    result = client.traces.list()

    assert valid.trace is not None
    assert [item.trace_id for item in result] == [valid.trace.trace_id]
    assert result.complete is False
    assert result.diagnostics[0].path == corrupt_path
    assert result.diagnostics[0].error_type == "invalid_json"


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


def test_metadata_trace_redacts_provider_metadata_and_warnings(tmp_path):
    client = Crupier(make_config(tmp_path))
    secret = SYNTHETIC_GOOGLE_API_KEY
    request = RequestEnvelope(
        task="Expediente de Marta Pérez",
        constraints={"store_trace": True, "store_prompt": False},
    )
    result = client.deal(
        request.task,
        constraints=request.constraints,
        dry_run=True,
        trace="summary",
    )
    result.provider_metadata["upstream"] = {"credential": secret}
    result.warnings.append(f"El proveedor devolvió {secret}")

    client.traces.write(
        project=client.config.project.name,
        request=request,
        result=result,
        dry_run=True,
        trace_level="summary",
    )

    serialized = json.dumps(client.traces.read(result.trace.trace_id))
    assert secret not in serialized
    assert "[redacted]" in serialized


def test_metadata_trace_omits_orchestrator_error_text_when_store_response_is_false(tmp_path):
    client = Crupier(make_config(tmp_path))
    fragment = "Marta Perez NIF 12345678Z"
    request = RequestEnvelope(
        task="Clasifica el expediente",
        constraints={"store_trace": True, "store_prompt": False, "store_response": False},
    )
    result = client.deal(
        request.task,
        constraints=request.constraints,
        dry_run=True,
        trace="summary",
    )
    result.provider_metadata["orchestrator_calls"] = [
        {
            "provider": "openai",
            "model": "openai:planner",
            "error_type": "ProviderError",
            "error": f"Falló al procesar {fragment}",
        }
    ]
    assert result.route is not None
    result.route.reason = f"Fallback tras error: {fragment}"

    client.traces.write(
        project=client.config.project.name,
        request=request,
        result=result,
        dry_run=True,
        trace_level="summary",
    )

    record = client.traces.read(result.trace.trace_id)
    serialized = json.dumps(record)
    call = record["result"]["provider_metadata"]["orchestrator_calls"][0]
    assert fragment not in serialized
    assert "error" not in call
    assert call["error_type"] == "ProviderError"


def test_orchestrator_call_record_redacts_provider_exception() -> None:
    secret = SYNTHETIC_GOOGLE_API_KEY
    context = PlanningContext(
        request=RequestEnvelope(task="planificar"),
        candidates=[],
    )

    ModelOrchestrator._record_orchestrator_call(
        context,
        {
            "error_type": "ProviderError",
            "error": str(CrupierProviderUnavailableError(f"rechazo para {secret}")),
        },
    )

    call = context.request.metadata["_crupier_orchestrator_calls"][0]
    assert secret not in call["error"]
    assert "[redacted]" in call["error"]


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


def test_trace_store_tool_calls_redaction_respects_store_flags(tmp_path):
    client = Crupier(make_config(tmp_path))
    records = {}
    for store_content in (False, True):
        constraints = {
            "store_trace": True,
            "store_prompt": store_content,
            "store_response": store_content,
        }
        request = RequestEnvelope(task=f"tool trace {store_content}", constraints=constraints)
        result = client.deal(request.task, constraints=constraints, dry_run=True, trace="summary")
        result.provider_metadata["tool_calls"] = [
            {
                "name": "run_sql",
                "arguments": {"query": "SELECT row FROM cases", "api_key": "do-not-store"},
                "result": {"row": 1},
            }
        ]
        client.traces.write(
            project=client.config.project.name,
            request=request,
            result=result,
            dry_run=True,
            trace_level="summary",
        )
        records[store_content] = client.traces.read(result.trace.trace_id)

    omitted_call = records[False]["result"]["provider_metadata"]["tool_calls"][0]
    stored_call = records[True]["result"]["provider_metadata"]["tool_calls"][0]
    assert "SELECT" not in json.dumps(records[False])
    assert "row" not in json.dumps(records[False])
    assert "arguments" not in omitted_call
    assert "result" not in omitted_call
    assert stored_call["arguments"] == {"api_key": "[redacted]", "query": "SELECT row FROM cases"}
    assert stored_call["result"] == {"row": 1}


def test_trace_store_jsonable_handles_exotic_types():
    class CustomPayload:
        def to_dict(self):
            return {"items": ("one", {"two"})}

    converted = _jsonable({"set": {"value"}, "tuple": (1, 2), "custom": CustomPayload()})

    assert converted == {
        "set": "{'value'}",
        "tuple": [1, 2],
        "custom": {"items": ["one", "{'two'}"]},
    }
    json.dumps(converted)


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
    ("openai", SYNTHETIC_OPENAI_API_KEY),
    ("anthropic", SYNTHETIC_ANTHROPIC_API_KEY),
    ("google", SYNTHETIC_GOOGLE_API_KEY),
    ("ollama", "ollama_live_abcdefghijklmnopqrstuvwxyz012345"),
    ("openrouter", "sk-or-v1-" + "ab" * 32),
    ("nan", "nan_live_abcdefghijklmnopqrstuvwxyz012345"),
    ("aws", "AKIAIOSFODNN7EXAMPLE"),
    ("bearer", SYNTHETIC_BEARER_TOKEN),
]


@pytest.mark.parametrize(("provider", "secret"), PROVIDER_SECRETS)
def test_central_redactor_covers_supported_provider_key_formats(provider: str, secret: str) -> None:
    redacted = redact_text(f"{provider} credential {secret} stays hidden")
    assert secret not in redacted
    assert "[redacted]" in redacted


def test_project_audit_report_uses_central_redactor() -> None:
    secret = SYNTHETIC_GOOGLE_API_KEY
    report_error = _canary_error(
        "provider.generate",
        "generate",
        "google:gemini-2.5-flash",
        CrupierProviderUnavailableError(secret),
    )
    assert secret not in report_error["error"]
    assert report_error["error"] == "[redacted]"


@pytest.mark.parametrize(
    "secret",
    [
        "https://user:pass@host.example/path",
        "X-Api-Key: prose-secret-value",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue",
    ],
)
def test_central_redactor_covers_url_credentials_header_values_and_jwt(secret: str) -> None:
    redacted = redact_text(f"upstream returned {secret} in diagnostics")
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
        raise CrupierProviderUnavailableError(f"upstream rejected {SYNTHETIC_GOOGLE_API_KEY}")


def test_trace_feedback_eval_and_experiment_use_same_redactor(tmp_path) -> None:
    secret = SYNTHETIC_GOOGLE_API_KEY
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

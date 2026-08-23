import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import crupier.cli as cli
from crupier.config import CrupierConfig, ProviderSettings
from crupier.errors import CrupierConfigError, CrupierError
from crupier.models import CapabilityCard, ModelRef
from crupier.project_audit import _iter_source_files


def test_config_free_handoff_rejects_online_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli.Crupier,
        "from_project",
        lambda _project: (_ for _ in ()).throw(CrupierConfigError("missing")),
    )
    args = NS(
        project=tmp_path,
        real=True,
        production=False,
        dataset=None,
        provider=None,
        orchestrator_mode=None,
        paths=None,
        max_files=1,
        all=False,
        write_report=False,
        json=False,
    )

    with pytest.raises(CrupierError, match="Config-free handoff only supports offline"):
        cli.cmd_adopt_handoff(args)


def test_adopt_project_name_falls_back_after_unreadable_pyproject(
    monkeypatch, tmp_path
):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\n', encoding="utf-8")
    original_read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == pyproject:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)

    assert cli._adopt_project_name(tmp_path) == tmp_path.name


def test_code_comments_plain_output_lists_written_reports(monkeypatch, capsys):
    monkeypatch.setattr(cli, "scan_code_comments", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        cli, "write_code_comments_report", lambda *_args, **_kwargs: [Path("report.md")]
    )
    monkeypatch.setattr(
        cli,
        "summarize_code_comment_reviews",
        lambda *_args, **_kwargs: NS(reviewed_count=0, pending_count=0),
    )
    args = NS(
        project=".",
        paths=None,
        max_files=1,
        write_report=True,
        write_review_comments=False,
        write_sarif=False,
        write_decisions_template=False,
        import_decisions=None,
        ack_reviewed=False,
        reviewer_hash=None,
        note="",
        json=False,
    )

    assert cli.cmd_code_comments(args) == 0
    assert "written_report: report.md" in capsys.readouterr().out


def test_verify_provider_names_defaults_to_enabled_real_providers():
    config = CrupierConfig.from_dict(
        {"providers": {"anthropic": {"enabled": True}, "ollama": {"enabled": False}}}
    )

    assert cli._verify_provider_names(
        config, requested=None, include_openai_baseline=False
    ) == ["anthropic"]


def test_verify_provider_reports_configuration_blockers(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    missing = NS(
        config=CrupierConfig.from_dict({}),
        adapters={},
    )

    item = cli._verify_provider(missing, "anthropic", run_smoke=True, all_models=False)

    assert item["status"] == "blocked"
    assert any("not configured" in issue for issue in item["issues"])
    assert any("No adapter" in issue for issue in item["issues"])
    assert any("No allowed models" in issue for issue in item["issues"])

    disabled_config = CrupierConfig.from_dict(
        {"providers": {"ollama": {"enabled": False}}}
    )
    disabled = cli._verify_provider(
        NS(config=disabled_config, adapters={}),
        "ollama",
        run_smoke=True,
        all_models=False,
    )
    assert any("disabled" in issue for issue in disabled["issues"])


def test_verify_provider_keeps_reporting_after_boundary_failures(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = CrupierConfig.from_dict(
        {
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:test"]},
        }
    )
    client = NS(
        config=config,
        adapters={"openai": object()},
        models=NS(
            discover=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("secret"))
        ),
        capabilities=NS(
            readiness=lambda _models: (_ for _ in ()).throw(RuntimeError("down"))
        ),
    )

    item = cli._verify_provider(client, "openai", run_smoke=False, all_models=False)

    assert item["smoke_skipped"] is True
    assert any("Discovery failed" in issue for issue in item["issues"])
    assert any("Readiness check failed" in issue for issue in item["issues"])
    assert item["status"] == "failed"


def test_smoke_checks_skip_models_without_public_execution_facade():
    card = CapabilityCard(
        model_ref=ModelRef.parse("openai:audio-test"),
        last_updated="now",
        model_kind="audio",
    )
    client = NS(registry=NS(get=lambda _model_ref: card))

    assert cli._run_smoke_checks(client, ["openai:audio-test"]) == [
        {
            "ok": True,
            "model": "openai:audio-test",
            "provider": "openai",
            "kind": "audio",
            "skipped": True,
            "reason": "No public execution facade exists for this model kind.",
        }
    ]


def test_operation_smoke_reports_probe_exception():
    client = NS(
        capabilities=NS(
            probe=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("offline")
            )
        )
    )

    result = cli._run_operation_smoke(client, "nan:rerank", "reranker")

    assert result["ok"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["error"] == "offline"


def test_inference_provider_auth_option_controls_env_requirement(monkeypatch):
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)

    required = cli._provider_env_status(
        ProviderSettings(env_key="INFERENCE_API_KEY", options={"auth": "bearer"}),
        "inference",
    )
    optional = cli._provider_env_status(
        ProviderSettings(env_key="INFERENCE_API_KEY", options={"auth": "none"}),
        "inference",
    )

    assert required["required"] is True
    assert optional["required"] is False


def test_source_iteration_ignores_is_file_and_stat_races(tmp_path, monkeypatch):
    source = tmp_path / "app.py"
    source.write_text("print('x')", encoding="utf-8")
    original_is_file = Path.is_file

    def failing_is_file(path):
        if path == source:
            raise OSError("gone")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", failing_is_file)
    assert (
        list(_iter_source_files(tmp_path, paths=None, max_files=2, max_file_size=100))
        == []
    )

    monkeypatch.setattr(Path, "is_file", lambda path: path == source)
    original_stat = Path.stat

    def failing_stat(path, *args, **kwargs):
        if path == source:
            raise OSError("gone")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    assert (
        list(_iter_source_files(tmp_path, paths=None, max_files=2, max_file_size=100))
        == []
    )


def test_cli_module_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["crupier", "--help"])
    monkeypatch.delitem(sys.modules, "crupier.cli")

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("crupier.cli", run_name="__main__")

    assert exc_info.value.code == 0

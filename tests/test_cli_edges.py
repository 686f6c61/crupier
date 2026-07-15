import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import crupier.cli as cli_module
from crupier.adapters import ProviderModel
from crupier.cli import (
    _adopt_project_name,
    _capability_probe_model_refs,
    _cli_constraints,
    _compare_variants,
    _comparison_dry_run,
    _decode_env_double_quoted,
    _filter_model_cards,
    _load_env_file,
    _ollama_cloud_host,
    _parse_env_file_line,
    _parse_input,
    _parse_response_schema,
    _print_adoption_handoff,
    _print_adoption_patch_report,
    _print_adoption_plan,
    _print_audit_report,
    _print_compare_dataset_report,
    _print_compare_history_report,
    _print_compare_report,
    _print_eval_report,
    _print_human_review_packet,
    _print_probe_report,
    _print_project_doctor,
    _print_readiness_report,
    _print_release_check_report,
    _print_update_report,
    _print_verify_report,
    _provider_verify_status,
    _read_feedback_report,
    _route_models_from_record,
    _select_comparison_from_report,
    _select_variant_from_comparison,
    _strip_env_inline_comment,
    cmd_adopt_package,
    cmd_capabilities_probe,
    cmd_capabilities_readiness,
    cmd_code_comments,
    cmd_feedback_apply,
    cmd_feedback_import_decisions,
    cmd_feedback_record,
    cmd_feedback_summary,
    cmd_models_discover,
    cmd_models_list,
    cmd_models_show,
    cmd_orchestrator_show,
    cmd_profiles_list,
    cmd_registry_snapshot_diff,
    cmd_registry_snapshot_list,
    cmd_registry_snapshot_use,
    cmd_serve,
    cmd_smoke,
    cmd_trace_delete,
    cmd_trace_list,
    cmd_trace_replay,
    cmd_trace_show,
)
from crupier.config import CrupierConfig
from crupier.errors import CrupierConfigError, CrupierError
from crupier.models import CapabilityCard, ModelRef
from crupier.registry import ModelRegistry


def test_env_file_parser_supports_exports_quotes_comments_and_precedence(tmp_path, monkeypatch):
    env = tmp_path / "dev.env"
    env.write_text(
        "# comment\nexport NEW_KEY=plain # ignored\nQUOTED='a # value'\nDOUBLE=\"line\\nnext\\t\\\"ok\\\"\"\nEXISTING=new\nEMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("NEW_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("DOUBLE", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    monkeypatch.setenv("EXISTING", "old")

    loaded = _load_env_file("dev.env", project=str(tmp_path))

    assert loaded == {
        "NEW_KEY": "loaded",
        "QUOTED": "loaded",
        "DOUBLE": "loaded",
        "EXISTING": "already-set",
        "EMPTY": "loaded",
    }
    assert str(Path("dev.env")) not in loaded
    assert _parse_env_file_line("   ", line_number=1) is None
    assert _decode_env_double_quoted(r"a\\b\r\t\q") == "a\\b\r\tq"
    assert _strip_env_inline_comment("value#literal") == "value#literal"
    assert _strip_env_inline_comment("value # comment") == "value"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("INVALID", "expected KEY=value"),
        ("BAD-NAME=x", "Invalid environment variable name"),
        ('KEY="unterminated', "Invalid quoted"),
    ],
)
def test_env_file_parser_rejects_malformed_lines(line, message):
    with pytest.raises(CrupierConfigError, match=message):
        _parse_env_file_line(line, line_number=7)


def test_env_file_loader_rejects_missing_file(tmp_path):
    with pytest.raises(CrupierConfigError, match="not found"):
        _load_env_file("missing.env", project=str(tmp_path))


def test_print_update_probe_readiness_and_release_reports(capsys):
    update = NS(
        dry_run=False,
        requires_confirmation=True,
        diff={
            "added": ["openai:new"],
            "removed": ["openai:old"],
            "changed": [{"model": "openai:x", "fields": ["pricing"]}],
            "unchanged": 2,
        },
        added_models=[],
        removed_models=[],
        unchanged_models=[],
        model_states=[{"states": ["allowed", "locked"]}, {"states": ["allowed"]}],
        written_files=["cards.json"],
        warnings=["review change"],
    )
    probe_result = NS(
        status="failed",
        model="openai:x",
        probe="tools",
        latency_ms=12,
        error="bad schema",
    )
    probe = NS(
        dry_run=False,
        applied=True,
        summary=lambda: {"failed": 1},
        results=[probe_result],
        written_files=["probe.json"],
        warnings=["warning"],
    )
    readiness = NS(
        strict=True,
        summary=lambda: {"failed": 1},
        items=[
            NS(
                status="failed",
                model="openai:x",
                missing_probes=["tools"],
                inferred_probes=["streaming"],
                failed_probes=["structured_output"],
            )
        ],
    )
    release = NS(
        ok=False,
        project="crupier",
        version="0.4.0",
        summary={"fail": 1},
        checks=[NS(status="fail", id="coverage", summary="below gate", actions=["add tests"])],
        build={"ok": False, "wheel_count": 0},
    )

    _print_update_report(update)
    _print_probe_report(probe)
    _print_readiness_report(readiness)
    _print_release_check_report(release)

    output = capsys.readouterr().out
    assert "requires_confirmation: true" in output
    assert "latency_ms=12" in output
    assert "missing=tools" in output
    assert "action: add tests" in output
    assert "build: failed wheels=0" in output


def test_print_eval_and_compare_reports_cover_all_optional_metrics(capsys):
    eval_report = NS(
        ok=False,
        name="routes",
        orchestrator_mode="model",
        passed=0,
        total=1,
        results=[
            NS(
                status="fail",
                id="case-1",
                strategy="single",
                models=["openai:a"],
                failed_checks=["contains"],
                error="provider failed",
            )
        ],
        written_path="eval.json",
    )
    compare = NS(
        dry_run=False,
        passed=1,
        total=2,
        winner="quality",
        recommendation="use quality",
        variants=[
            NS(
                status="pass",
                name="quality",
                strategy="single",
                models=["openai:a"],
                estimated_cost_usd=0.1,
                actual_cost_usd=0.2,
                latency_ms=10,
                estimated_latency_ms=None,
                failed_checks=[],
                error=None,
                output_preview="answer",
                human_questions=["Is it correct?"],
            ),
            NS(
                status="fail",
                name="cheap",
                strategy=None,
                models=[],
                estimated_cost_usd=0.01,
                actual_cost_usd=None,
                latency_ms=None,
                estimated_latency_ms=5,
                failed_checks=["quality"],
                error="low quality",
                output_preview=None,
                human_questions=[],
            ),
        ],
        written_path="compare.json",
    )

    _print_eval_report(eval_report)
    _print_compare_report(compare)

    output = capsys.readouterr().out
    assert "failed: contains" in output
    assert "actual_cost=0.20000000" in output
    assert "est_latency_ms=5" in output
    assert "human_check: Is it correct?" in output


def _score(**overrides):
    values = {
        "model": "openai:a",
        "mode": "agentic",
        "appearances": 3,
        "passed": 2,
        "wins": 1,
        "score_key": "eval:agentic",
        "score_delta": 1.5,
        "avg_actual_cost_usd": 0.2,
        "avg_estimated_cost_usd": None,
        "avg_latency_ms": 30,
        "avg_estimated_latency_ms": None,
        "runs": 2,
        "confidence": 0.8,
        "trend": "up",
    }
    values.update(overrides)
    return NS(**values)


def test_print_dataset_and_history_reports_cover_apply_gates(capsys):
    apply_report = {
        "min_count": 2,
        "min_confidence": 0.7,
        "updated": [{"model": "openai:a", "score_key": "eval:a", "new_score": 2}],
        "skipped": [{"model": "openai:b", "reason": "low confidence"}],
    }
    dataset = NS(
        dry_run=True,
        name="dataset",
        passed_cases=1,
        total_cases=2,
        cases=[NS(ok=True, id="case", winner="quality")],
        model_scores=[
            _score(),
            _score(
                model="openai:b",
                avg_actual_cost_usd=None,
                avg_estimated_cost_usd=0.1,
                avg_latency_ms=None,
                avg_estimated_latency_ms=20,
            ),
        ],
        apply_report=apply_report,
        history_path="history.jsonl",
        written_path="dataset.json",
    )
    history = NS(
        total_runs=2,
        last_run_at="2026-07-15",
        model_scores=dataset.model_scores,
        apply_report=apply_report,
        warnings=["small sample"],
    )

    _print_compare_dataset_report(dataset)
    _print_compare_history_report(history)

    output = capsys.readouterr().out
    assert "avg_actual_cost=0.20000000" in output
    assert "avg_est_cost=0.10000000" in output
    assert "updated\topenai:a" in output
    assert "skipped\topenai:b" in output
    assert "warning: small sample" in output


def _comments(count=1):
    return [NS(priority=1, file=f"app{i}.py", line=i + 1, title="Review call") for i in range(count)]


def test_print_human_review_and_audit_reports(capsys):
    packet = NS(
        ok=True,
        source_path="compare.json",
        source_type="compare",
        total_items=1,
        recommended_items=1,
        dry_run=True,
        warnings=["dry run source", "check output"],
        items=[
            NS(
                id="item-1",
                recommended=True,
                status="pending",
                variant="quality",
                models=["openai:a"],
                human_questions=["Approve?"],
                feedback_commands={"approve": "crupier feedback record"},
            )
        ],
        written_files=["review.md"],
    )
    audit = NS(
        ok=False,
        summary={"fail": 1},
        checks=[NS(status="fail", id="risk", summary="risk found", actions=["review"])],
        route_reviews=[
            NS(
                status="review",
                id="route-1",
                strategy="fusion",
                models=["openai:a", "anthropic:b"],
                human_questions=["Useful?"],
            )
        ],
        real_canaries=[{"ok": False, "id": "canary", "latency_ms": 100, "error": "timeout"}],
        code_comments=_comments(21),
        written_files=["audit.json"],
    )

    _print_human_review_packet(packet)
    _print_audit_report(audit)

    output = capsys.readouterr().out
    assert "warning: dry-run compare report" in output
    assert "approve: crupier feedback record" in output
    assert "real_canaries:" in output
    assert "... 1 more" in output


def _adoption_plan():
    comments = _comments(21)
    option = NS(
        status="recommended",
        path="compat_client",
        score=9,
        summary="smallest integration",
        actions=["install", "configure", "test", "extra"],
        risks=["compatibility", "latency", "extra"],
    )
    return NS(
        ready=False,
        recommended_path="compat_client",
        confidence="high",
        blockers=["human review"],
        options=[option],
        checklist=["configure", "test"],
        code_comments=comments,
        warnings=["review required"],
        written_files=["plan.json"],
    )


def test_print_adoption_reports_cover_artifacts_and_human_gates(capsys):
    plan = _adoption_plan()
    patch = NS(
        status="ready",
        title="Replace client",
        summary="Use Crupier compatibility client",
        commands=["pip install crupier"],
        notes=["review environment"],
        diff="- old\n+ new\n",
    )
    patch_report = NS(
        ready=False,
        adoption_path="compat_client",
        blockers=["review"],
        patches=[patch],
        warnings=["manual merge"],
        written_files=["patches.md"],
    )
    doctor = NS(
        status="blocked",
        readiness_mode="production",
        recommended_path="compat_client",
        confidence="high",
        summary={"fail": 1},
        gates=[NS(status="fail", id="human", summary="missing", actions=["review"] * 5)],
        adoption_plan=plan,
        patch_report=patch_report,
        feedback_summary={"count": 2},
        applied_feedback_summary={"applied_count": 1, "count": 2},
        adoption_signoff_summary={"status": "pending"},
        written_files=["doctor.json"],
    )
    handoff = NS(
        status="blocked",
        doctor=doctor,
        required_human_actions=["review code"],
        suggested_commands=["crupier adopt signoff"],
        artifacts={"reports": ["a", "b", "c", "d"]},
        written_files=["handoff.json"],
    )

    _print_project_doctor(doctor)
    _print_adoption_handoff(handoff)
    _print_adoption_plan(plan)
    _print_adoption_patch_report(patch_report)

    output = capsys.readouterr().out
    assert "human_feedback_applied_groups: 1/2" in output
    assert "adoption_signoff: pending" in output
    assert "command: pip install crupier" in output
    assert "... 1 more" in output


def test_print_verify_report_includes_readiness_and_smoke_details(capsys):
    report = {
        "ok": False,
        "providers": ["openai"],
        "summary": {"failed": 1},
        "items": [
            {
                "status": "failed",
                "provider": "openai",
                "allowed_models": ["openai:a"],
                "discovered_count": 3,
                "env": {"key": "OPENAI_API_KEY", "present": True, "required": True},
                "issues": ["probe failed"],
                "readiness": {"summary": {"failed": 1}},
                "smoke": [
                    {
                        "ok": False,
                        "model": "openai:a",
                        "kind": "embedding",
                        "embedding_dimensions": 3,
                        "latency_ms": 20,
                        "error": "bad vector",
                    }
                ],
            }
        ],
    }

    _print_verify_report(report)

    output = capsys.readouterr().out
    assert "OPENAI_API_KEY=set (required)" in output
    assert "dimensions=3" in output
    assert "latency_ms=20" in output
    assert "error=bad vector" in output


def test_model_filter_and_cli_value_helpers():
    cards = [
        CapabilityCard(
            ModelRef.parse("openai:active"),
            "test",
            model_kind="chat",
            routing_hints={"routing_status": "recommended", "production_default": True},
        ),
        CapabilityCard(
            ModelRef.parse("anthropic:old"),
            "test",
            model_kind="chat",
            routing_hints={"routing_status": "deprecated", "lifecycle": "deprecated"},
        ),
    ]
    args = NS(provider="openai", kind="chat", status="recommended", recommended=True, include_deprecated=False)

    assert [item.model_ref.key for item in _filter_model_cards(cards, args)] == ["openai:active"]
    assert _parse_input(None) is None
    assert _parse_input('{"x": 1}') == {"x": 1}
    assert _parse_input("plain") == "plain"
    assert _parse_response_schema('{"type":"object"}') == {"type": "object"}
    with pytest.raises(CrupierError, match="JSON object"):
        _parse_response_schema("[]")

    constraints = _cli_constraints(
        NS(
            max_cost_usd=1.0,
            max_latency_ms=90000,
            max_output_tokens=20,
            force_model="openai:a",
            file_strategy="native",
            store_trace=False,
            store_prompt=True,
            store_response=True,
        )
    )
    assert constraints["store_trace"] is True
    assert constraints["max_latency_ms"] == 90000
    assert constraints["store_prompt"] is True
    assert constraints["store_response"] is True


def test_compare_variant_and_route_record_helpers():
    variants = _compare_variants(
        argparse.Namespace(
            model=["claude:claude-opus-4-8"],
            variant=['{"name":"custom","model":"openai:gpt-5.5","constraints":{"temperature":0}}'],
        )
    )

    assert [item.name for item in variants] == ["anthropic:claude-opus-4-8", "custom"]
    assert variants[1].constraints["force_model"] == "openai:gpt-5.5"
    with pytest.raises(CrupierError, match="JSON object"):
        _compare_variants(argparse.Namespace(model=[], variant=["bad"]))
    assert _route_models_from_record(
        {
            "steps": [
                {"model": "openai:a", "models": ["anthropic:b"]},
                {"model": "openai:a"},
            ]
        }
    ) == ["openai:a", "anthropic:b"]


def test_feedback_report_reading_and_selection_errors(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"variants": [{"name": "a", "models": ["openai:a"]}], "winner": "a"}))
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")

    assert _read_feedback_report(tmp_path, "valid.json")["winner"] == "a"
    with pytest.raises(CrupierError, match="not found"):
        _read_feedback_report(tmp_path, "missing.json")
    with pytest.raises(CrupierError, match="not valid JSON"):
        _read_feedback_report(tmp_path, "invalid.json")
    with pytest.raises(CrupierError, match="JSON object"):
        _read_feedback_report(tmp_path, "array.json")

    comparison = {"variants": [{"name": "a", "models": ["openai:a"]}], "winner": "a"}
    assert _select_comparison_from_report(comparison, case_id=None) == (comparison, None)
    assert _select_variant_from_comparison(comparison, variant=None)["name"] == "a"
    assert _select_variant_from_comparison(comparison, variant="openai:a")["name"] == "a"
    with pytest.raises(CrupierError, match="no winner"):
        _select_variant_from_comparison({"variants": []}, variant=None)
    with pytest.raises(CrupierError, match="not found"):
        _select_variant_from_comparison(comparison, variant="missing")
    with pytest.raises(CrupierError, match="Use --case-id"):
        _select_comparison_from_report({"cases": [{}, {}]}, case_id=None)
    with pytest.raises(CrupierError, match="was not found"):
        _select_comparison_from_report({"cases": []}, case_id="missing")
    with pytest.raises(CrupierError, match="variant details"):
        _select_comparison_from_report({"cases": [{"id": "only"}]}, case_id=None)
    assert _comparison_dry_run({"dry_run": False}, {}) is False
    assert _comparison_dry_run({}, {"dry_run": False}) is False


def test_adopt_project_name_uses_package_metadata_then_directory(tmp_path):
    (tmp_path / "package.json").write_text('{"name":"frontend"}', encoding="utf-8")
    assert _adopt_project_name(tmp_path) == "frontend"

    (tmp_path / "package.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "backend"\n', encoding="utf-8")
    assert _adopt_project_name(tmp_path) == "backend"

    (tmp_path / "pyproject.toml").unlink()
    assert _adopt_project_name(tmp_path) == tmp_path.name


def test_provider_status_and_ollama_host_helpers():
    base = {"issues": [], "readiness": {"summary": {}}, "smoke": []}
    assert _provider_verify_status(base, run_smoke=False) == "ready"
    assert _provider_verify_status({**base, "issues": ["x"]}, run_smoke=False) == "failed"
    assert _provider_verify_status({**base, "readiness": {"summary": {"failed": 1}}}, run_smoke=False) == "failed"
    assert _provider_verify_status({**base, "smoke": [{"ok": False}]}, run_smoke=True) == "failed"
    assert (
        _provider_verify_status({**base, "readiness": {"summary": {"needs_probes": 1}}}, run_smoke=False)
        == "needs_probes"
    )
    assert _ollama_cloud_host(None) is False
    assert _ollama_cloud_host("https://ollama.com/api") is True
    assert _ollama_cloud_host("http://localhost:11434") is False


def test_capability_probe_model_refs_filters_provider_and_explicit_models(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "models": {"allow": ["openai:gpt-5.5", "anthropic:claude-opus-4-8"]},
            "routing": {"require_operational_providers": False},
        }
    )
    config.root = tmp_path
    client = NS(registry=ModelRegistry(config))

    assert _capability_probe_model_refs(
        client,
        provider="anthropic",
        explicit=["claude:claude-opus-4-8", "openai:gpt-5.5"],
        all_models=False,
    ) == ["anthropic:claude-opus-4-8"]
    assert _capability_probe_model_refs(client, provider="openai", explicit=None, all_models=False) == [
        "openai:gpt-5.5"
    ]


def test_models_discover_handles_disabled_empty_json_and_text_results(monkeypatch, capsys):
    state = {"models": [], "warnings": []}

    def discover(*, provider, skip_unavailable, warnings):
        warnings.extend(state["warnings"])
        return state["models"]

    client = NS(adapters={}, models=NS(discover=discover))
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)

    assert cmd_models_discover(NS(project=".", provider="openai", json=False)) == 1
    client.adapters = {"openai": object()}
    assert cmd_models_discover(NS(project=".", provider="openai", json=False)) == 0

    state["models"] = [ProviderModel(id="gpt-test", provider="openai", name="GPT Test")]
    state["warnings"] = ["partial discovery"]
    assert cmd_models_discover(NS(project=".", provider="openai", json=False)) == 0
    assert cmd_models_discover(NS(project=".", provider="openai", json=True)) == 0

    captured = capsys.readouterr()
    assert "not enabled" in captured.err
    assert "No models discovered" in captured.out
    assert "name=GPT Test" in captured.out
    assert '"provider": "openai"' in captured.out


def test_smoke_command_continues_after_model_failure_and_prints_output(monkeypatch, capsys):
    config = CrupierConfig.from_dict({})

    def deal(*, constraints, **kwargs):
        assert constraints["disable_thinking"] is True
        if constraints["force_model"].endswith("bad"):
            raise RuntimeError("Bearer abcdefghijklmnop")
        return NS(
            output_text="crupier-ok details",
            latency_ms=12,
            provider_metadata={"calls": [{"model": constraints["force_model"]}]},
        )

    client = NS(config=config, deal=deal)
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    args = NS(
        project=".",
        provider=None,
        model=["openai:good", "openai:bad"],
        all=False,
        show_output=True,
        json=False,
    )

    assert cmd_smoke(args) == 1

    output = capsys.readouterr().out
    assert "ok\topenai:good\tlatency_ms=12" in output
    assert "output: crupier-ok details" in output
    assert "failed\topenai:bad\terror=Bearer [redacted]" in output

    args.json = True
    assert cmd_smoke(args) == 1
    assert '"error_type": "RuntimeError"' in capsys.readouterr().out


def test_smoke_command_rejects_empty_selection(monkeypatch, capsys):
    client = NS(config=CrupierConfig.from_dict({}), deal=lambda **kwargs: None)
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)

    assert cmd_smoke(NS(project=".", provider=None, model=None, all=False, show_output=False, json=False)) == 1
    assert "No allowed models" in capsys.readouterr().err


def test_serve_command_reports_address_cors_interrupt_and_closes(monkeypatch, capsys):
    closed = []

    class FakeServer:
        server_address = (b"127.0.0.1", 9123)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            closed.append(True)

    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: object())
    monkeypatch.setattr(cli_module, "build_openai_compatible_server", lambda **kwargs: FakeServer())
    args = NS(
        project=".",
        host="127.0.0.1",
        port=0,
        no_dry_run=False,
        compat_mode="balanced",
        allow_remote=False,
        cors_origin="http://localhost:3000",
        max_request_bytes=100,
        compat="openai",
    )

    assert cmd_serve(args) == 0

    output = capsys.readouterr().err
    assert "http://127.0.0.1:9123/v1" in output
    assert "Browser CORS enabled" in output
    assert "stopped" in output
    assert closed == [True]


def test_serve_command_supports_non_tuple_server_addresses(monkeypatch):
    class FakeServer:
        server_address = b"unix-socket"

        def serve_forever(self):
            return None

        def server_close(self):
            return None

    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: object())
    monkeypatch.setattr(cli_module, "build_openai_compatible_server", lambda **kwargs: FakeServer())
    args = NS(
        project=".",
        host="127.0.0.1",
        port=8787,
        no_dry_run=False,
        compat_mode="balanced",
        allow_remote=False,
        cors_origin=None,
        max_request_bytes=100,
        compat="openai",
    )

    assert cmd_serve(args) == 0


def test_feedback_commands_cover_text_output_apply_and_validation(monkeypatch, capsys, tmp_path):
    record = NS(
        feedback_id="fb_1",
        models=["openai:a"],
        mode="agentic",
        strategy="single",
        rating=5,
        verdict="accepted",
        to_dict=lambda: {"feedback_id": "fb_1"},
    )
    feedback = NS(
        record=lambda **kwargs: record,
        summary=lambda **kwargs: {
            "count": 2,
            "groups": [
                {
                    "status": "ready",
                    "model": "openai:a",
                    "mode": "agentic",
                    "count": 2,
                    "avg_rating": 4.5,
                    "score_delta": 1.0,
                    "top_tags": [{"tag": "quality", "count": 2}],
                }
            ],
        },
        apply_to_registry=lambda *args, **kwargs: {
            "dry_run": False,
            "updated": [
                {
                    "model": "openai:a",
                    "mode": "agentic",
                    "score_key": "human:agentic",
                    "new_score": 2,
                    "count": 2,
                }
            ],
            "skipped": [{"model": "openai:b", "reason": "few samples"}],
            "written_files": ["card.json"],
        },
    )
    client = NS(
        feedback=feedback,
        config=NS(project=NS(name="test")),
        traces=object(),
        registry=object(),
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    record_args = NS(
        project=str(tmp_path),
        trace_id=None,
        compare_report=None,
        variant=None,
        case_id=None,
        allow_dry_run_source=False,
        model=["openai:a"],
        mode="agentic",
        strategy="single",
        rating=5,
        verdict="accepted",
        tag=["quality"],
        note="good",
        reviewer_hash="reviewer",
        json=False,
    )

    assert cmd_feedback_record(record_args) == 0
    assert cmd_feedback_summary(NS(project=str(tmp_path), model=None, mode=None, json=False)) == 0
    assert cmd_feedback_apply(NS(project=str(tmp_path), min_count=2, dry_run=False, json=False)) == 0
    record_args.trace_id = "trc_1"
    record_args.compare_report = "compare.json"
    with pytest.raises(CrupierError, match="either"):
        cmd_feedback_record(record_args)

    output = capsys.readouterr().out
    assert "feedback_recorded: fb_1" in output
    assert "tags: quality:2" in output
    assert "updated\topenai:a" in output
    assert "skipped\topenai:b" in output


def test_feedback_summary_handles_empty_store(monkeypatch, capsys):
    client = NS(feedback=NS(summary=lambda **kwargs: {"count": 0, "groups": []}))
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)

    assert cmd_feedback_summary(NS(project=".", model=None, mode=None, json=False)) == 0
    assert "No human feedback" in capsys.readouterr().out


def test_feedback_import_decisions_outputs_applied_scores(monkeypatch, capsys, tmp_path):
    result = NS(
        decision_path="decisions.json",
        imported=2,
        skipped=["one"],
        to_dict=lambda: {"imported": 2, "skipped": ["one"]},
    )
    feedback = NS(
        apply_to_registry=lambda *args, **kwargs: {
            "updated": [{"model": "openai:a", "score_key": "human:a", "new_score": 1}],
            "skipped": [],
            "written_files": [],
        }
    )
    client = NS(feedback=feedback, config=NS(project=NS(name="test")), registry=object())
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    monkeypatch.setattr(cli_module, "import_human_decisions", lambda *args, **kwargs: result)
    args = NS(
        project=str(tmp_path),
        decisions="decisions.json",
        dry_run=False,
        reviewer_hash="reviewer",
        allow_dry_run_source=False,
        apply_to_registry=True,
        min_count=1,
        json=False,
    )

    assert cmd_feedback_import_decisions(args) == 0
    assert "applied_scores: 1" in capsys.readouterr().out

    args.dry_run = True
    args.json = True
    assert cmd_feedback_import_decisions(args) == 0
    assert '"dry_run": true' in capsys.readouterr().out


def test_trace_commands_cover_empty_list_text_show_replay_and_delete(monkeypatch, capsys):
    route = NS(strategy="single", model_summary="openai:a")
    replay = NS(output_text="replayed", route=route, to_dict=lambda **kwargs: {"output_text": "replayed"})
    ref = NS(
        trace_id="trc_1",
        created_at="now",
        strategy="single",
        models=["openai:a"],
        replayable=True,
        summary="task",
        to_dict=lambda: {"trace_id": "trc_1"},
    )
    state = {"refs": []}
    traces = NS(
        list=lambda: state["refs"],
        read=lambda trace_id: {
            "trace_id": trace_id,
            "created_at": "now",
            "project": "test",
            "replayable": True,
            "request": {"summary": "task"},
            "result": {
                "route": {
                    "strategy": "fallback",
                    "steps": [{"role": "fallback", "models": ["openai:a", "anthropic:b"]}],
                },
                "cost": {"estimated_usd": 0.1},
            },
            "storage_decision": {"trace": True},
        },
        replay=lambda *args, **kwargs: replay,
        delete=lambda trace_id: Path("trace.json"),
    )
    client = NS(traces=traces)
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)

    assert cmd_trace_list(NS(project=".", json=False)) == 0
    state["refs"] = [ref]
    assert cmd_trace_list(NS(project=".", json=False)) == 0
    assert cmd_trace_show(NS(project=".", trace_id="trc_1", json=False)) == 0
    assert cmd_trace_replay(NS(project=".", trace_id="trc_1", trace="summary", no_dry_run=False, json=False)) == 0
    assert cmd_trace_delete(NS(project=".", trace_id="trc_1")) == 0

    output = capsys.readouterr().out
    assert "No stored traces" in output
    assert "trc_1\tnow\tsingle" in output
    assert "models: openai:a, anthropic:b" in output
    assert "route: single | openai:a" in output
    assert "Deleted trace.json" in output


def test_registry_snapshot_commands_cover_text_json_and_empty_states(monkeypatch, capsys):
    state = {"snapshots": []}
    registry = NS(
        snapshot_list=lambda: state["snapshots"],
        snapshot_diff=lambda left, right: {
            "left": {"name": left, "card_count": 1},
            "right": {"name": right, "card_count": 2},
            "added": ["openai:new"],
            "removed": ["openai:old"],
            "changed": [{"model": "openai:x", "fields": ["pricing"]}],
            "unchanged": 0,
        },
        snapshot_use=lambda name, restore_allowlist: {
            "snapshot": name,
            "restored_models": ["openai:a"],
            "written_files": ["a.json"],
            "removed_files": ["old.json"],
            "allowlist_restored": True,
        },
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: NS(registry=registry))

    assert cmd_registry_snapshot_list(NS(project=".", json=False)) == 0
    state["snapshots"] = [{"name": "base", "card_count": 1, "allowlist_count": 1, "created_at": "now"}]
    assert cmd_registry_snapshot_list(NS(project=".", json=True)) == 0
    assert cmd_registry_snapshot_diff(NS(project=".", left="base", right="current", json=False)) == 0
    assert cmd_registry_snapshot_use(NS(project=".", name="base", restore_allowlist=True, json=False)) == 0

    output = capsys.readouterr().out
    assert "No registry snapshots" in output
    assert "openai:new" in output
    assert "allowlist: restored" in output


def test_capability_commands_handle_empty_selection_and_json_reports(monkeypatch, capsys):
    state = {"cards": []}
    registry = NS(list=lambda allowed_only: state["cards"])
    probe = NS(to_dict=lambda: {"results": []})
    readiness = NS(to_dict=lambda: {"items": []})
    client = NS(
        registry=registry,
        capabilities=NS(
            probe=lambda *args, **kwargs: probe,
            readiness=lambda *args, **kwargs: readiness,
        ),
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    base = {
        "project": ".",
        "provider": None,
        "model": None,
        "all": False,
        "json": False,
    }

    assert cmd_capabilities_probe(NS(**base, probe=None, apply=False, dry_run=True)) == 1
    assert cmd_capabilities_readiness(NS(**base, strict=False)) == 1

    state["cards"] = [CapabilityCard(ModelRef.parse("openai:a"), "test")]
    assert cmd_capabilities_probe(NS(**{**base, "json": True}, probe=None, apply=False, dry_run=True)) == 0
    assert cmd_capabilities_readiness(NS(**{**base, "json": True}, strict=True)) == 0
    captured = capsys.readouterr()
    assert "No models found" in captured.err
    assert '"results": []' in captured.out
    assert '"items": []' in captured.out


def test_models_list_show_profiles_and_orchestrator_json_paths(monkeypatch, capsys, tmp_path):
    card = CapabilityCard(
        ModelRef.parse("openai:a"),
        "test",
        routing_hints={"routing_status": "recommended", "production_default": True},
    )
    client = NS(
        models=NS(list=lambda allowed_only: [card], get=lambda model: card),
        registry=NS(model_states=lambda models: [{"model": "openai:a", "states": ["allowed"]}]),
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    filter_args = {
        "project": ".",
        "all": False,
        "provider": None,
        "kind": None,
        "status": None,
        "recommended": False,
        "include_deprecated": False,
        "json": True,
    }

    assert cmd_models_list(NS(**filter_args)) == 0
    assert cmd_models_show(NS(project=".", model="openai:a", json=True)) == 0

    cli_module.write_default_project(tmp_path)
    assert cmd_profiles_list(NS(project=str(tmp_path), json=True)) == 0
    assert cmd_orchestrator_show(NS(project=str(tmp_path), json=True)) == 0
    output = capsys.readouterr().out
    assert '"registry_state"' in output
    assert '"agentic"' in output
    assert '"candidate_limit"' in output


def test_code_comments_text_ack_import_and_conflict(monkeypatch, capsys):
    comment = NS(priority=1, file="app.py", line=3, title="Review", body="Inspect call", to_dict=lambda: {"file": "app.py"})
    state = {"comments": [comment]}
    monkeypatch.setattr(cli_module, "scan_code_comments", lambda *args, **kwargs: state["comments"])
    monkeypatch.setattr(
        cli_module,
        "summarize_code_comment_reviews",
        lambda *args, **kwargs: NS(reviewed_count=1, pending_count=0, to_dict=lambda: {}),
    )
    monkeypatch.setattr(
        cli_module,
        "acknowledge_code_comments",
        lambda *args, **kwargs: {"review_id": "rev_1", "comment_count": 1, "path": "review.json"},
    )
    monkeypatch.setattr(
        cli_module,
        "import_code_comment_decisions",
        lambda *args, **kwargs: {
            "review_id": "rev_2",
            "comment_count": 1,
            "pending_decision_count": 0,
            "path": "review2.json",
        },
    )
    base = {
        "project": ".",
        "paths": [],
        "max_files": 10,
        "write_report": False,
        "write_review_comments": False,
        "write_sarif": False,
        "write_decisions_template": False,
        "reviewer_hash": "reviewer",
        "note": "reviewed",
        "json": False,
    }

    assert cmd_code_comments(NS(**base, ack_reviewed=False, import_decisions=None)) == 0
    assert cmd_code_comments(NS(**base, ack_reviewed=True, import_decisions=None)) == 0
    assert cmd_code_comments(NS(**base, ack_reviewed=False, import_decisions="decisions.json")) == 0
    state["comments"] = []
    assert cmd_code_comments(NS(**base, ack_reviewed=False, import_decisions=None)) == 0
    with pytest.raises(CrupierError, match="mutually exclusive"):
        cmd_code_comments(NS(**base, ack_reviewed=True, import_decisions="decisions.json"))

    output = capsys.readouterr().out
    assert "Inspect call" in output
    assert "ack_reviewed: rev_1" in output
    assert "imported_decisions: rev_2" in output
    assert "No AI integration hotspots" in output


def test_adopt_package_configured_text_path(monkeypatch, capsys):
    plan = _adoption_plan()
    patch_report = NS(patches=[], to_dict=lambda: {})
    doctor = NS(
        adoption_plan=plan,
        patch_report=patch_report,
        status="blocked",
        ready=False,
        readiness_mode="production",
        recommended_path="compat_client",
        summary={"fail": 1},
        review_contract={"human": True},
    )
    handoff = NS(
        status="blocked",
        ready=False,
        required_human_actions=["review code"],
        suggested_commands=["crupier adopt signoff"],
    )
    client = NS(config=NS(project=NS(name="configured-project")))
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    monkeypatch.setattr(cli_module, "build_project_doctor", lambda *args, **kwargs: doctor)
    monkeypatch.setattr(cli_module, "build_adoption_handoff_from_doctor", lambda *args, **kwargs: handoff)
    monkeypatch.setattr(cli_module, "write_code_comments_report", lambda *args: [Path("comments.json")])
    monkeypatch.setattr(cli_module, "write_code_review_comments", lambda *args: [Path("review.json")])
    monkeypatch.setattr(cli_module, "write_code_comments_sarif", lambda *args: Path("comments.sarif"))
    monkeypatch.setattr(cli_module, "write_code_comment_decision_template", lambda *args: Path("decisions.json"))
    monkeypatch.setattr(cli_module, "write_adoption_patch_report", lambda *args: [Path("patches.json")])
    monkeypatch.setattr(cli_module, "write_project_doctor_report", lambda *args: [Path("doctor.json")])
    monkeypatch.setattr(cli_module, "write_adoption_handoff_report", lambda *args: [Path("handoff.json")])

    def package_index(project, payload):
        payload["artifact_groups"]["package_index"] = ["package.json"]
        payload["written_files"].append("package.json")
        return [Path("package.json")], payload

    monkeypatch.setattr(cli_module, "write_adoption_package_index", package_index)
    args = NS(
        project=".",
        paths=[],
        max_files=10,
        real=False,
        production=False,
        dataset=None,
        provider=None,
        orchestrator_mode=None,
        all=False,
        no_openai_baseline=False,
        json=False,
    )

    assert cmd_adopt_package(args) == 1

    output = capsys.readouterr().out
    assert "adoption_package: blocked" in output
    assert "configured-project" not in output
    assert "human_actions:" in output
    assert "package_index:" in output

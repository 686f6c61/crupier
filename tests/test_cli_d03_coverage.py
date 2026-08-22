import json
from types import SimpleNamespace as NS

import pytest

from crupier import cli
from crupier.config import CrupierConfig
from crupier.errors import CrupierConfigError


def test_cli_models_discover_empty_returns_nonzero(monkeypatch):
    client = NS(adapters={"openai": object()}, models=NS(discover=lambda **kwargs: []))
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)

    assert cli.cmd_models_discover(NS(project=".", provider="openai", json=False)) == 1


def test_cli_models_discover_empty_still_prints_friendly_message(monkeypatch, capsys):
    client = NS(adapters={"openai": object()}, models=NS(discover=lambda **kwargs: []))
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)

    cli.cmd_models_discover(NS(project=".", provider="openai", json=False))

    assert "No models discovered for openai" in capsys.readouterr().out


def test_purge_plain_output(monkeypatch, capsys):
    config = CrupierConfig.from_dict({})
    monkeypatch.setattr(cli.CrupierConfig, "from_toml", lambda project: config)
    monkeypatch.setattr(cli, "TraceStore", lambda *args, **kwargs: NS(last_pruned=2))
    monkeypatch.setattr(cli, "HumanFeedbackStore", lambda *args, **kwargs: NS(last_pruned=3))
    monkeypatch.setattr(cli, "purge_eval_artifacts", lambda *args: 4)

    assert cli.cmd_purge(NS(project=".", json=False)) == 0
    assert "total=9 traces=2 feedback=3 evals=4" in capsys.readouterr().out


@pytest.mark.parametrize("as_json", [False, True])
def test_models_refresh_outputs_report(monkeypatch, capsys, as_json):
    report = NS(
        dry_run=True,
        requires_confirmation=False,
        diff={},
        added_models=[],
        removed_models=[],
        unchanged_models=[],
        model_states=[],
        written_files=[],
        warnings=[],
        to_dict=lambda: {"ok": True},
    )
    client = NS(update=lambda **kwargs: report)
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)

    assert cli.cmd_models_refresh(NS(project=".", dry_run=True, provider=None, json=as_json)) == 0
    assert ("\"ok\": true" if as_json else "update: dry-run") in capsys.readouterr().out


def test_orchestrator_set_requires_setting():
    args = NS(
        mode=None,
        model=None,
        fallback_model=None,
        fallback=None,
        temperature=None,
        require_validated_plan=None,
        max_repairs=None,
        candidate_limit=None,
        allow_prompt_summary_only=None,
    )

    with pytest.raises(CrupierConfigError, match="at least one"):
        cli.cmd_orchestrator_set(args)


def test_feedback_rating_rejects_non_integer(capsys):
    assert cli.main(["feedback", "record", "--rating", "good"]) == 1
    assert "integer from 1 to 5" in capsys.readouterr().err


def test_orchestrator_set_json(monkeypatch, capsys):
    config = CrupierConfig.from_dict({"orchestrator": {"mode": "model"}})
    monkeypatch.setattr(cli, "write_orchestrator_settings", lambda *args, **kwargs: "crupier.toml")
    monkeypatch.setattr(cli.CrupierConfig, "from_toml", lambda project: config)
    args = NS(
        project=".",
        mode="model",
        model=None,
        fallback_model=None,
        fallback=None,
        temperature=None,
        require_validated_plan=None,
        max_repairs=None,
        candidate_limit=None,
        allow_prompt_summary_only=None,
        json=True,
    )

    assert cli.cmd_orchestrator_set(args) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "model"


def test_scoring_suggest_empty(monkeypatch, capsys):
    monkeypatch.setattr(cli.CrupierConfig, "from_toml", lambda project: NS())
    report = NS(applied=False, evidence={"eval_signal_count": 0, "human_feedback_signal_count": 0}, suggestions=[], written_path=None)
    monkeypatch.setattr(cli, "suggest_scoring_from_project", lambda *args, **kwargs: report)

    assert cli.cmd_scoring_suggest(NS(project=".", apply=False, min_samples=3, json=False)) == 0
    assert "No scoring updates suggested." in capsys.readouterr().out


def test_registry_snapshot_json_paths(monkeypatch, capsys):
    registry = NS(
        snapshot_create=lambda *args, **kwargs: {"name": "one"},
        snapshot_diff=lambda *args: {"added": []},
        snapshot_use=lambda *args, **kwargs: {"snapshot": "one"},
    )
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: NS(registry=registry))

    assert cli.cmd_registry_snapshot_create(NS(project=".", name="one", allowed_only=False, json=True)) == 0
    assert cli.cmd_registry_snapshot_diff(NS(project=".", left="one", right="two", json=True)) == 0
    assert cli.cmd_registry_snapshot_use(NS(project=".", name="one", restore_allowlist=False, json=True)) == 0
    assert capsys.readouterr().out.count("{") == 3


def test_profiles_plain_output(monkeypatch, capsys):
    config = NS(profiles={"fast": NS(strategy="single", prefer=["cheap"], options={})})
    monkeypatch.setattr(cli.CrupierConfig, "from_toml", lambda project: config)

    assert cli.cmd_profiles_list(NS(project=".", json=False)) == 0
    assert "fast\tstrategy=single\tprefer=cheap" in capsys.readouterr().out


def test_release_check_plain_output(monkeypatch, capsys):
    report = NS(ok=True, project="demo", version=None, summary={"ok": 1}, checks=[], build=None)
    monkeypatch.setattr(cli, "run_release_checks", lambda *args, **kwargs: report)
    args = NS(
        project=".",
        skip_build=True,
        check_pypi_name=False,
        verify_project_urls=False,
        verify_providers=False,
        strict_public=False,
        json=False,
    )

    assert cli.cmd_release_check(args) == 0
    assert "release_check: ready" in capsys.readouterr().out


def test_eval_command_plain_paths(monkeypatch, capsys):
    report = NS(
        ok=True,
        winner="a",
        written_path=None,
        to_dict=lambda: {"ok": True},
    )
    fake_evals = NS(
        run=lambda **kwargs: report,
        compare=lambda **kwargs: report,
        compare_dataset=lambda **kwargs: report,
        history=lambda **kwargs: report,
    )
    monkeypatch.setattr(cli.CrupierConfig, "from_toml", lambda project: CrupierConfig.from_dict({}))
    monkeypatch.setattr(cli, "Crupier", lambda config: NS(evals=fake_evals))
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: NS(evals=fake_evals), raising=False)
    monkeypatch.setattr(cli, "_print_eval_report", lambda value: print("eval-plain"))
    monkeypatch.setattr(cli, "_print_compare_report", lambda value: print("compare-plain"))
    monkeypatch.setattr(cli, "_print_compare_dataset_report", lambda value: print("dataset-plain"))
    monkeypatch.setattr(cli, "_print_compare_history_report", lambda value: print("history-plain"))
    monkeypatch.setattr(cli, "_compare_variants", lambda args: [])
    monkeypatch.setattr(cli, "_cli_constraints", lambda args: {})
    monkeypatch.setattr(cli, "_parse_response_schema", lambda value: None)

    assert cli.cmd_eval_run(NS(project=".", orchestrator_mode="model", dataset="d", write_report=False, json=False)) == 0
    common = NS(
        project=".",
        task="t",
        input_value=None,
        mode=None,
        strategy=None,
        response_schema=None,
        expect_contains=None,
        no_dry_run=False,
        write_report=False,
        store_content=False,
        dataset="d",
        apply=False,
        min_count=1,
        min_confidence="low",
        record_history=False,
        model=None,
        dry_run=True,
        json=False,
    )
    assert cli.cmd_eval_compare(common) == 0
    assert cli.cmd_eval_compare_dataset(common) == 0
    assert cli.cmd_eval_history(common) == 0
    assert capsys.readouterr().out.splitlines() == ["eval-plain", "compare-plain", "dataset-plain", "history-plain"]


def test_compare_dataset_history_path_output(capsys):
    report = NS(
        dry_run=True,
        name="demo",
        passed_cases=1,
        total_cases=1,
        cases=[],
        model_scores=[],
        history_path="history.jsonl",
        written_path=None,
        apply_report=None,
    )

    cli._print_compare_dataset_report(report)

    assert "history: history.jsonl" in capsys.readouterr().out


def test_compare_history_empty_output(capsys):
    report = NS(total_runs=0, last_run_at=None, model_scores=[], apply_report=None, warnings=[])

    cli._print_compare_history_report(report)

    assert "No compare history found." in capsys.readouterr().out


def test_feedback_record_optional_mode_and_strategy(monkeypatch, capsys):
    record = NS(feedback_id="f1", models=["openai:a"], mode="fast", strategy="single", rating=5, verdict="accept")
    client = NS(
        config=NS(project=NS(name="demo")),
        feedback=NS(record=lambda **kwargs: record),
        traces=NS(),
    )
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)
    args = NS(
        project=".", trace_id=None, compare_report=None, variant=None, case_id=None, allow_dry_run_source=False,
        model=["openai:a"], mode="fast", strategy="single", rating=5, verdict="accept", tag=None, note="",
        reviewer_hash=None, json=False,
    )

    assert cli.cmd_feedback_record(args) == 0
    assert "mode: fast\nstrategy: single" in capsys.readouterr().out


def test_feedback_apply_reports_written_files(monkeypatch, capsys):
    report = {"dry_run": False, "updated": [], "skipped": [], "written_files": ["one.json"]}
    client = NS(feedback=NS(apply_to_registry=lambda *args, **kwargs: report), registry=NS())
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)

    assert cli.cmd_feedback_apply(NS(project=".", min_count=1, dry_run=False, json=False)) == 0
    assert "written_files: 1" in capsys.readouterr().out


def test_feedback_review_plain_path(monkeypatch, capsys):
    packet = NS(ok=True, written_files=[])
    monkeypatch.setattr(cli, "build_human_review_packet", lambda *args, **kwargs: packet)
    monkeypatch.setattr(cli, "_print_human_review_packet", lambda value: print("review-plain"))
    args = NS(
        project=".", compare_report="report.json", case_id=None, variant=None, no_preview=False,
        write_report=False, write_decisions_template=False, reviewer_hash=None, json=False,
    )

    assert cli.cmd_feedback_review(args) == 0
    assert capsys.readouterr().out == "review-plain\n"


def test_feedback_summary_prints_corrupt_diagnostic(monkeypatch, capsys):
    summary = {
        "complete": False,
        "count": 0,
        "groups": [],
        "diagnostics": [{"line": 7, "path": "feedback.jsonl", "error_type": "JSONDecodeError"}],
    }
    client = NS(feedback=NS(summary=lambda **kwargs: summary))
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)

    assert cli.cmd_feedback_summary(NS(project=".", model=None, mode=None, json=False)) == 1
    assert "feedback.jsonl:7 (JSONDecodeError)" in capsys.readouterr().err


def test_feedback_report_defensive_errors(tmp_path):
    dry_run = tmp_path / "dry.json"
    dry_run.write_text(json.dumps({"dry_run": True, "winner": "a", "variants": [{"name": "a", "models": ["openai:a"]}]}))
    no_models = tmp_path / "no-models.json"
    no_models.write_text(json.dumps({"dry_run": False, "winner": "a", "variants": [{"name": "a", "models": []}]}))

    with pytest.raises(cli.CrupierError, match="dry-run"):
        cli._feedback_from_compare_report(tmp_path, report_path=str(dry_run), variant=None, case_id=None)
    with pytest.raises(cli.CrupierError, match="no route models"):
        cli._feedback_from_compare_report(tmp_path, report_path=str(no_models), variant=None, case_id=None)
    with pytest.raises(cli.CrupierError, match="eval compare"):
        cli._select_comparison_from_report({}, case_id=None)
    with pytest.raises(cli.CrupierError, match="does not contain variants"):
        cli._select_variant_from_comparison({}, variant="a")
    with pytest.raises(cli.CrupierError, match="not found"):
        cli._select_variant_from_comparison({"variants": [None], "winner": "a"}, variant=None)


def test_audit_plain_path(monkeypatch, capsys):
    report = NS(ok=True)
    client = NS(audit=NS(run=lambda **kwargs: report))
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: client)
    monkeypatch.setattr(cli, "_print_audit_report", lambda value: print("audit-plain"))
    args = NS(
        project=".", dataset=None, provider=None, no_openai_baseline=False, orchestrator_mode=None,
        real=False, all=False, no_code_comments=False, code_path=None, max_code_files=10,
        write_report=False, json=False,
    )

    assert cli.cmd_audit(args) == 0
    assert capsys.readouterr().out == "audit-plain\n"


@pytest.mark.parametrize("command_name", ["cmd_adopt_doctor", "cmd_adopt_handoff"])
def test_adopt_doctor_and_handoff_plain_paths(monkeypatch, capsys, command_name):
    report = NS(ready=True, doctor=NS(ready=True), written_files=[])
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: NS())
    builder_name = "build_project_doctor" if command_name == "cmd_adopt_doctor" else "build_adoption_handoff"
    printer_name = "_print_project_doctor" if command_name == "cmd_adopt_doctor" else "_print_adoption_handoff"
    monkeypatch.setattr(cli, builder_name, lambda *args, **kwargs: report)
    monkeypatch.setattr(cli, printer_name, lambda value: print("adopt-plain"))
    args = NS(
        project=".", paths=None, max_files=1, dataset=None, provider=None, no_openai_baseline=False,
        orchestrator_mode=None, real=False, all=False, production=False, write_report=False, json=False,
    )

    assert getattr(cli, command_name)(args) == 0
    assert capsys.readouterr().out == "adopt-plain\n"


def test_verify_json_output(monkeypatch, capsys):
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: NS())
    monkeypatch.setattr(cli, "_build_verify_report", lambda *args, **kwargs: {"ok": True})

    assert cli.cmd_verify(NS(project=".", provider=None, no_openai_baseline=False, skip_smoke=True, all=False, json=True)) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_embedding_smoke_reports_missing_adapter():
    result = cli._run_embedding_smoke(NS(adapters={}), "openai:text-embedding-test")

    assert result["ok"] is False
    assert result["error_type"] == "RuntimeError"


def test_adopt_signoff_config_free_plain_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.Crupier, "from_project", lambda project: (_ for _ in ()).throw(CrupierConfigError("missing")))
    monkeypatch.setattr(
        cli,
        "record_adoption_signoff",
        lambda *args, **kwargs: {"verdict": "accepted", "project": "demo", "path": "signoff.json", "handoff": "handoff.json"},
    )
    args = NS(
        project=str(tmp_path), verdict="accepted", reviewer_hash=None, note="", handoff="handoff.json",
        adoption_path=None, json=False,
    )

    assert cli.cmd_adopt_signoff(args) == 0
    assert "handoff: handoff.json" in capsys.readouterr().out


@pytest.mark.parametrize("command_name", ["cmd_adopt_plan", "cmd_adopt_patches"])
def test_adopt_plan_and_patches_plain_paths(monkeypatch, capsys, command_name):
    report = NS(ready=True, written_files=[], to_dict=lambda: {"ready": True})
    builder_name = "build_adoption_plan" if command_name == "cmd_adopt_plan" else "build_adoption_patches"
    printer_name = "_print_adoption_plan" if command_name == "cmd_adopt_plan" else "_print_adoption_patch_report"
    monkeypatch.setattr(cli, builder_name, lambda *args, **kwargs: report)
    monkeypatch.setattr(cli, printer_name, lambda value: print("plain"))
    args = NS(project=".", paths=None, max_files=1, write_report=False, json=False, path=None)

    assert getattr(cli, command_name)(args) == 0
    assert capsys.readouterr().out == "plain\n"

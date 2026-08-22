import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import crupier.project_audit as audit_module
from crupier.config import ProviderSettings
from crupier.errors import CrupierError
from crupier.project_audit import (
    AdoptionHandoffReport,
    AdoptionPatchReport,
    CodeComment,
    CodeCommentReviewSummary,
    DoctorGate,
    ProjectAdoptionPlan,
    ProjectAuditReport,
    ProjectAuditRunner,
    ProjectDoctorReport,
    _compat_client_patch_suggestions,
    _doctor_audit_gate,
    _doctor_eval_history_gate,
    _doctor_real_canary_gate,
    _framework_hints,
    _handoff_actions,
    _iter_source_files,
    _latest_production_decision_template,
    _provider_env_status,
    _read_code_comment_review_records,
    _sarif_level,
    build_adoption_patches,
    format_adoption_handoff_markdown,
    read_adoption_signoffs,
    record_adoption_signoff,
    scan_code_comments,
    summarize_applied_human_feedback,
)


def _minimal_plan() -> ProjectAdoptionPlan:
    return ProjectAdoptionPlan(
        project="demo",
        generated_at="now",
        recommended_path="compat_client",
        confidence="high",
        options=[],
        checklist=[],
    )


def _minimal_doctor(*, comments=None, gates=None) -> ProjectDoctorReport:
    plan = _minimal_plan()
    plan.code_comments = list(comments or [])
    plan.code_comment_review = CodeCommentReviewSummary(
        count=len(plan.code_comments),
        reviewed_count=0,
        pending_count=len(plan.code_comments),
        pending=plan.code_comments,
    )
    return ProjectDoctorReport(
        project="demo",
        generated_at="now",
        readiness_mode="production",
        adoption_plan=plan,
        patch_report=AdoptionPatchReport("demo", "now", "compat_client", []),
        audit_report=ProjectAuditReport("demo", "now", [], []),
        eval_history={},
        feedback_summary={},
        gates=list(gates or []),
    )


def test_project_audit_run_overrides_planner_and_writes_report(tmp_path, monkeypatch):
    original_planner = object()
    config = NS(
        root=tmp_path,
        project=NS(name="demo"),
        orchestrator=NS(mode="local"),
        models=NS(allow=[]),
        logging=NS(store_prompts=False, store_responses=False, redact_secrets=True),
        routing=NS(allow_latest_aliases=False, allow_preview_models=False),
        providers={},
    )
    eval_report = NS(ok=True, passed=1, total=1, failed=0, to_dict=lambda: {"ok": True})
    client = NS(config=config, planner=original_planner, adapters={}, evals=NS(run=lambda **_: eval_report))
    runner = ProjectAuditRunner(client)
    monkeypatch.setattr(runner, "_provider_checks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "_route_reviews", list)
    monkeypatch.setattr(audit_module, "ModelOrchestrator", lambda *_args, **_kwargs: "orchestrator")
    monkeypatch.setattr(audit_module, "RoutePlanner", lambda *_args, **_kwargs: "replacement")
    monkeypatch.setattr(
        audit_module,
        "write_project_audit_report",
        lambda root, report: [root / "audit.json", root / "audit.md"],
    )

    report = runner.run(
        orchestrator_mode="model",
        include_openai_baseline=False,
        include_code_comments=False,
        write_report=True,
    )

    assert report.written_files == [str(tmp_path / "audit.json"), str(tmp_path / "audit.md")]
    assert client.planner is original_planner
    assert client.config.orchestrator.mode == "local"


def test_project_audit_provider_and_route_failures_are_reported(monkeypatch):
    disabled = ProviderSettings(enabled=False, env_key="ANTHROPIC_API_KEY")
    config = NS(
        providers={"anthropic": disabled},
        models=NS(allow=[]),
    )
    readiness = NS(readiness=lambda _refs: (_ for _ in ()).throw(RuntimeError("secret-token")))
    client = NS(config=config, adapters={}, capabilities=readiness)
    runner = ProjectAuditRunner(client)

    assert runner._provider_names(["anthropic"], include_openai_baseline=True) == ["openai", "anthropic"]
    missing, disabled_check = runner._provider_checks(["openai", "anthropic"], real=True)
    assert "not configured" in missing.evidence["issues"][0]
    assert any("disabled" in issue for issue in disabled_check.evidence["issues"])
    assert any("adapter is unavailable" in issue for issue in disabled_check.evidence["issues"])
    assert any("No allowed" in issue for issue in disabled_check.evidence["issues"])

    config.models.allow = ["anthropic:claude-3-5-sonnet-20241022"]
    failed_readiness = runner._provider_checks(["anthropic"], real=True)
    assert failed_readiness[-1].id == "provider_anthropic_readiness"
    assert failed_readiness[-1].status == "fail"

    plans = iter(
        [
            NS(route=None),
            NS(
                route=NS(
                    reason="",
                    selection_scores=[],
                    strategy="single",
                    models=[],
                    estimated_cost=NS(estimated_usd=0),
                )
            ),
            RuntimeError("route failed"),
            NS(
                route=NS(
                    reason="ok",
                    selection_scores=[1],
                    strategy="single",
                    models=["anthropic:x"],
                    estimated_cost=NS(estimated_usd=0),
                )
            ),
        ]
    )

    def deal(**_kwargs):
        value = next(plans)
        if isinstance(value, Exception):
            raise value
        return value

    runner.client.deal = deal
    reviews = runner._route_reviews()
    assert reviews[0].error == "No route plan."
    assert reviews[1].failed_checks == ["missing_route_reason", "missing_selection_scores"]
    assert reviews[2].status == "fail"


def test_project_audit_canary_errors_and_cleanup_failures(monkeypatch):
    runner = ProjectAuditRunner(NS(deal=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))))

    assert runner._text_canary("openai:x")["kind"] == "text_smoke"
    assert runner._structured_canary("openai:x")["kind"] == "structured"
    assert runner._tool_canary("openai:x")["kind"] == "tool"

    original_unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")))
    assert runner._text_file_canary("openai:x")["kind"] == "text_file"
    assert runner._image_canary("openai:x")["kind"] == "image"
    monkeypatch.setattr(Path, "unlink", original_unlink)


def test_project_audit_scan_skips_regex_source_and_duplicate_matches(tmp_path, monkeypatch):
    source = tmp_path / "app.py"
    source.write_text("re.compile('openai')\nopenai = OpenAI()\n", encoding="utf-8")
    pattern = audit_module.re.compile("OpenAI", audit_module.re.IGNORECASE)
    monkeypatch.setattr(
        audit_module,
        "COMMENT_PATTERNS",
        [(pattern, "same", "body", 1, "drop_in"), (pattern, "same", "body", 1, "drop_in")],
    )

    comments = scan_code_comments(tmp_path)

    assert [(comment.line, comment.title) for comment in comments] == [(2, "same")]


@pytest.mark.parametrize("target", ["autopatch", "unknown", "compat_client"])
def test_project_audit_patch_building_covers_non_default_paths(tmp_path, monkeypatch, target):
    monkeypatch.setattr(audit_module, "build_adoption_plan", lambda *_args, **_kwargs: _minimal_plan())
    if target == "compat_client":
        monkeypatch.setattr(audit_module, "_compat_client_patch_suggestions", lambda *_args, **_kwargs: [])

    report = build_adoption_patches(tmp_path, adoption_path=target)

    assert report.patches
    if target == "unknown":
        assert report.blockers == ["Unknown adoption path 'unknown'."]
    if target == "compat_client":
        assert report.patches[0].status == "manual"


def test_project_audit_feedback_signoff_and_doctor_edges(tmp_path):
    feedback = summarize_applied_human_feedback(
        NS(get=lambda _model: (_ for _ in ()).throw(KeyError("missing"))),
        {"groups": [{"model": "openai:x", "mode": "fast", "score_delta": 1}]},
    )
    assert feedback["pending"][0]["reason"] == "'missing'"

    with pytest.raises(CrupierError, match="Unknown adoption signoff verdict"):
        record_adoption_signoff(tmp_path, project="demo", verdict="later")

    signoffs = tmp_path / ".crupier" / "handoffs"
    signoffs.mkdir(parents=True)
    (signoffs / "signoffs.jsonl").write_text("\n" + json.dumps({"project": "demo"}) + "\n", encoding="utf-8")
    assert len(read_adoption_signoffs(tmp_path)) == 1

    assert _doctor_audit_gate(ProjectAuditReport("demo", "now", [], [])).status == "pass"
    real_report = ProjectAuditReport("demo", "now", [], [], real_canaries=[{"id": "ok", "ok": True}])
    assert _doctor_real_canary_gate(real_report, real=True, production=True).status == "pass"
    confident = NS(total_runs=1, last_run_at="now", model_scores=[NS(appearances=1, confidence="high")])
    assert _doctor_eval_history_gate(confident, production=True).status == "pass"


def test_project_audit_handoff_renders_overflow_and_feedback_commands(tmp_path):
    comments = [CodeComment("app.py", index, "Review", "body", priority=1) for index in range(21)]
    doctor = _minimal_doctor(comments=comments, gates=[DoctorGate("human_feedback", "warn", "review")])
    handoff = AdoptionHandoffReport("demo", "now", "needs-human-review", doctor, {}, [], [])

    assert "1 more comment" in format_adoption_handoff_markdown(handoff)

    actions, commands = _handoff_actions(
        doctor,
        {"feedback_review_reports": ["review.md"], "compare_reports": ["compare.json"]},
        paths=None,
    )
    assert actions and any("compare-report compare.json" in command for command in commands)

    actions, commands = _handoff_actions(doctor, {"compare_reports": ["compare.json"]}, paths=None)
    assert actions and any("compare-report compare.json" in command for command in commands)


def test_project_audit_file_error_and_filter_helpers(tmp_path, monkeypatch):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    assert _latest_production_decision_template([str(tmp_path / "missing"), str(invalid)]) is None
    assert _sarif_level(1) == "error"

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("fastapi", encoding="utf-8")
    original_read_text = Path.read_text

    def failing_read_text(path, *args, **kwargs):
        if path == pyproject:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read_text)
    assert _framework_hints(tmp_path)["frameworks"] == []

    monkeypatch.setattr(Path, "read_text", original_read_text)
    (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "plain.py").write_text("print('x')", encoding="utf-8")
    assert _compat_client_patch_suggestions(tmp_path, paths=None, max_files=10) == []


def test_project_audit_read_errors_invalid_review_lines_and_google_env(tmp_path, monkeypatch):
    source = tmp_path / "unreadable.py"
    source.write_text("OpenAI()", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_source(path, *args, **kwargs):
        if path == source:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_source)
    assert scan_code_comments(tmp_path, paths=[source]) == []
    monkeypatch.setattr(Path, "read_text", original_read_text)

    review_dir = tmp_path / ".crupier" / "code-comments"
    review_dir.mkdir(parents=True)
    (review_dir / "reviews.jsonl").write_text('\n{bad\n{"ok": true}\n', encoding="utf-8")
    assert _read_code_comment_review_records(tmp_path) == [{"ok": True}]

    monkeypatch.setattr(audit_module, "google_env_label", lambda _settings: "GOOGLE_KEYS")
    monkeypatch.setattr(audit_module, "google_env_present", lambda _settings: True)
    status = _provider_env_status(ProviderSettings(enabled=True), "google")
    assert status == {"key": "GOOGLE_KEYS", "required": True, "present": True, "host": None}


def test_project_audit_patch_filter_and_read_failure(tmp_path, monkeypatch):
    javascript = tmp_path / "app.js"
    python = tmp_path / "app.py"
    javascript.write_text("OpenAI", encoding="utf-8")
    python.write_text("from openai import OpenAI\n", encoding="utf-8")
    monkeypatch.setattr(audit_module, "_iter_source_files", lambda *_args, **_kwargs: iter([javascript, python]))
    original_read_text = Path.read_text

    def fail_python(path, *args, **kwargs):
        if path == python:
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_python)
    assert _compat_client_patch_suggestions(tmp_path, paths=None, max_files=10) == []


def test_project_audit_source_iteration_handles_external_and_stat_errors(tmp_path, monkeypatch):
    external = tmp_path.parent / "external-audit.py"
    external.write_text("print('x')", encoding="utf-8")
    try:
        assert list(_iter_source_files(tmp_path, paths=[external], max_files=2, max_file_size=100)) == [external]

        original_stat = Path.stat

        def failing_stat(path, *args, **kwargs):
            if path == external:
                raise OSError("gone")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", failing_stat)
        assert list(_iter_source_files(tmp_path, paths=[external], max_files=2, max_file_size=100)) == []
    finally:
        monkeypatch.undo()
        external.unlink(missing_ok=True)


def test_project_audit_source_iteration_skips_oversized_files(tmp_path):
    large = tmp_path / "large.py"
    large.write_text("x" * 20, encoding="utf-8")

    assert list(_iter_source_files(tmp_path, paths=[large], max_files=2, max_file_size=10)) == []

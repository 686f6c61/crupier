import json
from types import SimpleNamespace as NS

import pytest

from crupier.config import ProviderSettings
from crupier.errors import CrupierError
from crupier.project_audit import (
    AdoptionHandoffReport,
    AdoptionOption,
    AdoptionPatchReport,
    AdoptionPatchSuggestion,
    AuditCheck,
    CodeComment,
    CodeCommentReviewSummary,
    DoctorGate,
    ProjectAdoptionPlan,
    ProjectAuditReport,
    ProjectDoctorReport,
    RouteReview,
    _adoption_checklist,
    _adoption_warnings,
    _autopatch_patch_suggestions,
    _canary_error,
    _compat_client_python_rewrite,
    _confidence_from_score,
    _first_per_provider,
    _framework_hints,
    _handoff_actions,
    _iter_source_files,
    _ollama_cloud_host,
    _provider_env_status,
    _relative_path,
    build_adoption_review_contract,
    ensure_audit_ok,
    format_adoption_handoff_markdown,
    format_adoption_package_markdown,
    format_adoption_patch_markdown,
    format_adoption_plan_markdown,
    format_code_comments_markdown,
    format_code_review_comments_markdown,
    format_project_audit_markdown,
    format_project_doctor_markdown,
    import_code_comment_decisions,
    read_adoption_signoffs,
    write_project_audit_report,
)


def build_reports(tmp_path):
    comments = [
        CodeComment(
            file="app.py",
            line=4,
            title="OpenAI integration point",
            body="Route this call through Crupier.",
            priority=1,
            category="drop_in",
        )
    ]
    review_summary = CodeCommentReviewSummary(
        count=1,
        reviewed_count=0,
        pending_count=1,
        pending=comments,
    )
    option = AdoptionOption(
        path="compat_client",
        status="recommended",
        score=90,
        summary="Use compatibility client.",
        actions=["replace import"],
        risks=["review unsupported SDK methods"],
    )
    plan = ProjectAdoptionPlan(
        project="demo",
        generated_at="now",
        recommended_path="compat_client",
        confidence="high",
        options=[option],
        checklist=["run evals"],
        blockers=["human review pending"],
        warnings=["mixed providers"],
        code_comments=comments,
        code_comment_review=review_summary,
    )
    patch = AdoptionPatchSuggestion(
        adoption_path="compat_client",
        title="Replace OpenAI import",
        status="suggested",
        summary="Use Crupier's compatibility client.",
        diff="- from openai import OpenAI\n+ from crupier.compat.openai import OpenAI\n",
        commands=["pytest"],
        notes=["review async usage"],
        files=["app.py"],
    )
    patch_report = AdoptionPatchReport(
        project="demo",
        generated_at="now",
        adoption_path="compat_client",
        patches=[patch],
        blockers=["human review pending"],
        warnings=["manual patch"],
    )
    audit = ProjectAuditReport(
        project="demo",
        generated_at="now",
        checks=[
            AuditCheck("configuration", "pass", "valid"),
            AuditCheck("provider", "fail", "provider failed", severity="high", actions=["fix key"]),
        ],
        route_reviews=[
            RouteReview(
                id="route",
                task="test",
                status="warn",
                strategy="fusion",
                models=["openai:a", "anthropic:b"],
                reason="compare answers",
                human_questions=["Is the result useful?"],
            )
        ],
        real_canaries=[{"id": "text", "ok": True}, {"id": "tool", "ok": False}],
        code_comments=comments,
    )
    gates = [
        DoctorGate("configuration", "fail", "invalid", severity="high", actions=["fix"]),
        DoctorGate("human_feedback", "warn", "review", severity="high"),
        DoctorGate("adoption_signoff", "fail", "signoff", severity="high"),
        DoctorGate("programmer_code_comments", "warn", "comments"),
        DoctorGate("patch_suggestions", "warn", "patches"),
        DoctorGate("real_canaries", "fail", "canaries", severity="high"),
        DoctorGate("eval_history", "warn", "history"),
        DoctorGate("adoption_blockers", "fail", "blockers"),
    ]
    doctor = ProjectDoctorReport(
        project="demo",
        generated_at="now",
        readiness_mode="production",
        adoption_plan=plan,
        patch_report=patch_report,
        audit_report=audit,
        eval_history=NS(to_dict=lambda: {"runs": 1}),
        feedback_summary={"count": 0},
        gates=gates,
    )
    handoff = AdoptionHandoffReport(
        project="demo",
        generated_at="now",
        status="needs-human-review",
        doctor=doctor,
        artifacts={"reports": [], "audit": ["audit.json", "audit.md"]},
        required_human_actions=["review output"],
        suggested_commands=["crupier audit --real"],
    )
    return comments, plan, patch_report, audit, doctor, handoff


def test_markdown_formatters_render_complete_reports(tmp_path):
    comments, plan, patch_report, audit, doctor, handoff = build_reports(tmp_path)
    contract = doctor.review_contract
    package = {
        "project": "demo",
        "status": "needs-human-review",
        "doctor_status": "blocked",
        "readiness_mode": "production",
        "recommended_path": "compat_client",
        "review_contract": contract,
        "artifact_groups": {
            "adoption_handoff": ["handoff.json", "handoff.md"],
            "project_doctor": ["doctor.json", "doctor.md"],
            "code_review_comments": ["comments.json", "comments.md"],
            "code_sarif": ["comments.sarif"],
            "code_comment_decisions": ["decisions.json"],
        },
        "required_human_actions": ["review"],
        "suggested_commands": ["crupier adopt signoff"],
    }

    outputs = [
        format_project_audit_markdown(audit),
        format_project_doctor_markdown(doctor),
        format_adoption_handoff_markdown(handoff),
        format_adoption_package_markdown(package),
        format_adoption_plan_markdown(plan),
        format_adoption_patch_markdown(patch_report),
        format_code_comments_markdown(comments),
        format_code_review_comments_markdown(comments),
    ]

    combined = "\n".join(outputs)
    assert "Technical blockers: configuration" in combined
    assert "Human open gates" in combined
    assert "Is the result useful?" in combined
    assert "```diff" in combined
    assert "comments.sarif" in combined
    assert "Pending Programmer Code Comments" in combined


def test_empty_markdown_reports_have_explicit_no_action_copy(tmp_path):
    _, _, _, _, doctor, _ = build_reports(tmp_path)
    doctor.gates = []
    doctor.adoption_plan.blockers = []
    doctor.adoption_plan.code_comments = []
    doctor.adoption_plan.code_comment_review = None
    handoff = AdoptionHandoffReport(
        project="demo",
        generated_at="now",
        status="ready",
        doctor=doctor,
        artifacts={"empty": []},
        required_human_actions=[],
        suggested_commands=[],
    )

    handoff_md = format_adoption_handoff_markdown(handoff)
    package_md = format_adoption_package_markdown({"project": "demo", "artifact_groups": {}})

    assert "No required human actions remain" in handoff_md
    assert "none found" in handoff_md
    assert "No suggested commands" in package_md
    assert "No AI integration hotspots" in format_code_comments_markdown([])
    assert "No AI integration hotspots" in format_code_review_comments_markdown([])


def test_audit_report_writer_creates_json_and_markdown(tmp_path):
    _, _, _, audit, _, _ = build_reports(tmp_path)

    paths = write_project_audit_report(tmp_path, audit)

    assert {path.suffix for path in paths} == {".json", ".md"}
    assert json.loads(next(path for path in paths if path.suffix == ".json").read_text())["project"] == "demo"
    assert "# Crupier Project Audit" in next(path for path in paths if path.suffix == ".md").read_text()


@pytest.mark.parametrize(
    ("gates", "overall", "summary_fragment"),
    [
        ([DoctorGate("technical", "fail", "x")], "blocked", "Technical gates are failing"),
        ([DoctorGate("human_feedback", "warn", "x")], "needs-human-review", "human review"),
        ([DoctorGate("technical", "warn", "x")], "ready_with_warnings", "technical warnings"),
        ([DoctorGate("technical", "pass", "x")], "ready", "closed"),
    ],
)
def test_review_contract_distinguishes_machine_and_human_gates(gates, overall, summary_fragment):
    contract = build_adoption_review_contract(gates)

    assert contract["overall_status"] == overall
    assert summary_fragment in contract["summary"]


def test_handoff_actions_cover_all_open_gate_remediations(tmp_path):
    _, _, _, _, doctor, _ = build_reports(tmp_path)
    decision = tmp_path / "decisions.json"
    decision.write_text('{"source_dry_run": false}', encoding="utf-8")
    artifacts = {
        "feedback_decision_templates": [str(decision)],
        "code_comment_decision_templates": ["code-decisions.json"],
        "compare_reports": ["compare.json"],
    }

    actions, commands = _handoff_actions(doctor, artifacts, paths=["app.py"])

    assert any("human decision template" in action for action in actions)
    assert any("adoption signoff" in action for action in actions)
    assert any("programmer code comments" in action for action in actions)
    assert any("real provider canaries" in action for action in actions)
    assert any("--import-decisions code-decisions.json" in command for command in commands)
    assert len(commands) == len(set(commands))


def test_handoff_actions_handle_dry_run_review_and_missing_compare_paths(tmp_path):
    _, _, _, _, doctor, _ = build_reports(tmp_path)
    doctor.gates = [DoctorGate("human_feedback", "warn", "review")]
    dry = tmp_path / "dry.json"
    dry.write_text('{"source_dry_run": true}', encoding="utf-8")

    actions, commands = _handoff_actions(
        doctor,
        {"feedback_decision_templates": [str(dry)]},
        paths=None,
    )
    assert "dry-run" in actions[0]
    assert "compare-dataset" in commands[0]

    actions, commands = _handoff_actions(doctor, {"feedback_review_reports": ["review.md"]}, paths=None)
    assert "review packet" in actions[0]
    assert commands == ["crupier feedback apply"]

    actions, commands = _handoff_actions(doctor, {}, paths=None)
    assert "Generate a feedback review packet" in actions[0]
    assert any("compare-dataset" in command for command in commands)


def test_framework_hints_detect_mixed_python_and_node_stacks(tmp_path):
    (tmp_path / "pyproject.toml").write_text("FastAPI Flask Django", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text("# flask", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        json.dumps({"dependencies": {"next": "1", "express": "1"}, "devDependencies": {"next.js": "1"}}),
        encoding="utf-8",
    )

    hints = _framework_hints(tmp_path)

    assert hints["python"] is True
    assert hints["node"] is True
    assert hints["frameworks"] == ["django", "express", "fastapi", "flask", "nextjs"]

    (tmp_path / "package.json").write_text("{bad", encoding="utf-8")
    assert _framework_hints(tmp_path)["node"] is True


def test_import_code_comment_decisions_validates_files_and_tracks_unknowns(tmp_path):
    comment = CodeComment("app.py", 1, "Review", "Inspect", priority=1)
    missing = tmp_path / "missing.json"
    with pytest.raises(CrupierError, match="Could not read"):
        import_code_comment_decisions(tmp_path, [comment], missing)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    with pytest.raises(CrupierError, match="Invalid code comment"):
        import_code_comment_decisions(tmp_path, [comment], invalid)

    no_list = tmp_path / "no-list.json"
    no_list.write_text("{}", encoding="utf-8")
    with pytest.raises(CrupierError, match="comments list"):
        import_code_comment_decisions(tmp_path, [comment], no_list)

    bad_verdict = tmp_path / "bad-verdict.json"
    bad_verdict.write_text('{"comments":[{"fingerprint":"x","verdict":"maybe"}]}', encoding="utf-8")
    with pytest.raises(CrupierError, match="Unknown code comment verdict"):
        import_code_comment_decisions(tmp_path, [comment], bad_verdict)

    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "comments": [
                    "ignored",
                    {"fingerprint": "unknown", "verdict": "accepted"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = import_code_comment_decisions(tmp_path, [comment], decisions, note="Bearer abcdefghijklmnop")
    assert result["unknown_count"] == 1
    assert result["missing_current_count"] == 1
    assert "[redacted]" in result["note"]


def test_signoff_reader_skips_invalid_records_and_filters_project(tmp_path):
    signoffs = tmp_path / ".crupier" / "handoffs"
    signoffs.mkdir(parents=True)
    (signoffs / "signoffs.jsonl").write_text(
        "{bad\n"
        + json.dumps(["not-object"])
        + "\n"
        + json.dumps({"project": "other", "verdict": "approve"})
        + "\n"
        + json.dumps({"project": "demo", "verdict": "reject"})
        + "\n",
        encoding="utf-8",
    )

    assert read_adoption_signoffs(tmp_path, project="demo") == [{"project": "demo", "verdict": "reject"}]


def test_source_iteration_and_relative_paths_respect_limits_and_boundaries(tmp_path):
    (tmp_path / "app.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / "large.py").write_text("x" * 100, encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")

    paths = list(_iter_source_files(tmp_path, paths=["missing", "app.py", "large.py"], max_files=1, max_file_size=20))
    assert paths == [tmp_path / "app.py"]

    external = tmp_path.parent / "external.py"
    external.write_text("x", encoding="utf-8")
    try:
        assert _relative_path(tmp_path, external) == str(external.resolve())
    finally:
        external.unlink()


def test_adoption_helpers_cover_all_paths_and_warnings():
    for path, expected in (
        ("proxy", "crupier serve"),
        ("compat_client", "crupier.compat.openai.OpenAI"),
        ("autopatch", "crupier.install"),
        ("native_sdk", "Crupier.from_project"),
    ):
        checklist = _adoption_checklist(path, {"hardcoded_models": 1}, blocked=True)
        assert any(expected in item for item in checklist)
        assert any("hard-coded" in item for item in checklist)

    warnings = _adoption_warnings(
        {"provider_count": 2, "drop_in": 0, "credential_fixtures": 1},
        {"python": True, "node": True},
    )
    assert len(warnings) == 4
    assert _confidence_from_score(80) == "high"
    assert _confidence_from_score(60) == "medium"
    assert _confidence_from_score(10) == "low"
    assert _first_per_provider(["openai:a", "openai:b", "anthropic:c"]) == ["openai:a", "anthropic:c"]


def test_compat_rewrite_autopatch_provider_and_canary_helpers(monkeypatch):
    rewritten = _compat_client_python_rewrite(
        "from openai import OpenAI\nfrom openai import OpenAI, AsyncOpenAI\nimport os\n"
    )
    assert "from crupier.compat.openai import OpenAI" in rewritten
    assert "TODO: review AsyncOpenAI" in rewritten
    assert _autopatch_patch_suggestions()[0].files == ["crupier_bootstrap.py"]

    monkeypatch.setenv("OLLAMA_API_KEY", "key")
    status = _provider_env_status(
        ProviderSettings(enabled=True, env_key="OLLAMA_API_KEY", host="https://ollama.com/api"),
        "ollama",
    )
    assert status["required"] is True
    assert status["present"] is True
    assert _ollama_cloud_host(None) is False
    assert _ollama_cloud_host("http://127.0.0.1:11434") is False

    error = _canary_error("text", "chat", "openai:a", RuntimeError("sk-abcdefghijklmno"))
    assert error["error"] == "[redacted]"


def test_ensure_audit_ok_rejects_failed_report(tmp_path):
    _, _, _, audit, _, _ = build_reports(tmp_path)

    with pytest.raises(CrupierError, match="not ready"):
        ensure_audit_ok(audit)

import json
import stat
from pathlib import Path

import pytest

from crupier import Crupier
from crupier.cli import main
from crupier.config import CrupierConfig, write_default_project
from crupier.errors import CrupierError
from crupier.feedback import (
    HumanReviewItem,
    HumanReviewPacket,
    _resolve_report_path,
    _review_items_from_comparison,
    _review_items_from_report,
    _score_status,
    _validate_rating,
    build_human_decision_template,
    build_human_review_packet,
    format_human_review_markdown,
)
from crupier.models import CapabilityCard, ModelRef


def make_feedback_client(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "feedback-test", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:bad-model", "openai:good-model"]},
            "routing": {"default_strategy": "single"},
            "profiles": {"agentic": {"prefer": [], "strategy": "single"}},
        }
    )
    config.root = tmp_path
    client = Crupier(config, adapters={})
    for model in config.models.allow:
        client.registry.save_card(
            CapabilityCard(
                model_ref=ModelRef.parse(model),
                last_updated="test",
                quality_tier="strong",
                cost_tier="low",
                latency_tier="fast",
            )
        )
    return client


def test_human_feedback_apply_changes_selector_decision(tmp_path):
    client = make_feedback_client(tmp_path)

    baseline = client.deal("Plan a small agent task", mode="agentic", strategy="single", dry_run=True, trace=True)
    assert baseline.route is not None
    first_choice = baseline.route.models[0]
    second_choice = next(model for model in client.config.models.allow if model != first_choice)

    record = client.feedback.record(
        project=client.config.project.name,
        models=[first_choice],
        mode="agentic",
        strategy="single",
        rating=1,
        verdict="reject",
        tags=["wrong_route"],
        note="Technically passed, but the answer was not useful.",
    )
    assert record.feedback_id.startswith("hfb_")

    summary = client.feedback.summary()
    assert summary["groups"][0]["score_delta"] < 0

    report = client.feedback.apply_to_registry(client.registry)
    assert report["updated"][0]["score_key"] == "human:agentic"

    updated = Crupier(client.config, adapters={})
    routed = updated.deal("Plan a small agent task", mode="agentic", strategy="single", dry_run=True, trace=True)
    assert routed.route is not None
    assert routed.route.models[0] == second_choice
    rejected_score = next(item for item in routed.route.selection_scores if item["model"] == first_choice)
    assert any(term["name"] == "human_feedback" and term["value"] < 0 for term in rejected_score["terms"])


def test_human_feedback_can_derive_route_from_stored_trace(tmp_path):
    client = make_feedback_client(tmp_path)
    result = client.deal(
        "Trace a route for review",
        mode="agentic",
        strategy="single",
        constraints={"store_trace": True},
        dry_run=True,
        trace="summary",
    )
    assert result.trace is not None
    assert result.route is not None

    record = client.feedback.record(
        project=client.config.project.name,
        trace_id=result.trace.trace_id,
        rating=5,
        verdict="accept",
        trace_store=client.traces,
    )

    assert record.models == [result.route.models[0]]
    assert record.mode == "agentic"
    assert record.strategy == "single"


def test_sensitive_artifact_writer_rejects_symlink_targets(tmp_path):
    client = make_feedback_client(tmp_path)
    client.config.feedback_dir.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    client.feedback.path.symlink_to(victim)

    with pytest.raises(CrupierError, match="symbolic link"):
        client.feedback.record(
            project=client.config.project.name,
            models=["openai:good-model"],
            rating=5,
        )

    assert victim.read_text(encoding="utf-8") == "keep me"


def test_sensitive_append_is_atomic_and_preserves_private_mode(tmp_path):
    client = make_feedback_client(tmp_path)
    for rating in (1, 3, 5):
        client.feedback.record(
            project=client.config.project.name,
            models=["openai:good-model"],
            rating=rating,
        )

    lines = client.feedback.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["rating"] for line in lines] == [1, 3, 5]
    assert stat.S_IMODE(client.feedback.path.stat().st_mode) == 0o600


def test_feedback_summary_refuses_to_apply_when_jsonl_is_corrupt(tmp_path):
    client = make_feedback_client(tmp_path)
    client.feedback.record(
        project=client.config.project.name,
        models=["openai:good-model"],
        rating=5,
        verdict="accept",
    )
    with client.feedback.path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated\n")
    original = client.registry.get("openai:good-model").to_dict()

    summary = client.feedback.summary()
    with pytest.raises(CrupierError, match="corrupt"):
        client.feedback.apply_to_registry(client.registry)

    assert summary["count"] == 1
    assert summary["complete"] is False
    assert client.registry.get("openai:good-model").to_dict() == original


def test_feedback_diagnostic_includes_line_without_echoing_content(tmp_path):
    client = make_feedback_client(tmp_path)
    secret = "private-feedback-content-must-not-leak"
    client.feedback.record(
        project=client.config.project.name,
        models=["openai:good-model"],
        rating=4,
    )
    with client.feedback.path.open("a", encoding="utf-8") as handle:
        handle.write(secret + "\n")

    summary = client.feedback.summary()
    serialized = json.dumps(summary["diagnostics"])

    assert summary["diagnostics"][0]["line"] == 2
    assert summary["diagnostics"][0]["error_type"] == "invalid_json"
    assert secret not in serialized


def test_cli_feedback_record_summary_and_apply(tmp_path, capsys):
    write_default_project(tmp_path)

    record_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "record",
            "--model",
            "openai:gpt-5.4-mini",
            "--mode",
            "fast",
            "--rating",
            "2",
            "--verdict",
            "needs_work",
            "--tag",
            "too_slow",
            "--json",
        ]
    )
    record_payload = json.loads(capsys.readouterr().out)

    summary_status = main(["--project", str(tmp_path), "feedback", "summary", "--json"])
    summary_payload = json.loads(capsys.readouterr().out)

    apply_status = main(["--project", str(tmp_path), "feedback", "apply", "--json"])
    apply_payload = json.loads(capsys.readouterr().out)

    assert record_status == 0
    assert record_payload["models"] == ["openai:gpt-5.4-mini"]
    assert summary_status == 0
    assert summary_payload["count"] == 1
    assert apply_status == 0
    assert apply_payload["updated"][0]["score_key"] == "human:fast"


def test_cli_feedback_record_can_derive_from_compare_report(tmp_path, capsys):
    write_default_project(tmp_path)

    compare_status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare",
            "Answer briefly",
            "--mode",
            "fast",
            "--model",
            "openai:gpt-5.5",
            "--model",
            "openai:gpt-5.4-mini",
            "--write-report",
            "--json",
        ]
    )
    compare_payload = json.loads(capsys.readouterr().out)

    record_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "record",
            "--compare-report",
            compare_payload["written_path"],
            "--allow-dry-run-source",
            "--rating",
            "5",
            "--verdict",
            "accept",
            "--json",
        ]
    )
    record_payload = json.loads(capsys.readouterr().out)

    assert compare_status == 0
    assert record_status == 0
    assert record_payload["models"] == ["openai:gpt-5.4-mini"]
    assert record_payload["mode"] == "fast"
    assert "compare_report" in record_payload["tags"]
    assert "dry_run_source" in record_payload["tags"]
    assert "output_preview" not in json.dumps(record_payload)


def test_compare_report_write_warns_path_and_sensitivity(tmp_path, capsys):
    write_default_project(tmp_path)

    status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare",
            "Answer briefly",
            "--mode",
            "fast",
            "--write-report",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert status == 0
    assert payload["written_path"] in captured.err
    assert "sensitive" in captured.err.lower()


def test_cli_feedback_review_creates_actionable_packet(tmp_path, capsys):
    write_default_project(tmp_path)

    compare_status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare",
            "Answer briefly",
            "--mode",
            "fast",
            "--model",
            "openai:gpt-5.5",
            "--model",
            "openai:gpt-5.4-mini",
            "--write-report",
            "--json",
        ]
    )
    compare_payload = json.loads(capsys.readouterr().out)

    review_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "review",
            "--compare-report",
            compare_payload["written_path"],
            "--no-preview",
            "--write-report",
            "--json",
        ]
    )
    review_payload = json.loads(capsys.readouterr().out)

    assert compare_status == 0
    assert review_status == 0
    assert review_payload["ok"] is True
    assert review_payload["source_type"] == "compare"
    assert review_payload["total_items"] == 2
    assert review_payload["recommended_items"] == 1
    assert len(review_payload["written_files"]) == 2
    assert all("output_preview" not in item for item in review_payload["items"])
    recommended = next(item for item in review_payload["items"] if item["recommended"])
    assert recommended["variant"] == "openai:gpt-5.4-mini"
    assert "--verdict accept" in recommended["feedback_commands"]["accept"]
    assert "--allow-dry-run-source" in recommended["feedback_commands"]["accept"]
    assert "--tag human_review" in recommended["feedback_commands"]["reject"]
    assert "recommended_variant" in recommended["feedback_commands"]["accept"]
    for path in review_payload["written_files"]:
        assert (Path(path) if path.startswith("/") else tmp_path / path).exists()


def test_cli_feedback_review_decision_template_imports_and_applies(tmp_path, capsys):
    write_default_project(tmp_path)

    compare_status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare",
            "Answer briefly",
            "--mode",
            "fast",
            "--model",
            "openai:gpt-5.5",
            "--model",
            "openai:gpt-5.4-mini",
            "--write-report",
            "--json",
        ]
    )
    compare_payload = json.loads(capsys.readouterr().out)

    review_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "review",
            "--compare-report",
            compare_payload["written_path"],
            "--no-preview",
            "--write-decisions-template",
            "--json",
        ]
    )
    review_payload = json.loads(capsys.readouterr().out)
    decision_path = next(Path(path) for path in review_payload["written_files"] if "human_decisions_" in path)
    template = json.loads(decision_path.read_text(encoding="utf-8"))

    first_decision = template["decisions"][0]
    first_decision["record"] = True
    first_decision["rating"] = 5
    first_decision["verdict"] = "accept"
    first_decision["note"] = "Human accepted this route; redact " + "s" + "k-testsecret0000000000."
    decision_path.write_text(json.dumps(template), encoding="utf-8")

    blocked_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "import-decisions",
            "--decisions",
            str(decision_path),
            "--json",
        ]
    )
    blocked_stderr = capsys.readouterr().err

    import_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "import-decisions",
            "--decisions",
            str(decision_path),
            "--allow-dry-run-source",
            "--apply-to-registry",
            "--json",
        ]
    )
    import_payload = json.loads(capsys.readouterr().out)
    summary_status = main(["--project", str(tmp_path), "feedback", "summary", "--json"])
    summary_payload = json.loads(capsys.readouterr().out)

    assert compare_status == 0
    assert review_status == 0
    assert "output_preview" not in json.dumps(template)
    assert "feedback_commands" not in json.dumps(template)
    assert blocked_status == 1
    assert "dry-run compare report" in blocked_stderr
    assert import_status == 0
    assert import_payload["imported"] == 1
    assert import_payload["records"][0]["note"] == "Human accepted this route; redact [redacted]."
    assert "dry_run_source" in import_payload["records"][0]["tags"]
    assert import_payload["apply_report"]["updated"][0]["score_key"] == "human:fast"
    assert summary_status == 0
    assert summary_payload["count"] == 1
    assert summary_payload["dry_run_source_count"] == 1
    assert summary_payload["production_feedback_count"] == 0


def test_cli_feedback_record_can_derive_from_compare_dataset_case(tmp_path, capsys):
    write_default_project(tmp_path)
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "review-dataset",
                "cases": [
                    {"id": "fast", "task": "Answer briefly.", "mode": "fast"},
                    {"id": "structured", "task": "Extract JSON.", "mode": "structured"},
                ],
            }
        ),
        encoding="utf-8",
    )

    compare_status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare-dataset",
            "--dataset",
            str(dataset),
            "--model",
            "openai:gpt-5.5",
            "--model",
            "openai:gpt-5.4-mini",
            "--write-report",
            "--json",
        ]
    )
    compare_payload = json.loads(capsys.readouterr().out)

    record_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "record",
            "--compare-report",
            compare_payload["written_path"],
            "--allow-dry-run-source",
            "--case-id",
            "structured",
            "--variant",
            "openai:gpt-5.4-mini",
            "--rating",
            "2",
            "--verdict",
            "needs_work",
            "--json",
        ]
    )
    record_payload = json.loads(capsys.readouterr().out)

    assert compare_status == 0
    assert record_status == 0
    assert record_payload["models"] == ["openai:gpt-5.4-mini"]
    assert record_payload["mode"] == "structured"
    assert "compare_case:structured" in record_payload["tags"]


def test_cli_feedback_review_filters_compare_dataset_case_and_variant(tmp_path, capsys):
    write_default_project(tmp_path)
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "review-dataset",
                "cases": [
                    {"id": "fast", "task": "Answer briefly.", "mode": "fast"},
                    {"id": "structured", "task": "Extract JSON.", "mode": "structured"},
                ],
            }
        ),
        encoding="utf-8",
    )

    compare_status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare-dataset",
            "--dataset",
            str(dataset),
            "--model",
            "openai:gpt-5.5",
            "--model",
            "openai:gpt-5.4-mini",
            "--write-report",
            "--json",
        ]
    )
    compare_payload = json.loads(capsys.readouterr().out)

    review_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "review",
            "--compare-report",
            compare_payload["written_path"],
            "--case-id",
            "structured",
            "--variant",
            "openai:gpt-5.4-mini",
            "--json",
        ]
    )
    review_payload = json.loads(capsys.readouterr().out)

    assert compare_status == 0
    assert review_status == 0
    assert review_payload["source_type"] == "compare_dataset"
    assert review_payload["total_items"] == 1
    item = review_payload["items"][0]
    assert item["case_id"] == "structured"
    assert item["variant"] == "openai:gpt-5.4-mini"
    assert "--case-id structured" in item["feedback_commands"]["needs_work"]


def test_feedback_review_reports_empty_selection_and_case_tag(tmp_path):
    report = tmp_path / "compare.json"
    report.write_text(json.dumps({"dry_run": False, "variants": []}), encoding="utf-8")
    packet = build_human_review_packet(tmp_path, report_path=str(report))
    assert packet.ok is False
    assert packet.warnings == ["No review items matched the selected case/variant."]

    item = HumanReviewItem(
        id="case-1:variant-a",
        case_id="case-1",
        variant="variant-a",
        models=["openai:gpt-5.5"],
        recommended=True,
    )
    template = build_human_decision_template(
        HumanReviewPacket("report.json", "compare_dataset", False, 1, 1, [item])
    )
    assert "compare_case:case-1" in template["decisions"][0]["tags"]


def test_feedback_markdown_renders_optional_review_context() -> None:
    item = HumanReviewItem(
        id="case-1:variant-a",
        case_id="case-1",
        variant="variant-a",
        models=["openai:gpt-5.5"],
        task="Review this result",
        failed_checks=["latency too high"],
        output_preview="preview text",
    )
    markdown = format_human_review_markdown(
        HumanReviewPacket("report.json", "compare_dataset", False, 1, 0, [item])
    )
    assert "- case: `case-1`" in markdown
    assert "- task: Review this result" in markdown
    assert "- failed_check: latency too high" in markdown
    assert "preview text" in markdown


def test_feedback_validation_and_report_paths_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(CrupierError, match="integer from 1 to 5"):
        _validate_rating("invalid")
    with pytest.raises(CrupierError, match="integer from 1 to 5"):
        _validate_rating(6)

    nested = tmp_path / "reports" / "report.json"
    nested.parent.mkdir()
    nested.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    assert _resolve_report_path(tmp_path, "reports/report.json") == nested.resolve()
    with pytest.raises(CrupierError, match="not found"):
        _resolve_report_path(tmp_path, "reports/missing.json")


def test_feedback_dataset_skips_invalid_cases_and_variants() -> None:
    data = {
        "dry_run": False,
        "cases": [
            "invalid",
            {"id": "missing-comparison"},
            {
                "id": "valid",
                "comparison": {"variants": ["invalid", {"name": "kept", "models": []}]},
            },
        ],
    }
    items, dry_run = _review_items_from_report(
        data,
        report_path="report.json",
        case_id=None,
        variant=None,
        include_output_preview=False,
    )
    assert dry_run is False
    assert [item.id for item in items] == ["valid:kept"]

    class InconsistentCases(dict):
        reads = 0

        def get(self, key, default=None):
            if key == "cases":
                self.reads += 1
                return [] if self.reads == 1 else "invalid"
            return super().get(key, default)

    with pytest.raises(CrupierError, match="invalid cases"):
        _review_items_from_report(
            InconsistentCases(),
            report_path="report.json",
            case_id=None,
            variant=None,
            include_output_preview=False,
        )

    with pytest.raises(CrupierError, match="invalid variants"):
        _review_items_from_comparison(
            {"variants": "invalid"},
            report_path="report.json",
            case_id=None,
            task="",
            winner=None,
            dry_run=False,
            variant=None,
            include_output_preview=False,
        )


def test_feedback_score_status_neutral_boundary() -> None:
    assert _score_status(0.5) == "neutral"

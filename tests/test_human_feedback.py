import json
from pathlib import Path

from crupier import Crupier
from crupier.cli import main
from crupier.config import CrupierConfig, write_default_project
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

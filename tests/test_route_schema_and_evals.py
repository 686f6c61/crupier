import hashlib
import json
from pathlib import Path

from crupier import Crupier
from crupier import evals as evals_module
from crupier.cli import main
from crupier.config import CrupierConfig
from crupier.errors import CrupierRouteValidationError
from crupier.evals import (
    CompareDatasetModelScore,
    CompareVariant,
    RoutingEvalRunner,
    apply_compare_scores_to_registry,
    evaluate_expectations,
)
from crupier.models import CostEstimate, CrupierResult, RoutePlan, RouteStep
from crupier.route_schema import validate_route_plan_shape


def make_config(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "eval-test", "default_profile": "agentic"},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "anthropic": {"enabled": True, "env_key": "ANTHROPIC_API_KEY"},
                "ollama": {"enabled": True, "host": "http://localhost:11434"},
            },
            "models": {
                "allow": [
                    "openai:gpt-5.5",
                    "openai:gpt-5.4-mini",
                    "anthropic:claude-opus-4-8",
                    "ollama:qwen3.5:122b",
                ]
            },
            "routing": {"default_strategy": "orchestrated", "allow_fusion": True, "max_calls": 8},
            "profiles": {
                "agentic": {"prefer": ["tool_use", "coding"], "strategy": "orchestrated"},
                "cheap": {"prefer": ["low_cost"], "strategy": "orchestrated"},
                "fast": {"prefer": ["low_latency"], "strategy": "single"},
                "private": {"prefer": ["local"], "strategy": "local_first"},
                "research": {"prefer": ["consensus"], "strategy": "fusion"},
                "structured": {"prefer": ["structured_output"], "strategy": "cascade"},
            },
            "orchestrator": {"model": "openai:gpt-5.4-mini"},
        }
    )
    config.root = tmp_path
    return config


def test_route_plan_shape_rejects_invalid_role_for_strategy():
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="panel", models=["openai:gpt-5.5"])],
        estimated_cost=CostEstimate(0.0),
    )

    try:
        validate_route_plan_shape(plan)
    except CrupierRouteValidationError as exc:
        assert "not valid for strategy" in str(exc)
    else:
        raise AssertionError("invalid role should be rejected")


def test_route_plan_shape_rejects_negative_cost():
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:gpt-5.5")],
        estimated_cost=CostEstimate(-1.0),
    )

    try:
        validate_route_plan_shape(plan)
    except CrupierRouteValidationError as exc:
        assert "cost cannot be negative" in str(exc)
    else:
        raise AssertionError("negative cost should be rejected")


def test_eval_expectations_report_human_relevant_failures():
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:gpt-5.5")],
    )

    failures = evaluate_expectations(
        plan,
        {"strategy": "fusion", "providers_exclude": ["openai"], "min_models": 2},
    )

    assert "strategy expected 'fusion', got 'single'" in failures
    assert "unexpected provider 'openai'" in failures
    assert "expected at least 2 models, got 1" in failures


def test_eval_expectations_fail_on_unknown_keys():
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:gpt-5.5")],
    )

    failures = evaluate_expectations(plan, {"strategy": "single", "provider_exclude": ["openai"]})

    assert failures == ["unknown expectation key 'provider_exclude'"]


def test_eval_case_loads_tools_and_rejects_a_non_list():
    case = evals_module.EvalCase.from_dict(
        {
            "id": "with-tools",
            "task": "Review",
            "tools": [{"name": "read_changed_file"}],
        }
    )

    assert case.tools == [{"name": "read_changed_file"}]
    try:
        evals_module.EvalCase.from_dict({"id": "bad", "task": "x", "tools": {"name": "x"}})
    except TypeError as exc:
        assert "tools must be a list" in str(exc)
    else:
        raise AssertionError("non-list tools should be rejected")


def test_route_plan_shape_accepts_delegate_strategy():
    plan = RoutePlan(
        strategy="delegate",
        steps=[
            RouteStep(
                role="delegate",
                model="openai:gpt-5.4-mini",
                params={"task": "Research then write", "strategy": "orchestrated"},
            )
        ],
    )

    validate_route_plan_shape(plan)


def test_builtin_routing_evals_pass_with_seed_config(tmp_path):
    report = RoutingEvalRunner(Crupier(make_config(tmp_path))).run()

    assert report.ok
    assert report.total == 5
    assert report.passed == 5


def test_public_routing_eval_dataset_passes_offline(tmp_path):
    dataset = Path(__file__).resolve().parents[1] / "examples" / "routing-eval.json"
    report = RoutingEvalRunner(Crupier(make_config(tmp_path))).run(dataset=dataset)

    assert report.ok, [(item.id, item.failed_checks, item.error) for item in report.results if not item.ok]
    assert report.total == 7
    assert report.passed == 7


def test_eval_runner_loads_dataset_and_writes_report(tmp_path):
    dataset = tmp_path / "routing.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "tiny",
                "cases": [
                    {
                        "id": "fast",
                        "task": "Short answer",
                        "mode": "fast",
                        "expect": {"strategy": "single", "max_models": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = Crupier(make_config(tmp_path))

    report = client.evals.run(dataset=dataset, write_report=True)

    assert report.ok
    assert report.written_path is not None
    written = json.loads((tmp_path / report.written_path).read_text(encoding="utf-8"))
    assert written["name"] == "tiny"


def test_eval_compare_recommends_lower_cost_passing_variant(tmp_path):
    client = Crupier(make_config(tmp_path))

    report = client.evals.compare(
        task="Answer a short project question.",
        mode="fast",
        variants=[
            CompareVariant(name="frontier", constraints={"force_model": "openai:gpt-5.5"}),
            CompareVariant(name="mini", constraints={"force_model": "openai:gpt-5.4-mini"}),
        ],
        dry_run=True,
    )

    assert report.ok
    assert report.winner == "mini"
    assert report.variants[0].human_questions
    assert all(item.estimated_cost_usd is not None for item in report.variants)


def test_compare_report_default_omits_task_and_output_preview_content(tmp_path):
    client = Crupier(make_config(tmp_path))
    task = "Summarize the account for alice@example.test"
    input_value = {"customer": "Alice Example", "notes": "private renewal"}

    report = client.evals.compare(
        task=task,
        input=input_value,
        mode="fast",
        write_report=True,
    )

    assert report.written_path is not None
    payload = json.loads((tmp_path / report.written_path).read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert "alice@example.test" not in serialized
    assert "Alice Example" not in serialized
    assert "private renewal" not in serialized
    assert "output_preview" not in payload["variants"][0]
    assert payload["task_hash"] == hashlib.sha256(task.encode()).hexdigest()
    assert payload["task_length"] == len(task)
    assert payload["input_hash"] == hashlib.sha256(
        json.dumps(input_value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    assert payload["input_length"] > 0
    assert payload["variants"][0]["output_preview_hash"]
    assert payload["variants"][0]["output_preview_length"] > 0


def test_compare_report_store_content_is_explicit_and_redacted(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)
    secret = "s" + "k-testsecret0000000000"
    task = f"Summarize the harmless project brief; API key: {secret}"

    status = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare",
            task,
            "--input",
            json.dumps({"context": f"public product description; token={secret}"}),
            "--mode",
            "fast",
            "--write-report",
            "--store-content",
            "--json",
        ]
    )
    command_payload = json.loads(capsys.readouterr().out)

    assert status == 0
    payload = json.loads((tmp_path / command_payload["written_path"]).read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert "harmless project brief" in payload["task"]
    assert "public product description" in payload["input"]["context"]
    assert "output_preview" in payload["variants"][0]
    assert secret not in serialized
    assert "[redacted]" in serialized


def test_eval_compare_dataset_can_apply_scores_to_registry(tmp_path):
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "compare-smoke",
                "cases": [
                    {"id": "short_1", "task": "Answer briefly.", "mode": "fast"},
                    {"id": "short_2", "task": "Summarize briefly.", "mode": "fast"},
                ],
            }
        ),
        encoding="utf-8",
    )
    client = Crupier(make_config(tmp_path))

    report = client.evals.compare_dataset(
        dataset=dataset,
        variants=[
            CompareVariant(name="frontier", constraints={"force_model": "openai:gpt-5.5"}),
            CompareVariant(name="mini", constraints={"force_model": "openai:gpt-5.4-mini"}),
        ],
        apply=True,
        min_count=1,
        min_confidence="low",
        dry_run=True,
    )

    assert report.ok
    assert report.passed_cases == 2
    mini_score = next(item for item in report.model_scores if item.model == "openai:gpt-5.4-mini")
    assert mini_score.score_key == "eval:fast"
    assert mini_score.wins == 2
    assert mini_score.score_delta > 0
    card = client.registry.get("openai:gpt-5.4-mini")
    assert card.local_eval_scores["eval:fast"] == mini_score.score_delta


def test_eval_compare_history_summarizes_and_applies_confident_scores(tmp_path):
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "history-smoke",
                "cases": [
                    {"id": "short_1", "task": "Answer briefly.", "mode": "fast"},
                    {"id": "short_2", "task": "Summarize briefly.", "mode": "fast"},
                ],
            }
        ),
        encoding="utf-8",
    )
    client = Crupier(make_config(tmp_path))
    variants = [
        CompareVariant(name="frontier", constraints={"force_model": "openai:gpt-5.5"}),
        CompareVariant(name="mini", constraints={"force_model": "openai:gpt-5.4-mini"}),
    ]

    first = client.evals.compare_dataset(dataset=dataset, variants=variants, record_history=True)
    second = client.evals.compare_dataset(dataset=dataset, variants=variants, record_history=True)
    history = client.evals.history(apply=True, min_count=3, min_confidence="medium", dry_run=False)

    assert first.history_path is not None
    assert second.history_path is not None
    assert history.total_runs == 2
    mini = next(item for item in history.model_scores if item.model == "openai:gpt-5.4-mini")
    assert mini.appearances == 4
    assert mini.confidence == "medium"
    assert mini.trend == "stable"
    assert any(item["score_key"] == "eval:fast" for item in history.apply_report["updated"])
    assert client.registry.get("openai:gpt-5.4-mini").local_eval_scores["eval:fast"] == mini.score_delta


def test_eval_apply_skips_low_signal_and_missing_models(tmp_path):
    client = Crupier(make_config(tmp_path))
    low_count_card = client.registry.get("openai:gpt-5.5")
    low_confidence_card = client.registry.get("openai:gpt-5.4-mini")
    scores = [
        CompareDatasetModelScore(
            model=low_count_card.model_ref.key,
            mode="fast",
            appearances=1,
            passed=1,
            wins=1,
            score_delta=4.0,
            score_key="eval:fast",
            confidence="low",
        ),
        CompareDatasetModelScore(
            model=low_confidence_card.model_ref.key,
            mode="fast",
            appearances=3,
            passed=3,
            wins=2,
            score_delta=3.0,
            score_key="eval:fast",
            confidence="low",
        ),
        CompareDatasetModelScore(
            model="missing:model",
            mode="fast",
            appearances=3,
            passed=3,
            wins=3,
            score_delta=6.0,
            score_key="eval:fast",
            confidence="medium",
        ),
    ]

    report = apply_compare_scores_to_registry(
        scores,
        client.registry,
        min_count=2,
        min_confidence="medium",
    )

    reasons = {item["model"]: item["reason"] for item in report["skipped"]}
    assert "below min_count" in reasons[low_count_card.model_ref.key]
    assert "below min_confidence" in reasons[low_confidence_card.model_ref.key]
    assert "missing:model" in reasons["missing:model"]
    assert report["updated"] == []
    assert "eval:fast" not in low_count_card.local_eval_scores
    assert "eval:fast" not in low_confidence_card.local_eval_scores


def test_eval_residual_score_aggregation_and_threshold_helpers():
    variant = evals_module.CompareVariantResult(
        name="candidate",
        status="pass",
        ok=True,
        mode="research",
        models=["openai:gpt-5.5"],
        estimated_cost_usd=0.2,
        actual_cost_usd=0.1,
        estimated_latency_ms=200,
        latency_ms=150,
    )
    comparison = evals_module.CompareRunReport(
        task="Compare",
        dry_run=False,
        total=1,
        passed=1,
        failed=0,
        variants=[variant],
        winner="candidate",
    )
    cases = [
        evals_module.CompareDatasetCaseResult(id="empty", ok=False, winner=None, task="Empty"),
        evals_module.CompareDatasetCaseResult(
            id="scored",
            ok=True,
            winner="candidate",
            task="Compare",
            mode="research",
            comparison=comparison,
        ),
    ]

    scores = evals_module.aggregate_compare_scores(cases)

    assert scores[0].to_dict() == {
        "model": "openai:gpt-5.5",
        "mode": "research",
        "appearances": 1,
        "passed": 1,
        "wins": 1,
        "runs": 1,
        "avg_estimated_cost_usd": 0.2,
        "avg_actual_cost_usd": 0.1,
        "avg_estimated_latency_ms": 200,
        "avg_latency_ms": 150,
        "score_delta": 6.0,
        "score_key": "eval:research",
        "confidence": "low",
        "trend": "current",
    }
    assert evals_module._compare_score_delta(appearances=0, passed=0, wins=0) == 0.0
    assert evals_module._confidence_from_appearances(10) == "high"
    assert evals_module._avg_weighted([]) is None
    assert evals_module._avg_weighted_int([]) is None
    assert evals_module._trend([1.0]) == "insufficient"
    assert evals_module._trend([0.0, 1.0]) == "improving"
    assert evals_module._trend([1.0, 0.0]) == "declining"
    assert evals_module._trend([1.0, 1.1]) == "stable"


def test_eval_residual_dataset_formats_and_expectation_messages(tmp_path):
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '\n{"id":"one","task":"First"}\n{"id":"two","task":"Second"}\n',
        encoding="utf-8",
    )
    list_path = tmp_path / "cases.json"
    list_path.write_text('[{"id":"only","task":"Listed"}]', encoding="utf-8")

    jsonl_name, jsonl_cases = evals_module.load_eval_cases(jsonl_path)
    list_name, list_cases = evals_module.load_eval_cases(list_path)
    variant = CompareVariant.from_dict({"model": "candidate", "mode": "fast"})
    fallback_variant = CompareVariant.from_dict({})
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:gpt-5.5")],
        risk_level="low",
    )
    failures = evaluate_expectations(
        plan,
        {
            "strategy_in": ["fusion"],
            "risk_level": "high",
            "models_include": ["anthropic:missing"],
            "models_exclude": ["openai:gpt-5.5"],
            "providers_include": ["anthropic"],
            "max_models": 0,
            "roles_include": ["panel"],
            "roles_exclude": ["primary"],
        },
    )

    assert (jsonl_name, [case.id for case in jsonl_cases]) == ("cases", ["one", "two"])
    assert (list_name, [case.task for case in list_cases]) == ("cases", ["Listed"])
    assert variant.to_dict()["name"] == "candidate"
    assert fallback_variant.name == "variant"
    assert evaluate_expectations(plan, {}) == []
    assert failures == [
        "strategy expected one of ['fusion'], got 'single'",
        "risk_level expected 'high', got 'low'",
        "missing expected model 'anthropic:missing'",
        "unexpected model 'openai:gpt-5.5'",
        "missing expected provider 'anthropic'",
        "expected at most 0 models, got 1",
        "missing expected role 'panel'",
        "unexpected role 'primary'",
    ]


def test_eval_residual_compare_results_and_error_routes(tmp_path, monkeypatch):
    client = Crupier(make_config(tmp_path))
    runner = client.evals
    case = evals_module.EvalCase(id="broken", task="Break")

    def raise_deal(**_kwargs):
        raise RuntimeError("route exploded")

    monkeypatch.setattr(client, "deal", raise_deal)
    route_error = runner._run_case(case)
    compare_error = runner._run_compare_variant(
        task="Break",
        input=None,
        base_mode="fast",
        base_strategy=None,
        base_constraints={},
        variant=CompareVariant(name="broken"),
        response_schema=None,
        expect_contains=[],
        dry_run=True,
    )

    monkeypatch.setattr(client, "deal", lambda **_kwargs: CrupierResult(output_text="preview", route=None))
    no_route = runner._run_case(case)
    compare_no_route = runner._run_compare_variant(
        task="No route",
        input=None,
        base_mode="fast",
        base_strategy=None,
        base_constraints={},
        variant=CompareVariant(name="none"),
        response_schema=None,
        expect_contains=[],
        dry_run=True,
    )

    result = CrupierResult(
        output_text="Useful but incomplete",
        route=RoutePlan(
            strategy="single",
            steps=[RouteStep(role="primary", model="openai:gpt-5.5")],
            estimated_cost=CostEstimate(0.25),
            estimated_latency_ms=300,
        ),
        cost=CostEstimate(0.2, actual_usd=0.2),
        latency_ms=250,
    )
    monkeypatch.setattr(client, "deal", lambda **_kwargs: result)
    real_result = runner._run_compare_variant(
        task="Real",
        input=None,
        base_mode="fast",
        base_strategy="single",
        base_constraints={},
        variant=CompareVariant(name="real"),
        response_schema=None,
        expect_contains=["missing"],
        dry_run=False,
    )

    assert route_error.failed_checks == ["route_error"]
    assert route_error.error == "route exploded"
    assert compare_error.failed_checks == ["route_or_execution_error"]
    assert no_route.failed_checks == ["no_route_plan"]
    assert compare_no_route.output_preview == "preview"
    assert real_result.status == "fail"
    assert real_result.checks == {
        "route_planned": True,
        "non_empty_output": True,
        "contains:missing": False,
    }
    assert real_result.actual_cost_usd == 0.2
    assert real_result.latency_ms == 250
    assert "real output" in real_result.human_questions[-1]
    assert evals_module.recommend_compare_winner([real_result])[0] is None


def test_eval_residual_report_format_and_artifact_purge(tmp_path, monkeypatch):
    client = Crupier(make_config(tmp_path))
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps({"name": "written", "cases": [{"id": "one", "task": "Short", "mode": "fast"}]}),
        encoding="utf-8",
    )

    report = client.evals.compare_dataset(dataset=dataset, write_report=True)

    assert report.written_path is not None
    assert json.loads(Path(report.written_path).read_text(encoding="utf-8"))["name"] == "written"

    purge_root = tmp_path / "purge"
    purge_root.mkdir()
    (purge_root / "invalid.json").write_text("{", encoding="utf-8")
    (purge_root / "fresh.json").write_text('{"created_at":"fresh"}', encoding="utf-8")
    (purge_root / "expired.json").write_text('{"created_at":"expired"}', encoding="utf-8")
    (purge_root / "blocked.json").write_text('{"created_at":"expired"}', encoding="utf-8")
    monkeypatch.setattr(
        evals_module,
        "is_expired",
        lambda created_at, _ttl_days, *, fallback_path: created_at == "expired",
    )
    original_unlink = Path.unlink

    def selective_unlink(path, *args, **kwargs):
        if path.name == "blocked.json":
            raise OSError("busy")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", selective_unlink)

    assert evals_module.purge_eval_artifacts(purge_root, 1) == 1
    assert not (purge_root / "expired.json").exists()
    assert (purge_root / "blocked.json").exists()


def test_eval_residual_history_filters_and_malformed_records(tmp_path):
    assert evals_module.summarize_compare_history(tmp_path).warnings == ["No compare history found."]
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    history_path = history_dir / "compare_runs.jsonl"
    score = CompareDatasetModelScore(
        model="openai:gpt-5.5",
        mode="research",
        appearances=10,
        passed=8,
        wins=6,
        avg_estimated_cost_usd=0.4,
        avg_actual_cost_usd=0.3,
        avg_estimated_latency_ms=400,
        avg_latency_ms=350,
        score_delta=3.0,
        score_key="eval:research",
        confidence="high",
    )
    record = {"created_at": "2026-08-22T10:00:00Z", "model_scores": [score.to_dict()]}
    history_path.write_text(
        "\nnot-json\n[]\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    filtered_model = evals_module.summarize_compare_history(tmp_path, model="anthropic:missing")
    filtered_mode = evals_module.summarize_compare_history(tmp_path, mode="fast")
    summary = evals_module.summarize_compare_history(tmp_path, model=score.model, mode=score.mode)

    assert filtered_model.model_scores == []
    assert filtered_mode.model_scores == []
    assert summary.total_runs == 1
    assert summary.last_run_at == "2026-08-22T10:00:00Z"
    assert summary.model_scores[0].confidence == "high"
    assert summary.model_scores[0].avg_actual_cost_usd == 0.3
    assert summary.model_scores[0].trend == "insufficient"


def test_eval_residual_invalid_confidence_and_preview():
    try:
        evals_module.apply_compare_scores_to_registry([], object(), min_confidence="unknown")
    except ValueError as exc:
        assert str(exc) == "confidence must be low, medium, or high"
    else:
        raise AssertionError("invalid confidence should be rejected")

    assert evals_module._preview("  one   two  ", limit=20) == "one two"
    assert evals_module._preview("abcdefghij", limit=8) == "abcde..."


def test_eval_run_cli_outputs_json(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)

    exit_code = main(["--project", str(tmp_path), "eval", "run", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data["ok"] is True
    assert data["total"] == 5


def test_eval_compare_cli_outputs_json(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)

    exit_code = main(
        [
            "--project",
            str(tmp_path),
            "eval",
            "compare",
            "Answer briefly",
            "--model",
            "openai:gpt-5.5",
            "--model",
            "openai:gpt-5.4-mini",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data["winner"] == "openai:gpt-5.4-mini"
    assert data["total"] == 2


def test_eval_compare_dataset_cli_outputs_json_and_applies(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "cli-compare",
                "cases": [{"id": "fast", "task": "Answer briefly.", "mode": "fast"}],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
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
            "--apply",
            "--min-count",
            "1",
            "--min-confidence",
            "low",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data["name"] == "cli-compare"
    assert data["passed_cases"] == 1
    assert any(item["score_key"] == "eval:fast" for item in data["apply_report"]["updated"])


def test_eval_history_cli_outputs_recorded_history(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)
    dataset = tmp_path / "compare.json"
    dataset.write_text(
        json.dumps(
            {
                "name": "cli-history",
                "cases": [{"id": "fast", "task": "Answer briefly.", "mode": "fast"}],
            }
        ),
        encoding="utf-8",
    )

    record_status = main(
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
            "--record-history",
            "--json",
        ]
    )
    capsys.readouterr()
    history_status = main(["--project", str(tmp_path), "eval", "history", "--json"])
    captured = capsys.readouterr()

    assert record_status == 0
    assert history_status == 0
    data = json.loads(captured.out)
    assert data["total_runs"] == 1
    assert any(item["confidence"] == "low" for item in data["model_scores"])


def test_route_cli_budget_flag_blocks_over_budget(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)

    exit_code = main(["--project", str(tmp_path), "route", "Say hi", "--max-cost-usd", "0"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "exceeds max" in captured.err


def test_route_cli_response_schema_flag_outputs_input_plan(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)
    schema = '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}'

    exit_code = main(["--project", str(tmp_path), "route", "Extract name", "--response-schema", schema, "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data["strategy"] in {"cascade", "single"}
    assert data["estimated_cost"]["estimated_usd"] > 0


def test_route_cli_force_model_flag(tmp_path, capsys):
    from crupier.config import write_default_project

    write_default_project(tmp_path)

    exit_code = main(
        [
            "--project",
            str(tmp_path),
            "route",
            "Use exact model",
            "--force-model",
            "openai:gpt-5.4-mini",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data["strategy"] == "single"
    assert data["steps"][0]["model"] == "openai:gpt-5.4-mini"

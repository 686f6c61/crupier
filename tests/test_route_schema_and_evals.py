import json

from crupier import Crupier
from crupier.config import CrupierConfig
from crupier.errors import CrupierRouteValidationError
from crupier.evals import CompareVariant, RoutingEvalRunner, evaluate_expectations
from crupier.models import CostEstimate, RoutePlan, RouteStep
from crupier.route_schema import validate_route_plan_shape
from crupier.cli import main


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

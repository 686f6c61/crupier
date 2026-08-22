import json
from types import SimpleNamespace

import crupier.cli as cli_module
from crupier import Crupier
from crupier.cli import main


def write_project(path):
    (path / "crupier.toml").write_text(
        """
[project]
name = "control-plane-cli"
default_profile = "agentic"

[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[models]
allow = ["openai:gpt-5.5", "openai:gpt-5.4-mini"]

[routing]
default_strategy = "single"
require_operational_providers = false

[profiles.agentic]
prefer = ["quality"]
strategy = "single"

[experiments.rollout]
traffic = "shadow"
sample_rate = 1.0
execution = "plan_only"
candidate_models = ["openai:gpt-5.4-mini"]

[experiments.rollout.promotion]
min_samples = 1
max_error_rate = 1.0
max_error_rate_delta = 1.0
max_cost_ratio = 10.0
max_p95_latency_ratio = 10.0
confidence = 0.0
quality_check = "quality_delta"
require_quality_evaluator = true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_approval_and_session_commands_share_durable_state(tmp_path, capsys):
    write_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    prepared = client.prepare(
        "Apply reviewed deployment",
        constraints={"requires_human_approval": True},
        dry_run=True,
    )
    pending = client.request_approval(prepared)

    assert main(["--project", str(tmp_path), "approvals", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["approval_id"] == pending.approval_id

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "approve",
                pending.trace_id,
                "--reviewer",
                "ana",
                "--json",
            ]
        )
        == 0
    )
    granted = json.loads(capsys.readouterr().out)
    assert granted["status"] == "granted"
    assert granted["token"].startswith(f"{pending.approval_id}.")

    second = client.request_approval(prepared)
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "reject",
                second.approval_id,
                "--reviewer",
                "ana",
                "--reason",
                "Rollback is missing",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"

    session = client.session(persist=True)
    assert main(["--project", str(tmp_path), "sessions", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["session_id"] == session.session_id
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "sessions",
                "close",
                session.session_id,
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "closed"


def test_approvals_commands_text_output(tmp_path, capsys):
    write_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    prepared = client.prepare(
        "Apply reviewed deployment",
        constraints={"requires_human_approval": True},
        dry_run=True,
    )
    pending = client.request_approval(prepared)

    assert main(["--project", str(tmp_path), "approvals", "list"]) == 0
    listed = capsys.readouterr().out
    assert pending.approval_id in listed
    assert pending.status in listed

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "approvals",
                "show",
                pending.approval_id,
            ]
        )
        == 0
    )
    shown = capsys.readouterr().out
    assert f"approval_id: {pending.approval_id}" in shown
    assert f"status: {pending.status}" in shown

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "approvals",
                "show",
                pending.approval_id,
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["approval"]["status"] == "pending"

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "approve",
                pending.approval_id,
                "--reviewer",
                "ana",
            ]
        )
        == 0
    )
    granted = capsys.readouterr().out
    assert f"approval_id: {pending.approval_id}" in granted
    assert "status: granted" in granted
    assert "approval_token:" in granted

    second = client.request_approval(prepared)
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "reject",
                second.approval_id,
                "--reviewer",
                "ana",
                "--reason",
                "Rollback is missing",
            ]
        )
        == 0
    )
    rejected = capsys.readouterr().out
    assert f"approval_id: {second.approval_id}" in rejected
    assert "status: rejected" in rejected


def test_sessions_commands_text_output(tmp_path, capsys):
    write_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    session = client.session(persist=True)

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "sessions",
                "show",
                session.session_id,
            ]
        )
        == 0
    )
    shown = capsys.readouterr().out
    assert f"session_id: {session.session_id}" in shown
    assert "status: active" in shown
    assert "turns: 0" in shown

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "sessions",
                "show",
                session.session_id,
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["session"]["status"] == "active"

    assert main(["--project", str(tmp_path), "sessions", "list"]) == 0
    listed = capsys.readouterr().out
    assert session.session_id in listed
    assert "active" in listed
    assert "turns=0" in listed

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "sessions",
                "close",
                session.session_id,
            ]
        )
        == 0
    )
    closed = capsys.readouterr().out
    assert f"session_id: {session.session_id}" in closed
    assert "status: closed" in closed


def test_approval_execute_reads_token_only_from_environment(tmp_path, capsys, monkeypatch):
    write_project(tmp_path)
    monkeypatch.delenv("CRUPIER_APPROVAL_TOKEN", raising=False)

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "approvals",
                "execute",
            ]
        )
        == 1
    )
    assert "CRUPIER_APPROVAL_TOKEN" in capsys.readouterr().err


def test_approval_execute_text_and_json_output(capsys, monkeypatch):
    calls = []
    result = SimpleNamespace(
        output_text="approved output",
        route=SimpleNamespace(strategy="single", model_summary="openai:gpt-5.5"),
        to_dict=lambda *, trace_summary: {
            "output_text": "approved output",
            "trace_summary": trace_summary,
        },
    )
    client = SimpleNamespace(
        execute_approved=lambda token, *, trace: calls.append((token, trace)) or result
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)
    monkeypatch.setenv("CRUPIER_APPROVAL_TOKEN", "approval-token")

    assert main(["approvals", "execute", "--trace", "none"]) == 0
    output = capsys.readouterr().out
    assert "approved output" in output
    assert "route: single | openai:gpt-5.5" in output

    assert main(["approvals", "execute", "--trace", "debug", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["trace_summary"] is False
    assert calls == [("approval-token", False), ("approval-token", "debug")]


def test_experiment_cli_evidence_and_lifecycle(tmp_path, capsys):
    write_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    result = client.deal(
        "Compare rollout",
        constraints={"force_model": "openai:gpt-5.5"},
        dry_run=True,
        experiment="rollout",
    )
    assert result.experiment is not None

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "evaluate",
                result.experiment.observation_id,
                "--actor",
                "ana",
                "--checks",
                '{"quality_delta": 0.2}',
                "--json",
            ]
        )
        == 0
    )
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["checks"]["quality_delta"] == 0.2

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "report",
                "rollout",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["promotion"]["gates"]["live_execution_evidence"] is False

    for action in ("pause", "resume"):
        assert (
            main(
                [
                    "--project",
                    str(tmp_path),
                    "experiments",
                    action,
                    "rollout",
                    "--actor",
                    "ana",
                    "--json",
                ]
            )
            == 0
        )
        capsys.readouterr()

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "promote",
                "rollout",
                "--actor",
                "ana",
                "--force",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "promoted"

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "rollback",
                "rollout",
                "--actor",
                "ana",
                "--reason",
                "Latency regression",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "rolled_back"


def test_experiments_report_text_output(tmp_path, capsys):
    write_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    result = client.deal(
        "Compare rollout",
        constraints={"force_model": "openai:gpt-5.5"},
        dry_run=True,
        experiment="rollout",
    )
    assert result.experiment is not None

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "evaluate",
                result.experiment.observation_id,
                "--actor",
                "ana",
                "--checks",
                "not-json",
            ]
        )
        == 1
    )
    assert "--checks must be valid JSON" in capsys.readouterr().err

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "evaluate",
                result.experiment.observation_id,
                "--actor",
                "ana",
                "--checks",
                "[]",
            ]
        )
        == 1
    )
    assert "--checks must be a JSON object" in capsys.readouterr().err

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "evaluate",
                result.experiment.observation_id,
                "--actor",
                "ana",
                "--checks",
                '{"quality_delta": 0.2}',
            ]
        )
        == 0
    )
    evaluated = capsys.readouterr().out
    assert f"observation_id: {result.experiment.observation_id}" in evaluated
    assert "quality_delta" in evaluated

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "report",
                "rollout",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "experiment: rollout" in output
    assert "primary: count=1" in output
    assert "candidate: count=1" in output
    assert "paired_count: 1" in output

    for action, status in (("pause", "paused"), ("resume", "active")):
        assert (
            main(
                [
                    "--project",
                    str(tmp_path),
                    "experiments",
                    action,
                    "rollout",
                    "--actor",
                    "ana",
                ]
            )
            == 0
        )
        action_output = capsys.readouterr().out
        assert "experiment: rollout" in action_output
        assert f"status: {status}" in action_output

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "promote",
                "rollout",
                "--actor",
                "ana",
                "--force",
            ]
        )
        == 0
    )
    promoted = capsys.readouterr().out
    assert "experiment: rollout" in promoted
    assert "status: promoted" in promoted

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "experiments",
                "rollback",
                "rollout",
                "--actor",
                "ana",
                "--reason",
                "Latency regression",
            ]
        )
        == 0
    )
    rolled_back = capsys.readouterr().out
    assert "experiment: rollout" in rolled_back
    assert "status: rolled_back" in rolled_back


def test_control_plane_cli_json_no_route_and_trace_diagnostics(
    tmp_path, capsys, monkeypatch
):
    write_project(tmp_path)

    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "deal",
                "Route this task",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["route"]["strategy"] == "single"

    class TraceRefs(list):
        def __init__(self, items):
            super().__init__(items)
            self.diagnostics = [
                SimpleNamespace(path="broken.json", line=7, error_type="JSONDecodeError")
            ]
            self.complete = False

    refs = TraceRefs(
        [
            SimpleNamespace(
                trace_id="trc_1",
                created_at="now",
                strategy="single",
                models=["openai:gpt-5.5"],
                replayable=True,
                summary="task",
                to_dict=lambda: {"trace_id": "trc_1"},
            )
        ]
    )
    client = SimpleNamespace(
        deal=lambda **kwargs: SimpleNamespace(route=None),
        traces=SimpleNamespace(list=lambda: refs),
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)

    assert main(["route", "No route available"]) == 1
    assert "No route planned." in capsys.readouterr().out

    assert main(["trace", "list", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == [{"trace_id": "trc_1"}]
    assert "broken.json:7 (JSONDecodeError)" in captured.err

    assert main(["trace", "list"]) == 1
    captured = capsys.readouterr()
    assert "trc_1\tnow\tsingle" in captured.out
    assert "broken.json:7 (JSONDecodeError)" in captured.err


def test_update_and_model_show_cover_human_facing_edges(tmp_path, capsys, monkeypatch):
    blocked = SimpleNamespace(adapters={})
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: blocked)

    assert main(["update", "--provider", "openai"]) == 1
    assert "Provider 'openai' is not enabled" in capsys.readouterr().err

    report = SimpleNamespace(to_dict=lambda: {"status": "ready"})
    configured = SimpleNamespace(
        adapters={"openai": object()},
        update=lambda **kwargs: report,
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: configured)

    assert main(["update", "--provider", "openai", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"

    model_ref = SimpleNamespace(
        key="openai:gpt-5.5",
        provider="openai",
        stability="stable",
    )
    card = SimpleNamespace(
        model_ref=model_ref,
        model_kind="chat",
        routing_hints={"routing_status": "recommended", "production_default": True},
        quality_tier="high",
        cost_tier="high",
        latency_tier="medium",
        modalities_input=["text"],
        modalities_output=["text"],
        natural_profile={
            "summary": "Frontier model",
            "status_reason": "Current default",
            "replacement": "openai:gpt-5.6",
        },
        skill_scores={"reasoning": 9.5, "ignored": "unknown"},
        to_dict=lambda: {"model": "openai:gpt-5.5"},
    )
    client = SimpleNamespace(
        models=SimpleNamespace(get=lambda model: card),
        registry=SimpleNamespace(
            model_states=lambda **kwargs: [{"states": ["allowed", "recommended"]}]
        ),
    )
    monkeypatch.setattr(cli_module.Crupier, "from_project", lambda project: client)

    assert main(["models", "show", "openai:gpt-5.5"]) == 0
    output = capsys.readouterr().out
    assert "replacement: openai:gpt-5.6" in output
    assert "top_skills: reasoning=9.5" in output

    filtered_cards = [
        SimpleNamespace(
            model_ref=SimpleNamespace(provider="openai", stability="stable"),
            model_kind="image",
            routing_hints={"routing_status": "recommended"},
        ),
        SimpleNamespace(
            model_ref=SimpleNamespace(provider="openai", stability="stable"),
            model_kind="chat",
            routing_hints={"routing_status": "legacy"},
        ),
        SimpleNamespace(
            model_ref=SimpleNamespace(provider="openai", stability="deprecated"),
            model_kind="chat",
            routing_hints={"routing_status": "recommended"},
        ),
    ]
    filter_args = SimpleNamespace(
        provider=None,
        kind="chat",
        status="recommended",
        recommended=False,
        include_deprecated=False,
    )
    assert cli_module._filter_model_cards(filtered_cards, filter_args) == []

    write_project(tmp_path)
    assert main(["--project", str(tmp_path), "orchestrator", "show"]) == 0
    orchestrator_output = capsys.readouterr().out
    assert "mode:" in orchestrator_output
    assert "max_repairs:" in orchestrator_output

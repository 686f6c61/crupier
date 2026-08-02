import json

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

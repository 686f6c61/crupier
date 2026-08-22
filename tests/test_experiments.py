import json
from threading import Lock
from time import sleep

import pytest

from crupier import Crupier
from crupier.adapters import AdapterResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierError, CrupierProviderUnavailableError
from crupier.experiments import (
    ExperimentReport,
    PromotionReport,
    _aggregate_metrics,
    _ExperimentStateUnavailable,
    _percentile,
    _promotion_report,
    _ratio_gate,
    _result_metrics,
    _sample,
    _sticky_value,
    _validated_checks,
    _wilson_upper,
)
from crupier.models import ExperimentObservation


class ExperimentAdapter:
    provider = "openai"

    def __init__(self, *, fail_models=None, slow_models=None):
        self.fail_models = set(fail_models or [])
        self.slow_models = set(slow_models or [])
        self.calls = []
        self._lock = Lock()

    def generate(self, *, model, prompt, request):
        with self._lock:
            self.calls.append(model)
        if model in self.slow_models:
            sleep(0.05)
        if model in self.fail_models:
            raise CrupierProviderUnavailableError(f"{model} failed")
        if request.tools:
            text = json.dumps({"tool_calls": [], "final": f"{model}-done"})
        else:
            text = f"{model}-answer"
        return AdapterResponse(
            text=text,
            usage={"input_tokens": 2, "output_tokens": 2},
            metadata={"provider": "openai", "model": model},
        )


def make_client(tmp_path, *, adapter=None, evaluator=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "experiments", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {
                "allow": ["openai:gpt-5.5", "openai:gpt-5.4-mini"],
            },
            "routing": {
                "default_strategy": "single",
                "max_provider_retries": 0,
                "circuit_breaker_failure_threshold": 5,
            },
            "profiles": {
                "agentic": {"prefer": ["quality"], "strategy": "single"},
            },
            "experiments": {
                "shadow-sync": {
                    "traffic": "shadow",
                    "sample_rate": 1.0,
                    "execution": "sync",
                    "candidate_models": ["openai:gpt-5.4-mini"],
                    "max_shadow_cost_usd": 1.0,
                    "promotion": {
                        "action": "recommend",
                        "min_samples": 2,
                        "max_error_rate": 1.0,
                        "max_error_rate_delta": 1.0,
                        "min_quality_delta": 0.0,
                        "quality_check": "domain_quality_delta",
                        "max_cost_ratio": 10.0,
                        "max_p95_latency_ratio": 10.0,
                        "confidence": 0.0,
                    },
                },
                "shadow-plan": {
                    "traffic": "shadow",
                    "sample_rate": 1.0,
                    "execution": "plan_only",
                    "candidate_models": ["openai:gpt-5.4-mini"],
                },
                "shadow-async": {
                    "traffic": "shadow",
                    "sample_rate": 1.0,
                    "execution": "async",
                    "candidate_models": ["openai:gpt-5.4-mini"],
                },
                "canary": {
                    "traffic": "canary",
                    "sample_rate": 1.0,
                    "execution": "sync",
                    "candidate_models": ["openai:gpt-5.4-mini"],
                },
                "half": {
                    "traffic": "canary",
                    "sample_rate": 0.5,
                    "candidate_models": ["openai:gpt-5.4-mini"],
                },
            },
        }
    )
    config.root = tmp_path
    return Crupier(
        config,
        adapters={"openai": adapter or ExperimentAdapter()},
        experiment_evaluator=evaluator,
    )


def base_constraints():
    return {"force_model": "openai:gpt-5.5"}


def test_shadow_sync_returns_primary_and_records_candidate_diffs_without_outputs(tmp_path):
    adapter = ExperimentAdapter()
    client = make_client(tmp_path, adapter=adapter)

    result = client.deal(
        "Compare architectures",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert set(adapter.calls) == {"gpt-5.5", "gpt-5.4-mini"}
    assert result.experiment is not None
    assert result.experiment.status == "completed"
    assert result.experiment.primary["models"] == ["openai:gpt-5.5"]
    assert result.experiment.candidate["models"] == ["openai:gpt-5.4-mini"]
    assert "output" not in result.experiment.primary
    assert "output" not in result.experiment.candidate
    assert result.experiment.diffs["same_output_hash"] is False
    assert len(result.experiment.checks["config_hash"]) == 16
    stored = client.experiments.observation(result.experiment.observation_id)
    assert stored.to_dict() == result.experiment.to_dict()


def test_shadow_plan_only_cannot_be_promoted_as_live_evidence(tmp_path):
    adapter = ExperimentAdapter()
    client = make_client(tmp_path, adapter=adapter)

    result = client.deal(
        "Plan only comparison",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-plan",
    )

    assert adapter.calls == ["gpt-5.5"]
    assert result.experiment is not None
    assert result.experiment.candidate["dry_run"] is True
    report = client.experiments.report("shadow-plan")
    assert report.promotion.eligible is False
    assert report.promotion.gates["live_execution_evidence"] is False


def test_shadow_async_returns_scheduled_reference_and_finishes_in_background(tmp_path):
    adapter = ExperimentAdapter(slow_models={"gpt-5.4-mini"})
    client = make_client(tmp_path, adapter=adapter)

    result = client.deal(
        "Async comparison",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-async",
    )

    assert result.experiment is not None
    assert result.experiment.status == "scheduled"
    completed = client.experiments.wait(
        result.experiment.observation_id,
        timeout_seconds=2,
    )
    assert completed.status == "completed"
    assert completed.candidate["models"] == ["openai:gpt-5.4-mini"]


def test_canary_serves_candidate_and_falls_back_to_baseline_on_failure(tmp_path):
    healthy = make_client(tmp_path / "healthy", adapter=ExperimentAdapter())
    candidate = healthy.deal(
        "Canary request",
        constraints=base_constraints(),
        dry_run=False,
        experiment="canary",
    )

    assert candidate.output_text == "gpt-5.4-mini-answer"
    assert candidate.experiment is not None
    assert candidate.experiment.cohort == "candidate"
    assert candidate.experiment.status == "candidate_completed"

    failing = make_client(
        tmp_path / "failing",
        adapter=ExperimentAdapter(fail_models={"gpt-5.4-mini"}),
    )
    baseline = failing.deal(
        "Canary fallback",
        constraints=base_constraints(),
        dry_run=False,
        experiment="canary",
    )

    assert baseline.output_text == "gpt-5.5-answer"
    assert baseline.experiment is not None
    assert baseline.experiment.status == "failed"
    assert "failed" in baseline.experiment.error
    assert any("baseline served" in warning for warning in baseline.warnings)
    report = failing.experiments.report("canary")
    assert report.metrics["candidate"]["errors"] == 1


def test_shadow_side_effect_tools_are_forced_to_plan_only(tmp_path):
    adapter = ExperimentAdapter()
    client = make_client(tmp_path, adapter=adapter)

    result = client.deal(
        "Prepare a write",
        constraints={**base_constraints(), "approve_tool_calls": True},
        tools=[
            {
                "name": "write_record",
                "description": "Write a record.",
                "side_effects": True,
                "handler": lambda: "written",
            }
        ],
        dry_run=True,
        experiment="shadow-sync",
    )

    assert result.experiment is not None
    assert result.experiment.candidate["dry_run"] is True
    assert any("side-effecting" in warning for warning in result.experiment.warnings)
    assert adapter.calls == []


def test_sampling_is_sticky_and_session_metadata_keeps_one_cohort(tmp_path):
    client = make_client(tmp_path)
    first = client.deal(
        "Sticky",
        constraints=base_constraints(),
        metadata={"session_id": "ses-fixed"},
        dry_run=False,
        experiment="half",
    )
    second = client.deal(
        "Different turn",
        constraints=base_constraints(),
        metadata={"session_id": "ses-fixed"},
        dry_run=False,
        experiment="half",
    )

    assert first.experiment is not None and second.experiment is not None
    assert first.experiment.cohort == second.experiment.cohort
    assert _sample("half", '"ses-fixed"', 0.5) == first.experiment.sampled


def test_promotion_and_rollback_are_gated_recorded_and_reversible(tmp_path):
    client = make_client(
        tmp_path,
        evaluator=lambda primary, candidate: {"domain_quality_delta": 0.1},
    )
    for index in range(2):
        client.deal(
            f"Evidence {index}",
            constraints=base_constraints(),
            dry_run=False,
            experiment="shadow-sync",
        )

    report = client.experiments.report("shadow-sync")
    assert report.promotion.eligible is True
    promoted = client.experiments.promote("shadow-sync", actor="ana")
    assert promoted.status == "promoted"

    served = client.deal(
        "After promotion",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )
    assert served.output_text == "gpt-5.4-mini-answer"
    assert served.experiment is not None
    assert served.experiment.cohort == "promoted"

    rolled_back = client.experiments.rollback(
        "shadow-sync",
        actor="ana",
        reason="Latency regression",
    )
    assert rolled_back.status == "rolled_back"
    baseline = client.deal(
        "After rollback",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )
    assert baseline.output_text == "gpt-5.5-answer"
    assert baseline.experiment is not None
    assert baseline.experiment.status == "disabled"


def test_experiment_validation_and_custom_evaluator(tmp_path):
    client = make_client(
        tmp_path,
        evaluator=lambda primary, candidate: {"domain_quality_delta": 0.25},
    )
    result = client.deal(
        "Evaluate",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )
    assert result.experiment is not None
    assert result.experiment.checks["domain_quality_delta"] == 0.25

    recorded = client.experiments.record_evaluation(
        result.experiment.observation_id,
        {"human_review_score": 0.9},
        actor="ana",
    )
    assert recorded.checks["human_review_score"] == 0.9

    with pytest.raises(CrupierError, match="not configured"):
        client.deal("Unknown", dry_run=True, experiment="missing")
    with pytest.raises(CrupierError, match="actor cannot be empty"):
        client.experiments.promote("shadow-sync", actor="", force=True)
    with pytest.raises(CrupierError, match="Evaluation actor"):
        client.experiments.record_evaluation(
            result.experiment.observation_id,
            {"quality": 1.0},
            actor="",
        )


def test_evaluator_failure_is_contained_and_quality_gate_requires_real_evidence(tmp_path):
    client = make_client(
        tmp_path / "broken",
        evaluator=lambda primary, candidate: (_ for _ in ()).throw(
            RuntimeError("evaluator unavailable Bearer abcdefghijklmnop")
        ),
    )

    result = client.deal(
        "Do not break primary traffic",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert result.experiment is not None
    assert any("evaluator unavailable" in item for item in result.experiment.warnings)
    assert all("abcdefghijklmnop" not in item for item in result.experiment.warnings)
    assert client.experiments.report("shadow-sync").promotion.gates["quality"] is False

    no_evaluator = make_client(tmp_path / "no-evaluator")
    for index in range(2):
        no_evaluator.deal(
            f"Evidence {index}",
            constraints=base_constraints(),
            dry_run=False,
            experiment="shadow-sync",
        )
    report = no_evaluator.experiments.report("shadow-sync")
    assert report.promotion.gates["min_samples"] is True
    assert report.promotion.gates["quality"] is False
    assert report.promotion.eligible is False

    with pytest.raises(CrupierError, match="finite JSON-compatible"):
        no_evaluator.experiments.record_evaluation(
            no_evaluator.deal(
                "Another",
                constraints=base_constraints(),
                dry_run=False,
                experiment="shadow-sync",
            ).experiment.observation_id,
            {"quality_delta": float("nan")},
            actor="ana",
        )


def test_experiment_pause_resume_and_evaluation_state_validation(tmp_path):
    client = make_client(tmp_path)
    paused = client.experiments.pause("shadow-sync", actor="ana")
    assert paused.status == "paused"
    resumed = client.experiments.resume("shadow-sync", actor="ana")
    assert resumed.status == "active"

    with pytest.raises(CrupierError, match="Pause actor"):
        client.experiments.pause("shadow-sync", actor="")

    slow = make_client(
        tmp_path / "slow",
        adapter=ExperimentAdapter(slow_models={"gpt-5.4-mini"}),
    )
    result = slow.deal(
        "Async",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-async",
    )
    assert result.experiment is not None
    with pytest.raises(CrupierError, match="running experiment observation"):
        slow.experiments.record_evaluation(
            result.experiment.observation_id,
            {"quality_delta": 0.1},
            actor="ana",
        )
    slow.experiments.wait(result.experiment.observation_id, timeout_seconds=2)


def test_experiment_state_failure_fails_closed_to_baseline(tmp_path, monkeypatch):
    adapter = ExperimentAdapter()
    client = make_client(tmp_path, adapter=adapter)

    def fail_state(*args, **kwargs):
        raise OSError("disk unavailable Bearer abcdefghijklmnop")

    monkeypatch.setattr(client.experiments.store, "get", fail_state)
    result = client.deal(
        "Keep live traffic available",
        constraints=base_constraints(),
        dry_run=False,
        experiment="canary",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert adapter.calls == ["gpt-5.5"]
    assert result.experiment is not None
    assert result.experiment.status == "control_plane_unavailable"
    assert all("abcdefghijklmnop" not in warning for warning in result.warnings)
    assert any("baseline served" in warning for warning in result.warnings)


def test_experiment_state_update_failure_never_replaces_live_result(tmp_path, monkeypatch):
    adapter = ExperimentAdapter()
    client = make_client(tmp_path, adapter=adapter)
    client.experiments._runtime_status("shadow-sync")

    def fail_transition(*args, **kwargs):
        raise OSError("read only filesystem")

    monkeypatch.setattr(client.experiments.store, "transition", fail_transition)
    result = client.deal(
        "Return the primary even when telemetry cannot persist",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert set(adapter.calls) == {"gpt-5.5", "gpt-5.4-mini"}
    assert result.experiment is not None
    assert any("state update failed" in warning for warning in result.experiment.warnings)


def test_stored_experiment_output_rejects_non_json_values(tmp_path):
    client = make_client(tmp_path)
    result = client.deal(
        "Structured output",
        constraints=base_constraints(),
        dry_run=False,
    )
    result.output_json = {"bad": object()}

    metrics = _result_metrics(result, store_output=True)

    assert metrics["output_json"] is None
    assert metrics["output_json_serialization_error"] is True


def test_unsampled_shadow_and_canary_only_execute_the_baseline(tmp_path):
    for name in ("shadow-sync", "half"):
        adapter = ExperimentAdapter()
        client = make_client(tmp_path / name, adapter=adapter)
        client.config.experiments[name].sample_rate = 0.0

        result = client.deal(
            "Control cohort",
            constraints=base_constraints(),
            dry_run=False,
            experiment=name,
        )

        assert result.output_text == "gpt-5.5-answer"
        assert adapter.calls == ["gpt-5.5"]
        assert result.experiment is not None
        assert result.experiment.cohort == "control"
        assert result.experiment.status == "control_completed"


def test_shadow_candidate_failures_are_contained_in_sync_and_async_modes(tmp_path):
    for name in ("shadow-sync", "shadow-async"):
        client = make_client(
            tmp_path / name,
            adapter=ExperimentAdapter(fail_models={"gpt-5.4-mini"}),
        )
        result = client.deal(
            "Candidate may fail",
            constraints=base_constraints(),
            dry_run=False,
            experiment=name,
        )

        assert result.output_text == "gpt-5.5-answer"
        assert result.experiment is not None
        observation = (
            client.experiments.wait(result.experiment.observation_id, timeout_seconds=2)
            if name == "shadow-async"
            else result.experiment
        )
        assert observation.status == "failed"
        assert "gpt-5.4-mini failed" in observation.error
        report = client.experiments.report(name)
        assert report.metrics["candidate"]["count"] == 1
        assert report.metrics["candidate"]["errors"] == 1
        assert report.metrics["candidate"]["error_rate"] == 1.0
        assert report.metrics["primary"]["errors"] == 0


def test_primary_failure_is_recorded_and_propagated(tmp_path):
    client = make_client(
        tmp_path,
        adapter=ExperimentAdapter(fail_models={"gpt-5.5"}),
    )

    with pytest.raises(CrupierProviderUnavailableError, match="gpt-5.5 failed"):
        client.deal(
            "Primary failure",
            constraints=base_constraints(),
            dry_run=False,
            experiment="shadow-sync",
        )

    report = client.experiments.report("shadow-sync")
    assert report.failed == 1
    assert report.metrics["primary"]["count"] == 1
    assert report.metrics["primary"]["errors"] == 1
    assert report.metrics["candidate"]["count"] == 0
    assert report.metrics["candidate"]["errors"] == 0


def test_canary_records_candidate_and_baseline_failures_separately(tmp_path):
    client = make_client(
        tmp_path,
        adapter=ExperimentAdapter(
            fail_models={"gpt-5.5", "gpt-5.4-mini"},
        ),
    )

    with pytest.raises(CrupierProviderUnavailableError, match="gpt-5.5 failed"):
        client.deal(
            "Both routes fail",
            constraints=base_constraints(),
            dry_run=False,
            experiment="canary",
        )

    report = client.experiments.report("canary")
    assert report.metrics["primary"]["count"] == 1
    assert report.metrics["primary"]["errors"] == 1
    assert report.metrics["candidate"]["count"] == 1
    assert report.metrics["candidate"]["errors"] == 1
    observation = client.experiments.store.list("experiment_observation")[0]
    checks = observation.payload["observation"]["checks"]
    assert checks["failure_role"] == "both"


def test_promoted_candidate_failure_falls_back_to_baseline(tmp_path):
    client = make_client(
        tmp_path,
        adapter=ExperimentAdapter(fail_models={"gpt-5.4-mini"}),
    )
    client.experiments.promote("shadow-sync", actor="ana", force=True)

    result = client.deal(
        "Promoted fallback",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert result.experiment is not None
    assert result.experiment.status == "failed"
    assert any("Promoted candidate failed" in warning for warning in result.warnings)


def test_disabled_experiment_state_creation_failure_keeps_baseline(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client.experiments._runtime_status("shadow-sync")
    client.config.experiments["shadow-sync"].enabled = False

    def fail_create(*args, **kwargs):
        raise OSError("read only state")

    monkeypatch.setattr(client.experiments.store, "create", fail_create)
    result = client.deal(
        "Disabled experiment",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert result.experiment is not None
    assert result.experiment.status == "control_plane_unavailable"


def test_auto_promotion_failure_is_warning_not_live_failure(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client.config.experiments["shadow-sync"].promotion.action = "auto"

    def fail_report(*args, **kwargs):
        raise OSError("promotion database unavailable")

    monkeypatch.setattr(client.experiments, "report", fail_report)
    result = client.deal(
        "Automatic policy",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )

    assert result.output_text == "gpt-5.5-answer"
    assert any("Automatic promotion failed closed" in warning for warning in result.warnings)


def test_experiment_validation_and_metric_helpers_cover_boundary_values(tmp_path):
    client = make_client(tmp_path)
    settings = client.config.experiments["shadow-sync"]
    observation = client.experiments._create_observation(
        settings,
        cohort="shadow",
        sampled=True,
        sticky_value="timeout",
    )
    assert (
        client.experiments.wait(observation.observation_id, timeout_seconds=-1).status
        == "running"
    )

    with pytest.raises(CrupierError, match="cannot be empty"):
        client.experiments.record_evaluation(
            observation.observation_id,
            {},
            actor="ana",
        )
    with pytest.raises(CrupierError, match="non-empty strings"):
        client.experiments.record_evaluation(
            observation.observation_id,
            {"": 1.0},
            actor="ana",
        )
    with pytest.raises(CrupierError, match="dictionary"):
        _validated_checks(["not", "a", "mapping"])

    assert _ratio_gate(2.0, 1.0, None) is True
    assert _ratio_gate(None, 1.0, 2.0) is None
    assert _ratio_gate(0.0, 0.0, 2.0) is True
    assert _wilson_upper(0, 0, 0.95) == 1.0
    assert _percentile([], 0.95) is None
    assert _sample("zero", "key", 0.0) is False
    assert _sample("all", "key", 1.0) is True


def test_experiment_manager_supports_non_blocking_shutdown(tmp_path):
    client = make_client(tmp_path)

    client.experiments.close(wait=False)


@pytest.mark.parametrize(
    ("experiment", "message"),
    [
        ({"traffic": "invalid"}, "traffic"),
        ({"max_concurrency": 65}, "at most 64"),
        ({"promotion": {"quality_check": ""}}, "quality_check"),
        ({"unknown": True}, "unknown field"),
    ],
)
def test_experiment_configuration_fails_closed(experiment, message):
    with pytest.raises(CrupierError, match=message):
        CrupierConfig.from_dict({"experiments": {"rollout": experiment}})


def test_experiment_run_recovers_when_observation_state_initialization_fails(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client.experiments._runtime_status("shadow-sync")

    def fail_run(*args, **kwargs):
        raise _ExperimentStateUnavailable("state unavailable")

    monkeypatch.setattr(client.experiments, "_run_shadow", fail_run)
    result = client.deal(
        "Fallback after state failure",
        constraints=base_constraints(),
        dry_run=False,
        experiment="shadow-sync",
    )
    assert result.output_text == "gpt-5.5-answer"
    assert result.experiment.status == "control_plane_unavailable"


def test_experiment_lifecycle_rejects_ineligible_and_missing_actors(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(CrupierError, match="not eligible"):
        client.experiments.promote("shadow-sync", actor="ana")
    with pytest.raises(CrupierError, match="Rollback actor"):
        client.experiments.rollback("shadow-sync", actor="", reason="rollback")
    with pytest.raises(CrupierError, match="Resume actor"):
        client.experiments.resume("shadow-sync", actor="")


@pytest.mark.parametrize("name", ["shadow-sync", "half"])
def test_unsampled_baseline_failures_are_recorded_for_both_traffic_modes(tmp_path, name):
    client = make_client(
        tmp_path / name,
        adapter=ExperimentAdapter(fail_models={"gpt-5.5"}),
    )
    client.config.experiments[name].sample_rate = 0.0
    with pytest.raises(CrupierProviderUnavailableError, match="gpt-5.5 failed"):
        client.deal(
            "Fail control cohort",
            constraints=base_constraints(),
            dry_run=False,
            experiment=name,
        )
    assert client.experiments.report(name).failed == 1


def test_canary_auto_promotion_failure_is_attached_to_candidate(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client.config.experiments["canary"].promotion.action = "auto"
    monkeypatch.setattr(
        client.experiments,
        "report",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("report unavailable")),
    )
    result = client.deal(
        "Canary warning",
        constraints=base_constraints(),
        dry_run=False,
        experiment="canary",
    )
    assert any("Automatic promotion failed closed" in warning for warning in result.warnings)
    assert result.experiment.warnings == result.warnings


def test_promoted_candidate_and_baseline_failure_is_propagated(tmp_path):
    client = make_client(
        tmp_path,
        adapter=ExperimentAdapter(fail_models={"gpt-5.5", "gpt-5.4-mini"}),
    )
    client.experiments.promote("shadow-sync", actor="ana", force=True)
    with pytest.raises(CrupierProviderUnavailableError, match="gpt-5.5 failed"):
        client.deal(
            "Both promoted routes fail",
            constraints=base_constraints(),
            dry_run=False,
            experiment="shadow-sync",
        )
    assert client.experiments.report("shadow-sync").failed == 1


def test_candidate_kwargs_cover_strategy_approval_and_valid_structured_output(tmp_path):
    client = make_client(tmp_path)
    settings = client.config.experiments["shadow-sync"]
    settings.candidate_strategy = "single"
    candidate, warnings = client.experiments._candidate_kwargs(
        settings,
        {
            "task": "approval",
            "constraints": {"requires_human_approval": True},
        },
        shadow=True,
    )
    assert candidate["strategy"] == "single"
    assert candidate["dry_run"] is True
    assert any("requires human approval" in warning for warning in warnings)

    result = client.deal("Structured", constraints=base_constraints(), dry_run=False)
    result.output_json = {"answer": 42}
    assert _result_metrics(result, store_output=True)["output_json"] == {"answer": 42}


def test_canary_evaluator_failure_is_contained(tmp_path):
    client = make_client(
        tmp_path,
        evaluator=lambda primary, candidate: (_ for _ in ()).throw(RuntimeError("bad evaluator")),
    )
    result = client.deal(
        "Candidate evaluator",
        constraints=base_constraints(),
        dry_run=False,
        experiment="canary",
    )
    assert any("Experiment evaluator failed" in warning for warning in result.experiment.warnings)


def test_runtime_record_handles_concurrent_activation(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    original_get = client.experiments.store.get
    client.experiments._runtime_status("shadow-sync")
    calls = 0

    def race_get(kind, record_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CrupierError("missing in stale snapshot")
        return original_get(kind, record_id)

    monkeypatch.setattr(client.experiments.store, "get", race_get)
    monkeypatch.setattr(
        client.experiments.store,
        "create",
        lambda **kwargs: (_ for _ in ()).throw(CrupierError("already created")),
    )
    assert client.experiments._runtime_record("shadow-sync").status == "active"


def test_auto_promotion_executes_once_and_sticky_metadata_uses_fallback(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    settings = client.config.experiments["shadow-sync"]
    settings.promotion.action = "auto"
    report = ExperimentReport(
        experiment=settings.name,
        status="active",
        observations=2,
        sampled=2,
        completed=2,
        failed=0,
        cohorts={"shadow": 2},
        metrics={},
        promotion=PromotionReport(settings.name, True, 2, {}, {}),
    )
    promoted = []
    monkeypatch.setattr(client.experiments, "report", lambda name: report)
    monkeypatch.setattr(client.experiments, "promote", lambda name, actor: promoted.append((name, actor)))
    assert client.experiments._maybe_auto_promote(settings) is None
    assert promoted == [("shadow-sync", "automatic-policy")]
    assert _sticky_value(settings, {"metadata": {"tenant_id": "tenant-1"}}) == '"tenant-1"'


def test_promotion_uses_cohort_quality_and_reports_cost_failure(tmp_path):
    client = make_client(tmp_path)
    settings = client.config.experiments["shadow-sync"]
    settings.promotion.require_quality_evaluator = False
    settings.promotion.min_samples = 1
    settings.promotion.max_cost_ratio = 1.0
    settings.promotion.max_p95_latency_ratio = 1.0
    observation = ExperimentObservation(
        observation_id="obs_quality",
        experiment=settings.name,
        traffic="shadow",
        cohort="shadow",
        sampled=True,
        status="completed",
        primary={"quality_score": 0.5, "actual_cost_usd": 1.0, "latency_ms": 100},
        candidate={"quality_score": 0.8, "actual_cost_usd": 2.0, "latency_ms": 200},
    )
    metrics = _aggregate_metrics([observation])
    promotion = _promotion_report(settings, [observation], metrics)
    assert promotion.gates["quality"] is True
    assert promotion.gates["cost"] is False
    assert promotion.gates["latency"] is False
    assert "candidate cost ratio exceeds policy" in promotion.reasons
    assert "candidate p95 latency ratio exceeds policy" in promotion.reasons

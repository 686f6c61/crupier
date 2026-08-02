"""Plan a sampled shadow route and attach external quality evidence."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from _example_support import offline_client, print_route


def main() -> None:
    with TemporaryDirectory(prefix="crupier-rollout-example-") as temporary:
        crupier = offline_client(
            project="shadow-rollout",
            profile="agentic",
            allow=["openai:gpt-5.5", "openai:gpt-5.4-mini"],
            root=temporary,
            experiments={
                "support-rollout": {
                    "traffic": "shadow",
                    "sample_rate": 1.0,
                    "execution": "plan_only",
                    "candidate_models": ["openai:gpt-5.4-mini"],
                    "candidate_strategy": "single",
                    "promotion": {
                        "min_samples": 20,
                        "max_error_rate": 0.02,
                        "max_error_rate_delta": 0.01,
                        "quality_check": "quality_delta",
                        "require_quality_evaluator": True,
                    },
                }
            },
        )
        result = crupier.deal(
            "Draft a concise support response.",
            input={"ticket_id": "T-104", "message": "Where is my invoice?"},
            constraints={"force_model": "openai:gpt-5.5"},
            metadata={"session_id": "example-session"},
            dry_run=True,
            trace="summary",
            experiment="support-rollout",
        )
        if result.experiment is None:
            raise RuntimeError("Shadow observation was not attached")
        observation = crupier.experiments.record_evaluation(
            result.experiment.observation_id,
            {"quality_delta": 0.1, "reviewed_cases": 1},
            actor="example-reviewer",
        )
        report = crupier.experiments.report("support-rollout")

        print_route(
            "shadow_canary_rollout",
            result,
            extra={
                "cohort": observation.cohort,
                "observation_status": observation.status,
                "outputs_persisted": False,
                "promotion_eligible": report.promotion.eligible,
                "live_execution_gate": report.promotion.gates["live_execution_evidence"],
                "rollout_note": (
                    "plan_only evidence cannot promote; change execution to sync/async "
                    "only after cost and side-effect review"
                ),
            },
        )
        crupier.close()


if __name__ == "__main__":
    main()

import re

import pytest

from crupier.budgets import ExecutionBudget
from crupier.config import CrupierConfig
from crupier.constraints import validate_request_constraints
from crupier.errors import CrupierRouteValidationError
from crupier.learning import ScoringSuggestion, ScoringTuningReport, _score_values
from crupier.models import CapabilityCard, ModelRef, RequestEnvelope
from crupier.redaction import redact_value


def test_redact_value_handles_tuples_and_custom_patterns():
    value = ("ticket-123", "case-456")
    patterns = [(r"ticket-\d+", "ticket-[hidden]"), (re.compile(r"case-\d+"), "case-[hidden]")]

    assert redact_value(value, extra_patterns=patterns) == (
        "ticket-[hidden]",
        "case-[hidden]",
    )


def test_remaining_cost_is_bounded_at_zero_after_absorbing_snapshot(tmp_path):
    config = CrupierConfig.from_dict({"routing": {"max_cost_per_request_usd": 1.0}})
    config.root = tmp_path
    budget = ExecutionBudget(config, RequestEnvelope(task="test"), [])
    budget.absorb({"estimated_cost_reserved_usd": 1.5})

    assert budget.remaining_cost_usd() == 0.0


def test_non_finite_numeric_constraint_is_rejected():
    with pytest.raises(CrupierRouteValidationError, match="max_cost_usd.*finite"):
        validate_request_constraints({"max_cost_usd": float("inf")}, has_tools=False)


def test_scoring_report_updates_and_invalid_scores_are_explicit():
    suggestion = ScoringSuggestion(
        field="local_eval_weight",
        current=1.0,
        suggested=1.25,
        reason="test evidence",
    )
    report = ScoringTuningReport(applied=False, evidence={}, suggestions=[suggestion])
    card = CapabilityCard(
        model_ref=ModelRef.parse("openai:test"),
        last_updated="2026-08-23",
        local_eval_scores={"eval:valid": "2.5", "eval:invalid": object()},
    )

    assert report.updates == {"local_eval_weight": 1.25}
    assert _score_values([card], prefixes=("eval:",), include_plain=True) == [2.5]

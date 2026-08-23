import json
from types import SimpleNamespace

import pytest

from crupier import Crupier
from crupier.adapters import AdapterResponse
from crupier.config import CrupierConfig
from crupier.errors import (
    CrupierApprovalRequired,
    CrupierBudgetExceededError,
    CrupierError,
    CrupierExecutionLimitError,
)


class SessionAdapter:
    provider = "openai"

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "messages": list(request.messages),
            }
        )
        output = self.outputs.pop(0) if self.outputs else f"answer-{len(self.calls)}"
        return AdapterResponse(
            text=output,
            usage={"input_tokens": 2, "output_tokens": 2},
            metadata={"provider": "openai", "model": model},
        )


def make_client(tmp_path, *, adapter=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "sessions", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.4-mini"]},
            "routing": {"default_strategy": "single", "max_calls": 20},
            "profiles": {
                "agentic": {"prefer": ["tool_use"], "strategy": "single"},
                "structured": {"prefer": ["structured_output"], "strategy": "single"},
            },
        }
    )
    config.root = tmp_path
    return Crupier(config, adapters={"openai": adapter or SessionAdapter()})


def test_session_keeps_compatible_route_and_replans_on_capability_change(tmp_path):
    adapter = SessionAdapter()
    client = make_client(tmp_path, adapter=adapter)
    session = client.session(mode="agentic", sticky=True)

    first = session.deal("Summarize ticket", input={"id": "T-1"}, dry_run=False)
    second = session.deal("Draft the reply", dry_run=False, trace="summary")
    csv_path = tmp_path / "contract.csv"
    csv_path.write_text("clause,risk\nrenewal,high\n", encoding="utf-8")
    session.deal("Review the attached table", files=[csv_path], dry_run=False)

    assert first.output_text == "answer-1"
    assert second.trace is not None
    assert second.trace.final_quality_signals["sticky_route_reused"] is True
    assert session.route_history[1].reason == "compatible_route_retained"
    assert session.route_history[1].reused is True
    assert session.route_history[2].reason == "capability_changed"
    assert session.route_history[2].reused is False
    assert adapter.calls[1]["messages"][0]["content"]["task"] == "Summarize ticket"
    assert '"clause": "renewal"' in adapter.calls[2]["prompt"]


def test_session_keeps_compatible_route_in_dry_run(tmp_path):
    """Dry-run no puede pisar sticky_route_reused: las sesiones offline también retienen ruta."""

    client = make_client(tmp_path)
    session = client.session(mode="agentic", sticky=True)

    session.deal("Summarize ticket", input={"id": "T-1"}, dry_run=True)
    second = session.deal("Draft the reply", dry_run=True, trace="summary")
    csv_path = tmp_path / "contract.csv"
    csv_path.write_text("clause,risk\nrenewal,high\n", encoding="utf-8")
    session.deal("Review the attached table", files=[csv_path], dry_run=True)

    assert second.trace is not None
    assert second.trace.final_quality_signals["sticky_route_reused"] is True
    assert session.route_history[1].reason == "compatible_route_retained"
    assert session.route_history[1].reused is True
    assert session.route_history[2].reason == "capability_changed"
    assert session.route_history[2].reused is False


def test_persisted_session_resumes_and_detects_concurrent_writers(tmp_path):
    client = make_client(tmp_path)
    session = client.session(mode="agentic", persist=True)
    session.deal("First turn", dry_run=True)
    stale = client.sessions.resume(session.session_id)
    current = client.sessions.resume(session.session_id)

    current.deal("Second turn", dry_run=True)

    with pytest.raises(CrupierError, match="Concurrent update"):
        stale.deal("Conflicting second turn", dry_run=True)

    resumed = client.sessions.resume(session.session_id)
    assert resumed.turns == 2
    assert len(resumed.messages) == 4
    assert len(resumed.route_history) == 2
    assert client.sessions.list(status="active")[0].session_id == session.session_id
    assert resumed.close().status == "closed"
    with pytest.raises(CrupierError, match="closed"):
        resumed.deal("No more turns", dry_run=True)


def test_session_enforces_turn_and_cumulative_cost_limits(tmp_path):
    client = make_client(tmp_path)
    turn_limited = client.session(max_turns=1)
    turn_limited.deal("Only turn", dry_run=True)

    with pytest.raises(CrupierExecutionLimitError, match="max_turns"):
        turn_limited.deal("Too many", dry_run=True)

    cost_limited = client.session(max_session_cost_usd=1.0)
    cost_limited.cumulative_cost_usd = 1.0
    with pytest.raises(CrupierBudgetExceededError, match="max_session_cost_usd"):
        cost_limited.deal("No budget", dry_run=False)


def test_session_compacts_history_and_supports_a_custom_compactor(tmp_path):
    adapter = SessionAdapter(outputs=["x" * 900, "y" * 900])
    client = make_client(tmp_path, adapter=adapter)
    session = client.session(max_history_chars=1_000)

    session.deal("First", dry_run=False)
    session.deal("Second", dry_run=False)

    assert session.messages[0]["role"] == "system"
    assert "compacted" in session.messages[0]["content"]

    compacted = client.session(
        max_history_chars=1_000,
        compactor=lambda messages: [{"role": "system", "content": "custom summary"}],
    )
    compacted.deal("Large", input="z" * 2_000, dry_run=False)
    assert compacted.messages == [{"role": "system", "content": "custom summary"}]

    oversized = client.session(
        max_history_chars=1_000,
        compactor=lambda messages: [{"role": "system", "content": "x" * 2_000}],
    )
    with pytest.raises(CrupierError, match="compactor output exceeds"):
        oversized.deal("Large", input="z" * 2_000, dry_run=False)


def test_session_reuses_tool_idempotency_ledger_across_turns(tmp_path):
    tool_plan = json.dumps(
        {"tool_calls": [{"name": "lookup_ticket", "arguments": {"ticket_id": "T-1"}}]}
    )
    done = json.dumps({"tool_calls": [], "final": "done"})
    adapter = SessionAdapter(outputs=[tool_plan, done, tool_plan, done])
    client = make_client(tmp_path, adapter=adapter)
    calls = []

    def lookup_ticket(ticket_id: str):
        calls.append(ticket_id)
        return {"ticket_id": ticket_id, "status": "open"}

    session = client.session()
    session.deal("Check ticket", tools=[lookup_ticket], dry_run=False)
    second = session.deal("Check it again", tools=[lookup_ticket], dry_run=False)

    assert calls == ["T-1"]
    assert any(
        item["status"] == "skipped_duplicate"
        for item in second.provider_metadata["tool_calls"]
    )


def test_session_configuration_validation(tmp_path):
    client = make_client(tmp_path)

    with pytest.raises(CrupierError, match="stickiness"):
        client.session(stickiness="forever")
    with pytest.raises(CrupierError, match="max_turns"):
        client.session(max_turns=0)
    with pytest.raises(CrupierError, match="max_history_chars"):
        client.session(max_history_chars=100)


def test_session_replans_when_requested_strategy_changes_but_not_as_budget_is_spent(tmp_path):
    adapter = SessionAdapter()
    client = make_client(tmp_path, adapter=adapter)
    session = client.session(max_session_cost_usd=1.0)

    session.deal(
        "First",
        strategy="single",
        constraints={"max_cost_usd": 0.5},
        dry_run=False,
    )
    session.deal(
        "Second",
        strategy="single",
        constraints={"max_cost_usd": 0.5},
        dry_run=False,
    )
    session.deal(
        "Third",
        strategy="critique_repair",
        constraints={"max_cost_usd": 0.5},
        dry_run=True,
    )

    assert session.route_history[1].reason == "compatible_route_retained"
    assert session.route_history[1].reused is True
    assert session.route_history[2].reason == "strategy_changed"


def test_session_approval_is_bound_and_records_the_frozen_turn(tmp_path):
    adapter = SessionAdapter()
    client = make_client(tmp_path, adapter=adapter)
    session = client.session(persist=True)
    foreign_session = client.session()

    with pytest.raises(CrupierApprovalRequired) as pending:
        session.deal(
            "Apply the reviewed session change",
            constraints={"requires_human_approval": True},
            dry_run=False,
        )
    granted = client.approvals.grant(
        pending.value.approval_id,
        reviewer="ana",
    )
    assert granted.token is not None

    with pytest.raises(CrupierError, match="expected 'session_id' context"):
        foreign_session.execute_approved(granted.token)
    assert client.approvals.get(pending.value.approval_id).status == "granted"

    result = session.deal(
        "This wrapper text must not replace the frozen task",
        approval_token=granted.token,
        trace="summary",
    )

    assert result.output_text == "answer-1"
    assert result.trace is not None
    assert session.turns == 1
    assert session.messages[0]["content"]["task"] == "Apply the reviewed session change"
    assert session.route_history[0].reason == "initial_route"
    assert client.approvals.get(pending.value.approval_id).status == "consumed"


def test_session_close_is_idempotent_and_approved_turn_limit_is_enforced(tmp_path):
    client = make_client(tmp_path)
    session = client.session(max_turns=1, persist=True)
    session.turns = 1

    with pytest.raises(CrupierExecutionLimitError, match="max_turns"):
        session.execute_approved("approval-token")

    assert session.close().status == "closed"
    assert session.close().status == "closed"


def test_session_approved_execution_can_hide_trace(tmp_path):
    client = make_client(tmp_path)
    session = client.session()
    with pytest.raises(CrupierApprovalRequired) as pending:
        session.deal(
            "Approve without returning a trace",
            constraints={"requires_human_approval": True},
            dry_run=False,
        )
    granted = client.approvals.grant(pending.value.approval_id, reviewer="ana")

    result = session.execute_approved(granted.token or "", trace=False)

    assert result.trace is None
    assert session.turns == 1


def test_session_rejects_results_without_a_route(tmp_path):
    session = make_client(tmp_path).session()
    result = SimpleNamespace(route=None, trace=None)

    with pytest.raises(CrupierError, match="without a route"):
        session._record_turn(
            task="missing route",
            input=None,
            result=result,
            signature={},
            reason="initial_route",
            dry_run=True,
        )


def test_session_replan_reasons_cover_residual_changes_and_context_pressure(tmp_path, monkeypatch):
    session = make_client(tmp_path).session(stickiness="none")
    session.deal("First", dry_run=True)
    baseline = dict(session._previous_signature or {})

    assert session._replan_reason(baseline) == "stickiness_disabled"
    session.stickiness = "compatible"
    for key, value, expected in (
        ("mode", "structured", "mode_changed"),
        ("risk_level", "high", "risk_changed"),
        ("budget", {"max_cost_usd": 0.01}, "budget_changed"),
    ):
        changed = dict(baseline)
        changed[key] = value
        assert session._replan_reason(changed) == expected

    monkeypatch.setattr(session, "_context_pressure", lambda: True)
    assert session._replan_reason(baseline) == "context_pressure"


def test_session_context_pressure_handles_missing_or_unknown_windows(tmp_path, monkeypatch):
    session = make_client(tmp_path).session()
    assert session._context_pressure() is False
    session.deal("First", dry_run=True)

    monkeypatch.setattr(
        session.client.registry,
        "get",
        lambda model: (_ for _ in ()).throw(CrupierError(model)),
    )
    assert session._context_pressure() is False
    monkeypatch.setattr(
        session.client.registry,
        "get",
        lambda model: SimpleNamespace(context_window=None),
    )
    assert session._context_pressure() is False


def test_session_rejects_non_list_compactor_output(tmp_path):
    session = make_client(tmp_path).session(
        max_history_chars=1_000,
        compactor=lambda messages: {"messages": messages},
    )

    with pytest.raises(CrupierError, match="must return a list"):
        session.deal("Large", input="z" * 2_000, dry_run=False)

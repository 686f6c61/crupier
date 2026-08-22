from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from crupier.client import Crupier, ResponsesFacade
from crupier.config import CrupierConfig, write_default_project
from crupier.errors import (
    CrupierConfigError,
    CrupierError,
    CrupierPolicyError,
    CrupierProviderUnavailableError,
)
from crupier.models import (
    CapabilityCard,
    CrupierResult,
    DecisionTrace,
    FileRepresentation,
    FileRoutingPlan,
    ModelRef,
    PreparedDeal,
    RequestEnvelope,
    RoutePlan,
    RouteStep,
)
from crupier.policy import PolicyResult


def _config(tmp_path, *, providers: dict | None = None) -> CrupierConfig:
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "client-edges", "default_profile": "agentic"},
            "providers": providers or {},
            "orchestrator": {"mode": "deterministic"},
            "profiles": {"agentic": {"prefer": [], "strategy": "single"}},
        }
    )
    config.root = tmp_path
    return config


def _card(model: str) -> CapabilityCard:
    return CapabilityCard(ModelRef.parse(model), "2026-08-22")


def _prepared(*, requires_approval: bool = False, dry_run: bool = False) -> PreparedDeal:
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:a")],
        requires_user_confirmation=requires_approval,
    )
    return PreparedDeal(
        request=RequestEnvelope(task="x"),
        plan=plan,
        trace=DecisionTrace(trace_id="trace", request_summary="x"),
        dry_run=dry_run,
    )


def test_model_manager_handles_missing_and_unavailable_discovery_adapters(tmp_path):
    class Unavailable:
        def list_models(self):
            raise CrupierProviderUnavailableError("offline")

    client = Crupier(_config(tmp_path), adapters={"openai": Unavailable()})
    assert client.models.discover(provider="missing") == []

    with pytest.raises(CrupierProviderUnavailableError, match="offline"):
        client.models.discover(provider="openai")
    warnings: list[str] = []
    assert client.models.discover(provider="openai", skip_unavailable=True, warnings=warnings) == []
    assert "offline" in warnings[0]


def test_model_manager_allow_reloads_registry_configuration(tmp_path):
    write_default_project(tmp_path)
    client = Crupier.from_project(tmp_path)

    client.models.allow(["openai:gpt-5.5"], replace=True)

    assert client.models._config.models.allow == ["openai:gpt-5.5"]
    assert client.registry.config is client.models._config
    assert client.registry._cards is None


def test_response_facade_context_and_constructor_helpers(tmp_path, monkeypatch):
    calls = []
    facade = ResponsesFacade(
        SimpleNamespace(deal=lambda **kwargs: calls.append(kwargs) or CrupierResult(output_text="ok"))
    )
    assert facade.create(input="hello", mode="fast", task="answer", dry_run=True).output_text == "ok"
    assert calls == [{"task": "answer", "input": "hello", "mode": "fast", "dry_run": True}]

    config = _config(tmp_path)
    assert Crupier.from_config(config).config is config
    assert Crupier.from_config({}).config.project.name == "crupier-project"

    write_default_project(tmp_path / "toml")
    assert Crupier.from_toml(tmp_path / "toml").config.root == (tmp_path / "toml").resolve()

    client = Crupier(config)
    closed = []
    monkeypatch.setattr(client.experiments, "close", lambda *, wait=True: closed.append(wait))
    assert client.__enter__() is client
    client.__exit__(None, None, None)
    client.close(wait=False)
    assert closed == [True, False]


def test_from_project_precedence_and_missing_configuration(tmp_path, monkeypatch):
    current = tmp_path / "current"
    environment = tmp_path / "environment"
    write_default_project(current)
    write_default_project(environment)
    monkeypatch.chdir(current)
    monkeypatch.setenv("CRUPIER_PROJECT", str(environment))

    assert Crupier.from_project().config.root == current.resolve()

    monkeypatch.chdir(tmp_path)
    assert Crupier.from_project().config.root == environment.resolve()

    monkeypatch.delenv("CRUPIER_PROJECT")
    with pytest.raises(CrupierConfigError, match="No crupier.toml"):
        Crupier.from_project()


def test_update_orchestrator_validates_and_updates_every_option(tmp_path):
    client = Crupier(_config(tmp_path))
    with pytest.raises(CrupierConfigError, match="mode must be"):
        client.update_orchestrator(mode="invalid")

    client.update_orchestrator(
        mode="hybrid",
        model="openai:a",
        fallback_model="openai:b",
        fallback="deterministic",
        temperature=0.25,
        require_validated_plan=False,
        max_repairs=2,
        candidate_limit=4,
        allow_prompt_summary_only=True,
    )

    settings = client.config.orchestrator
    assert settings.fallback_model == "openai:b"
    assert settings.fallback == "deterministic"
    assert settings.temperature == 0.25
    assert settings.require_validated_plan is False


def test_deal_delegates_approval_token(tmp_path, monkeypatch):
    client = Crupier(_config(tmp_path))
    expected = CrupierResult(output_text="approved")
    monkeypatch.setattr(
        client,
        "execute_approved",
        lambda token, *, tools, trace: expected if (token, tools, trace) == ("token", ["tool"], "full") else None,
    )

    assert client.deal("ignored", tools=["tool"], trace="full", approval_token="token") is expected


def test_prepare_warns_about_fake_approval_and_invalid_sticky_plan(tmp_path, monkeypatch):
    client = Crupier(_config(tmp_path), adapters={})
    card = _card("openai:a")
    plan = RoutePlan(strategy="single", steps=[RouteStep(role="primary", model="openai:a")])
    monkeypatch.setattr(client.registry, "allowed_cards", lambda: [card])
    monkeypatch.setattr(client.policy, "filter_candidates", lambda request, cards: PolicyResult(allowed=[card]))
    monkeypatch.setattr(client.planner, "plan", lambda request, cards, filters: plan)
    validations = 0

    def validate_route(route, policy_result, request):
        nonlocal validations
        validations += 1
        if validations == 1:
            raise CrupierError("stale route")

    monkeypatch.setattr(client.policy, "validate_route", validate_route)

    prepared = client.prepare(
        "x",
        constraints={"human_approval_granted": True},
        metadata={"_crupier_sticky_plan": plan, "_crupier_orchestrator_calls": "invalid"},
        dry_run=True,
    )

    assert any("not authorization" in warning for warning in prepared.warnings)
    assert any("invalidated" in warning for warning in prepared.warnings)
    assert prepared.planning_calls == []


def test_approval_guards_reject_unapproved_prepared_deals(tmp_path):
    client = Crupier(_config(tmp_path))
    with pytest.raises(CrupierPolicyError, match="requires_user_confirmation"):
        client.request_approval(_prepared())
    with pytest.raises(CrupierPolicyError, match="granted approval token"):
        client.execute(_prepared(requires_approval=True))


def test_adapter_and_native_file_filters_report_total_exclusion(tmp_path):
    client = Crupier(_config(tmp_path), adapters={})
    card = _card("openai:a")
    with pytest.raises(CrupierPolicyError, match="adapter availability"):
        client._filter_adapter_candidates([card])

    class RejectNative:
        def supports_file_kind(self, *, model, kind):
            return False

    client.adapters["openai"] = RejectNative()
    request = RequestEnvelope(
        task="x",
        file_plan=FileRoutingPlan(
            representations=[FileRepresentation("report.pdf", "pdf", "native_pdf")]
        ),
    )
    with pytest.raises(CrupierPolicyError, match="file-transport"):
        client._filter_adapter_file_candidates(request, [card])


def test_provider_visibility_reports_missing_and_unexpected_adapter_failures(tmp_path):
    client = Crupier(_config(tmp_path), adapters={})
    missing = client._provider_visible_models("openai")
    assert missing[0] is None and "no configured adapter" in missing[1]

    class BrokenDiscovery:
        def list_models(self):
            raise RuntimeError("broken sdk")

    client.adapters["custom"] = BrokenDiscovery()
    broken = client._provider_visible_models("custom")
    assert broken[0] is None and "broken sdk" in broken[1]


def test_async_stream_and_apply_update_facades(tmp_path, monkeypatch):
    client = Crupier(_config(tmp_path))
    route = RoutePlan(strategy="single", steps=[RouteStep(role="primary", model="openai:a")])
    result = CrupierResult(output_text="ok", route=route)
    monkeypatch.setattr(client, "deal", lambda *args, **kwargs: result)

    assert asyncio.run(client.adeal("x")) is result
    events = list(client.stream("x"))
    assert [event.type for event in events] == ["route_started", "route_selected", "final"]
    assert events[1].route is route and events[2].result is result

    dry_runs = []
    monkeypatch.setattr(client.registry, "update", lambda *, dry_run: dry_runs.append(dry_run) or "report")
    assert client.update(apply=True) == "report"
    assert dry_runs == [False]

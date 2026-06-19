from crupier.config import CrupierConfig
from crupier.models import CapabilityCard, ModelRef, RequestEnvelope
from crupier.selector import ModelSelector


def test_selector_explains_scores_for_profile_and_task_signals():
    config = CrupierConfig.from_dict(
        {
            "project": {"default_profile": "agentic"},
            "profiles": {"agentic": {"prefer": ["tool_use", "coding"], "strategy": "single"}},
        }
    )
    selector = ModelSelector(config)
    card = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        supports_tools=True,
        strengths=["tool_use", "coding"],
        quality_tier="frontier",
    )

    score = selector.score(RequestEnvelope(task="Refactor repo code", mode="agentic", tools=[object()]), card)

    term_names = {term.name for term in score.terms}
    assert "quality_tier" in term_names
    assert "profile_preferences" in term_names
    assert "task_signals" in term_names
    assert "tool_support" in term_names
    assert score.score > 0


def test_selector_uses_local_eval_score():
    config = CrupierConfig.from_dict({"project": {"default_profile": "agentic"}})
    selector = ModelSelector(config)
    card = CapabilityCard(
        model_ref=ModelRef.parse("anthropic:claude-opus-4-8"),
        last_updated="test",
        local_eval_scores={"agentic": 9.5},
    )

    score = selector.score(RequestEnvelope(task="Plan agent", mode="agentic"), card)

    assert any(term.name == "local_eval" and term.value == 9.5 for term in score.terms)


def test_selector_uses_eval_namespace_score():
    config = CrupierConfig.from_dict({"project": {"default_profile": "agentic"}})
    selector = ModelSelector(config)
    card = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.4-mini"),
        last_updated="test",
        local_eval_scores={"eval:agentic": 4.25},
    )

    score = selector.score(RequestEnvelope(task="Plan agent", mode="agentic"), card)

    assert any(term.name == "local_eval" and term.value == 4.25 for term in score.terms)


def test_selector_scores_verified_capability_above_inferred():
    config = CrupierConfig.from_dict({"project": {"default_profile": "agentic"}})
    selector = ModelSelector(config)
    inferred = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.4-mini"),
        last_updated="test",
        supports_tools=True,
    )
    verified = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        supports_tools=True,
        capability_status={"tool_call": {"status": "verified", "source": "probe:tool_call"}},
    )

    request = RequestEnvelope(task="Use a tool", mode="agentic", tools=[object()])
    inferred_score = selector.score(request, inferred)
    verified_score = selector.score(request, verified)

    inferred_term = next(term for term in inferred_score.terms if term.name == "tool_support")
    verified_term = next(term for term in verified_score.terms if term.name == "tool_support")
    assert inferred_term.value == 2
    assert verified_term.value == 6
    assert verified_score.score > inferred_score.score

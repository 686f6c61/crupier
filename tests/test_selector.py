from crupier.config import CrupierConfig
from crupier.model_profiles import classify_task_signal_weights
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


def test_selector_uses_project_scoring_weights():
    config = CrupierConfig.from_dict(
        {
            "project": {"default_profile": "agentic"},
            "scoring": {
                "quality_weight": {"frontier": 0, "strong": 0, "unknown": 0},
                "cost_weight": {"low": 20, "high": 0, "medium": 0, "unknown": 0},
                "cheap_mode_cost_multiplier": 1,
            },
        }
    )
    selector = ModelSelector(config)
    cheap = CapabilityCard(
        model_ref=ModelRef.parse("google:gemini-3.5-flash"),
        last_updated="test",
        cost_tier="low",
        quality_tier="strong",
    )
    frontier = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        cost_tier="high",
        quality_tier="frontier",
    )

    ranked = selector.rank(RequestEnvelope(task="Clasifica barato", mode="cheap"), [frontier, cheap])
    cheap_score = selector.score(RequestEnvelope(task="Clasifica barato", mode="cheap"), cheap)

    assert ranked[0].model_ref.key == "google:gemini-3.5-flash"
    assert any(term.name == "cheap_mode_cost" and term.value == 20 for term in cheap_score.terms)


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


def test_selector_uses_decision_skill_scores_for_spanish_task():
    config = CrupierConfig.from_dict({"project": {"default_profile": "agentic"}})
    selector = ModelSelector(config)
    generic = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        quality_tier="strong",
        skill_scores={"reasoning": 7.5},
    )
    coder = CapabilityCard(
        model_ref=ModelRef.parse("ollama:qwen3-coder:480b"),
        last_updated="test",
        skill_scores={"coding": 9.0, "agentic": 8.4, "tool_use": 7.5},
    )

    ranked = selector.rank(
        RequestEnvelope(task="Arreglar un bug del repositorio desde la consola", mode="agentic"),
        [generic, coder],
    )
    score = selector.score(
        RequestEnvelope(task="Arreglar un bug del repositorio desde la consola", mode="agentic"),
        coder,
    )

    assert ranked[0].model_ref.key == "ollama:qwen3-coder:480b"
    assert any(term.name == "skill_fit" for term in score.terms)


def test_task_signal_classifier_returns_weighted_signals_without_substring_false_positive():
    weights = classify_task_signal_weights(
        RequestEnvelope(task="Contest results are ready", mode="fast")
    )

    assert "coding" not in weights
    assert weights["fast"] == 0.7


def test_selector_weights_stronger_task_signals_more_than_weak_mode_hint():
    config = CrupierConfig.from_dict({"project": {"default_profile": "agentic"}})
    selector = ModelSelector(config)
    card = CapabilityCard(
        model_ref=ModelRef.parse("openai:gpt-5.5"),
        last_updated="test",
        strengths=["research", "fast"],
    )

    score = selector.score(
        RequestEnvelope(task="Research sources, cite evidence, and compare results", mode="fast"),
        card,
    )
    term = next(item for item in score.terms if item.name == "task_signals")

    assert term.value > config.scoring.task_signal_weight
    assert "research=1" in term.reason
    assert "fast=0.7" in term.reason

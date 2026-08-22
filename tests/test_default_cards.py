from crupier.default_cards import BUILTIN_CAPABILITY_CARDS
from crupier.models import CapabilityCard

SUPPORTED_PROVIDERS = {
    "anthropic",
    "google",
    "inference",
    "nan",
    "ollama",
    "openai",
    "openrouter",
}
ADVERTISED_MODEL_KINDS = {
    "chat",
    "embedding",
    "image_generation",
    "reranker",
    "transcription",
    "tts",
}


def test_builtin_capability_cards_hold_contract_invariants():
    parsed_cards = [CapabilityCard.from_dict(data) for data in BUILTIN_CAPABILITY_CARDS]
    keys = [card.model_ref.key for card in parsed_cards]

    assert len(keys) == len(set(keys))
    assert {card.model_ref.provider for card in parsed_cards} <= SUPPORTED_PROVIDERS
    assert all(isinstance(card, CapabilityCard) for card in parsed_cards)
    assert all(
        "model_kind" in data
        for data, card in zip(BUILTIN_CAPABILITY_CARDS, parsed_cards, strict=True)
        if card.model_kind != "chat"
    )


def test_builtin_cards_cover_all_model_kinds_advertised():
    actual = {data.get("model_kind", "chat") for data in BUILTIN_CAPABILITY_CARDS}

    assert actual == ADVERTISED_MODEL_KINDS

from crupier import ModelRef


def test_model_ref_parse_keeps_ollama_tag_colons():
    ref = ModelRef.parse("ollama:qwen3.5:122b")

    assert ref.provider == "ollama"
    assert ref.model == "qwen3.5:122b"
    assert ref.key == "ollama:qwen3.5:122b"


def test_model_ref_detects_latest_alias():
    ref = ModelRef.parse("openai:gpt-latest")

    assert ref.stability == "latest"


def test_model_ref_normalizes_claude_provider_alias():
    ref = ModelRef.parse("claude:claude-opus-4-8")

    assert ref.provider == "anthropic"
    assert ref.key == "anthropic:claude-opus-4-8"

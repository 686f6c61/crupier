# Pull Request

## Summary

- 

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] README/examples/package metadata
- [ ] Release/readiness
- [ ] Refactor or maintenance

## Validation

- [ ] `python -m pytest`
- [ ] `python -m pytest --cov=crupier --cov-fail-under=95`
- [ ] `python -m ruff check src tests`
- [ ] `python -m mypy src/crupier`
- [ ] `crupier release check`
- [ ] `crupier release check --strict-public --verify-project-urls --check-pypi-name` for release-facing changes or first public uploads
- [ ] Real-provider checks documented when behavior touches OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, configurable inference servers, or OpenRouter adapters

## Safety

- [ ] No API keys, prompts, private provider outputs, `.env`, `.crupier/`, or generated traces are committed
- [ ] New examples run without secrets unless explicitly marked as real-provider checks
- [ ] Public README and package metadata avoid placeholder URLs

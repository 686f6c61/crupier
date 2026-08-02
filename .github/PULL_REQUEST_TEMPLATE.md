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

- [ ] `python -m pytest -q`
- [ ] `python -m pytest -q --cov=crupier --cov-report=term --cov-fail-under=95`
- [ ] `python -m ruff check src tests examples`
- [ ] `python -m mypy src/crupier`
- [ ] `python -m pip_audit --skip-editable --progress-spinner off`
- [ ] `crupier release check --strict-public --verify-project-urls --check-pypi-name --allow-existing-pypi-project` for release-facing changes
- [ ] Real-provider checks documented when behavior touches OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, configurable inference servers, or OpenRouter adapters

## Safety

- [ ] No API keys, prompts, private provider outputs, `.env`, `.crupier/`, or generated traces are committed
- [ ] New examples run without secrets unless explicitly marked as real-provider checks
- [ ] Public README and package metadata avoid placeholder URLs

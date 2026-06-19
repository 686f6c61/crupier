# Pull Request

## Summary

- 

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] Release/readiness
- [ ] Refactor or maintenance

## Validation

- [ ] `python -m pytest`
- [ ] `crupier release check`
- [ ] `crupier release check --strict-public` for release-facing changes
- [ ] `crupier release check --check-pypi-name` before a first public upload
- [ ] Real-provider checks documented when behavior touches OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, or OpenRouter adapters

## Safety

- [ ] No API keys, prompts, private provider outputs, `.env`, `.crupier/`, or generated traces are committed
- [ ] New examples run without secrets unless explicitly marked as real-provider checks
- [ ] Public docs avoid placeholder URLs for release metadata


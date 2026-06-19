# Contributing to Crupier

Thanks for helping make Crupier reliable for real AI projects. Keep changes small,
reviewable, and backed by tests or documented manual verification.

## Development Setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Run the offline SDK example before changing public onboarding behavior:

```bash
python examples/sdk_dry_run.py
```

## Provider Keys

Never commit provider keys, prompts, responses, customer data, traces, or local
`.crupier/` artifacts. Use environment variables or a local `.env` file ignored
by git:

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OLLAMA_API_KEY=
OLLAMA_HOST=https://ollama.com/api
```

Use real provider checks only when keys are intentionally loaded:

```bash
set -a
source .env
set +a
crupier release check --verify-providers --provider openai --provider anthropic --provider ollama
```

## Pull Request Checklist

Before opening a PR, run:

```bash
python -m pytest
python -m ruff check src tests --select E9,F63,F7,F82
python -m pip_audit --skip-editable --progress-spinner off
crupier release check
```

For release-facing changes, also run:

```bash
crupier release check --strict-public
crupier release check --check-pypi-name
```

`--strict-public` is expected to fail until real public `[project.urls]` are set.
Use `--check-pypi-name` before the first upload so package-name availability is checked against PyPI.
Do not bypass that warning for a public PyPI upload.

## Documentation

Update `README.md`, `CHANGELOG.md`, and the relevant document under `docs/`
whenever behavior changes. Public examples must be executable without secrets
unless they are explicitly marked as real-provider checks.

## Release Discipline

Crupier public releases use final numeric versions such as `0.1.0`. Do not
publish non-final, development, or local build versions unless the release
policy is intentionally changed.

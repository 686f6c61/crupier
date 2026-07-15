# Contributing to Crupier

Thanks for helping make Crupier reliable for real AI projects. Keep changes small,
reviewable, and backed by tests or documented manual verification.

## Development Setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=83" wheel
python -m pip install -e ".[all,dev]"
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
GEMINI_API_KEY=
INFERENCE_API_KEY=
OLLAMA_API_KEY=
OLLAMA_HOST=https://ollama.com/api
```

Use real provider checks only when keys are intentionally loaded:

```bash
crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama --provider inference
```

For changes to routing, role execution, tools, or multimodal behavior, run the public end-to-end matrix as well:

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u GEMINI_API_KEY \
  -u OLLAMA_API_KEY -u OLLAMA_HOST \
  python examples/live_routing_validation.py --real --project . --write-report
```

The command must finish with every selected case passing. Keep the generated `.crupier/evals/` report local; summarize only sanitized routing evidence in public changes.

For changes to operation classification, embeddings, reranking, audio, images, Python compatibility, or the HTTP server, run the operations matrix against an ignored local project containing executable models for every required model kind:

```bash
python examples/live_operations_validation.py \
  --real --project . --write-report
```

The command must finish with `7/7` cases passing. Its report is local evidence only and must never be committed.

`--env-file` loads missing variables only; exported shell or CI variables keep precedence.

## Pull Request Checklist

Before opening a PR, run:

```bash
python -m pytest
python -m pytest --cov=crupier --cov-fail-under=95
python -m ruff check src tests
python -m mypy src/crupier
python -m pip_audit --skip-editable --progress-spinner off
crupier release check
```

For release-facing changes, also run:

```bash
crupier release check --strict-public
crupier release check --strict-public --verify-project-urls --check-pypi-name
```

`--strict-public` fails if real public `[project.urls]` are missing.
Use the combined `--verify-project-urls --check-pypi-name` gate before publishing so package metadata
links are reachable and first-upload package-name availability is checked against PyPI.
Do not bypass those checks for a public PyPI upload.

## Public Onboarding

Update `README.md`, `CHANGELOG.md`, package metadata, and examples whenever behavior changes.
Public examples must be executable without secrets unless they are explicitly marked as real-provider checks.

## Public Repository Settings

Before changing repository visibility to public, keep the public surface focused:

- Issues enabled; wiki and projects disabled unless there is an active maintainer workflow for them.
- GitHub topics set for AI, agents, LLM routing, orchestration, and Python discoverability.
- Pull requests merged with squash commits and head branches deleted after merge.
- Dependabot security updates enabled and unpaused for dependency vulnerability remediation.
- Protect `main` after the final single release commit is accepted: require the CI workflow, require pull-request review for public changes, and disallow force pushes.
- Private vulnerability reporting enabled once GitHub exposes it for the public repository.
- Secret scanning and push protection enabled once available for the repository visibility/account.

After the visibility change, rerun:

```bash
crupier release check --strict-public --verify-project-urls --check-pypi-name
crupier release check --strict-public --verify-providers --provider openai --provider anthropic --provider google --provider ollama --provider inference
```

Publish `0.4.0` from a GitHub Release tagged `v0.4.0` or `0.4.0` only after
PyPI trusted publishing is configured for this repository and the `pypi`
environment. The publish workflow checks the release tag against the package
version before building or uploading distributions. Manual workflow dispatch
must provide `version=0.4.0` and `confirm_publish=true`; use it only to retry an
intentional release operation.
The workflow accepts the existing PyPI project for maintenance releases after
the first upload has established ownership.

## Release Discipline

Crupier public releases use final numeric versions such as `0.4.0`. Do not
publish non-final, development, or local build versions unless the release
policy is intentionally changed.

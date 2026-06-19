# Crupier Publishing

This is the public release checklist for `crupier==0.1.0`.

## Local Gate

Run from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests --select E9,F63,F7,F82
python -m pip_audit --skip-editable --progress-spinner off
crupier release check
crupier release check --strict-public
crupier release check --check-pypi-name
```

`crupier release check` validates public metadata, final version shape, non-final release language, public collaboration templates, vulnerability reporting and secret-handling policy, CI/dependency-update maintenance files, critical lint wiring, dependency vulnerability audit wiring, safe onboarding defaults, builds both `sdist` and wheel, runs `twine check`, installs both distributions in clean temporary virtual environments, imports `crupier`, verifies the `crupier` CLI, runs `crupier init`, confirms a fresh project can produce a dry-run `crupier route` decision, and executes the Python SDK quickstart with `Crupier.from_project().deal(..., dry_run=True)` without provider SDKs.

The normal gate warns, rather than fails, when `[project.urls]` is missing. The strict public gate fails while any warning remains or build checks are skipped. `--check-pypi-name` performs an explicit PyPI name lookup; for a first upload it fails if the project name is already claimed. Add real public URLs before the first PyPI upload; do not use placeholder repository links.

For the final local publish gate with real provider keys loaded:

```bash
set -a
source .env
set +a
crupier release check --strict-public --verify-providers --provider openai --provider anthropic --provider ollama
crupier release check --strict-public --check-pypi-name --verify-providers --provider openai --provider anthropic --provider ollama
```

`--verify-providers` appends a blocking provider readiness check. It runs provider config/env checks, model discovery, capability readiness, and real smoke calls for the selected providers. Omit it in public CI unless the workflow has deliberately configured provider secrets.

After the project is already published and controlled by the same maintainers, use `--allow-existing-pypi-project` together with `--check-pypi-name` for maintenance releases.

## Version Policy

Public PyPI releases use final numeric versions such as `0.1.0`. Do not publish non-final, development, or local builds unless the project explicitly changes this policy.

The release gate enforces the `X.Y.Z` shape and fails if the version is changed to a non-final suffix.

## Manual Upload

Use this when publishing directly with a PyPI API token:

```bash
rm -rf dist build src/crupier.egg-info
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
python -m twine upload dist/*
```

For a TestPyPI rehearsal:

```bash
rm -rf dist build src/crupier.egg-info
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

## Trusted Publishing

The repository includes `.github/workflows/publish.yml`. Configure the PyPI project trusted publisher to trust:

- repository: the real GitHub repository for Crupier
- workflow: `.github/workflows/publish.yml`
- environment: `pypi`

Then publish a GitHub Release to run tests, `crupier release check --strict-public`, build distributions, run `twine check`, upload the checked `dist/*` files as a GitHub Actions artifact, and publish to PyPI without storing a PyPI token in the repository.

Before the first public upload, set the real repository URL in `[project.urls]` once the public repository exists. Do not invent placeholder URLs just to fill PyPI metadata.

## Release Notes

Before publishing a future version, keep `CHANGELOG.md` dated for that release. Do not commit `.env`, provider keys, build artifacts, generated `.crupier` review/audit packages, or `.crupier` traces.

# Changelog

All notable changes to Crupier will be documented here.

## 0.1.0 - 2026-06-19

- Added multi-provider routing core with dry-run and real execution paths.
- Added OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud adapters with explicit local-host override support.
- Added model discovery, capability probes, readiness checks, and registry snapshots.
- Added structured output validation/repair, tool execution, multimodal file planning, and local trace storage.
- Added OpenAI-compatible client, optional autopatch, and HTTP proxy surfaces.
- Added project adoption audit, adoption doctor, production readiness gates, code comments, compare evals, and human feedback signals.
- Added adoption `review_contract` summaries that separate technical readiness from human approval and block auto-approval while human gates remain open.
- Added config-free `adopt doctor`, `adopt plan`, `adopt patches`, and offline `adopt handoff` so freshly cloned repos can get non-destructive adoption guidance before `crupier init`.
- Added `.env.example` creation and safe `.gitignore` entries to `crupier init` for new project onboarding.
- Added a release gate for safe public onboarding defaults, including Ollama Cloud as the default Ollama host and opt-in prompt/response storage.
- Added one-command `adopt package` to write the full human review bundle and persistent package index for an existing project.
- Tightened programmer credential comments to reduce false positives from redaction regexes, short identifiers, generated output, and dependency trees.
- Added human review packets for compare reports with ready-to-run feedback commands.
- Added editable human decision templates and feedback import for reviewer-filled verdicts.
- Added dry-run feedback source guards so simulated compare reports cannot satisfy production human-feedback gates by accident.
- Added programmer code-comment review acknowledgements for adoption doctor gates.
- Added PR/review-comment packets for programmer code comments.
- Added SARIF export for programmer code comments and adoption packages.
- Added editable programmer code-comment decision templates and import so partial human review keeps unresolved comments pending.
- Classified credential-like test fixtures as P3 reviewer notes instead of P1 production credential blockers.
- Added an executable offline SDK dry-run example for first-run onboarding without provider SDKs or API keys.
- Added adoption handoff reports for reviewer-facing rollout actions, decision templates, and artifacts.
- Added project-level adoption signoff records and production gates for human rollout approval/rejection.
- Added production doctor checks for human feedback applied to capability cards.
- Added release readiness checks and CI workflow.
- Added CONTRIBUTING.md with local development, provider-key handling, PR, documentation, and release-gate expectations.
- Added MIT license metadata for package release readiness.
- Tightened public PyPI metadata checks for keywords and classifiers without using legacy license classifiers rejected by modern setuptools.
- Added a non-blocking project URL readiness warning so public PyPI uploads do not ship with placeholder or missing repository links by accident.
- Added sdist + wheel build, artifact content inspection, `twine check`, py.typed packaging, and wheel/sdist install/import/CLI/`crupier init`/dry-run route/Python SDK quickstart smoke validation to release checks.
- Added `crupier release check --strict-public` so PyPI publishing blocks on warnings such as missing public project URLs and on skipped build/install smoke checks.
- Added opt-in `crupier release check --verify-providers` for final public release gates with real provider discovery, readiness, and smoke validation.
- Added release artifact inspection for the offline SDK example so the sdist keeps first-run onboarding examples.
- Added release checks and artifact inspection for CONTRIBUTING.md so public development guidance is shipped and maintained.
- Added a PyPI publish workflow for GitHub Release based publishing.
- Added a release-language guard so final `0.1.0` package metadata and publishing docs do not regress into non-final release labels.
- Added optional `crupier release check --check-pypi-name` with `--allow-existing-pypi-project` for first-upload and maintenance-release PyPI name preflights.
- Added public collaboration templates and release checks for Code of Conduct, bug reports, feature requests, and pull requests.
- Added Dependabot configuration for Python tooling and GitHub Actions plus release checks for dependency-update automation and minimal CI permissions.
- Updated CI/publish workflows to current GitHub Actions majors and added checked distribution artifact upload before PyPI publishing.
- Expanded SECURITY.md with scope, private reporting, supported versions, disclosure process, and secret-handling guidance, and added release checks for that content.
- Added `pip-audit` to the development workflow plus CI and publish dependency vulnerability audit steps.
- Added Ruff critical lint (`E9,F63,F7,F82`) to dev, CI, publish, and release-readiness checks.

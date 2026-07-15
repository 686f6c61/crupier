# Changelog

All notable changes to Crupier will be documented here.

## 0.4.0 - 2026-07-15

- Added real configurable inference-server coverage with live model discovery, chat/structured/streaming probes, native multimodal input, embeddings, model-kind classification, and curated routing cards without invented benchmarks or pricing.
- Added capability-aware execution for provider-specific reranking, transcription, text-to-speech, image generation, and multi-reference image editing, with operation-specific validation and probes.
- Added `Crupier.run()` model-powered operation classification plus explicit `embed`, `rerank`, `transcribe`, `synthesize`, `generate_image`, and `edit_image` methods under a shared end-to-end budget and trace.
- Made newly initialized projects model-powered by default, while retaining deterministic scoring, policy validation, repair limits, and deterministic fallback; programmatic configs can still opt into deterministic-only routing.
- Extended the OpenAI-like Python and HTTP surfaces with provider-neutral embeddings, reranking, images, speech, transcription, multipart uploads, binary responses, and bounded request bodies.
- Made request budgets cover model-orchestrator planning as well as execution, retries, tools, parallel routes, and delegation; explicit `force_model` and single-candidate routes now bypass redundant LLM planning.
- Corrected cost semantics so token-derived values remain estimates and `actual_usd` is only populated from provider-reported billed cost.
- Added request-aware reasoning controls for Qwen, Gemma, DeepSeek, and Mimo families, with explicit caller settings taking precedence.
- Added adapter-level native-file transport checks so model capability metadata cannot cause an adapter to silently omit images, audio, PDFs, or other files.
- Added OpenAI native-PDF transport, provider-specific native-audio routing, automatic native audio while transcript preprocessing is unavailable, bounded file loading, and mixed extracted/native file execution.
- Tightened runtime config validation, provider visibility caching, server request-size limits, embedding policy, retry/circuit-breaker budgets, tool result bounds, and multimodal data-URL handling.
- Made capability probes model-kind aware and strengthened structured, streaming, tool, embedding, reranking, transcription, speech, and image probe evidence from live configurable-server and Ollama Cloud checks.
- Expanded model-orchestrator context with bounded request content, natural capability summaries, context/output limits, reasoning hints, edge cases, and pricing evidence; prompt-summary-only mode is now opt-in.
- Added full `mypy` cleanliness for the typed package, full Ruff CI, a 95% coverage gate, package build checks, and dependency vulnerability auditing.
- Hardened CI and publishing environments with patched build tooling and complete optional provider dependencies so dependency audits and public type checks run consistently across supported Python versions.

- Tightened `orchestrator.route_plan.v3` with exact per-strategy role contracts, resilient provider-diverse fusion planning, Crupier-owned cost/latency estimates, validated-plan authorship in traces, and auditable primary/fallback orchestrator outcomes.
- Added a public live end-to-end routing harness covering autonomous single, cascade, fusion, critique-repair, tool-ledger review, delegate, native image, and native PDF routes without forcing executor models.
- Added a public real-provider operations harness covering model-powered operation classification, multiple embedding backends, reranking, TTS-to-transcription, image generation/editing, Python compatibility, route events, and the complete loopback HTTP API without persisting generated media or vectors.
- Versioned tool critique/repair prompts and made unstructured repair output use a parsed final-answer envelope so tool ledgers, critic notes, and internal verification material cannot leak into the user-facing response.
- Made empty provider text responses retryable and cost-visible, required two planned panel models, and made fusion fail closed unless at least two non-empty panel contributions reach the judge.
- Made general-purpose default profiles genuinely model-orchestrated, added hard `min_panel_size`/`max_panel_size` policy constraints, and preserved explicit profile strategies as user-owned locks rather than model suggestions.
- Added budget-aware fallback across validated models for generator, critic, repair, tool, judge, and final-writer roles, and isolated public live-validation cases so circuit state remains a deliberate cross-request test rather than suite-order contamination.
- Versioned general critique/repair prompts and parsed a final-answer envelope so non-tool agent workflows cannot expose drafts, critic commentary, or audit-only intermediate material in the user-facing result.
- Made tool-aware critique/repair consume authoritative tool results, and made cascade fail closed when every candidate response fails sufficiency validation.
- Tightened multimodal execution contracts by explicitly blocking unimplemented OCR, transcript-first video processing, spreadsheets, office-document extraction, and native non-image paths while keeping dedicated audio transcription executable through the operation API.
- Bumped the package to final `0.4.0`; publishing remains gated on full local and real-provider verification.

## 0.3.0 - 2026-06-21

- Added configurable project scoring weights, weighted task-signal classification, and `crupier scoring suggest` for conservative eval/feedback-driven scoring updates.
- Added model-orchestrator fallback models, versioned orchestrator prompts, richer CLI orchestrator settings, and shared `.crupier/profiles/` profile presets.
- Completed executable route strategies for cascade validation/escalation, parallel panel/fusion execution, iterative tool loops, and bounded `delegate` sub-routes.
- Added declarative policy rules for deny/required capability checks without patching core policy code.
- Added provider retry jitter, per-provider circuit breakers, and route-time degraded-provider exclusion when another provider remains available.
- Improved online model refresh reporting with pricing and profile/capability change details, plus the `models refresh` command.
- Tightened multimodal execution contracts by explicitly blocking unimplemented OCR, audio/video transcription, spreadsheets, office-document extraction, and native non-image paths.
- Bumped the package to final `0.3.0`; publishing remains gated on full local and real-provider verification.

## 0.2.0 - 2026-06-20

- Added curated model decision profiles for OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud so discovered models are separated from production-default, specialized, opt-in, legacy, and deprecated routing choices.
- Added `models list --recommended`, `models show`, and orchestrator CLI/SDK configuration so users can choose their project allowlist and route-orchestrator model explicitly; expensive OpenAI `o3`/`o4-mini` family models now require opt-in rather than appearing in the default recommended set.
- Tightened the production-default set so uncurated discovered models stay visible but require opt-in, and failed capability probes override inferred family support.
- Added operational-provider filtering so runtime routing only selects models visible to the configured API key, with explicit offline simulation opt-out.
- Tuned Google Gemini short-output calls with minimal thinking configuration, fixing false probe failures for Gemini 3.5 Flash text and JSON checks.

## 0.1.0 - 2026-06-20

- Added multi-provider routing core with dry-run and real execution paths.
- Added OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud adapters with explicit local-host override support.
- Added a real optional OpenRouter BYOK adapter using OpenAI-compatible SDK calls, explicit default host configuration, and onboarding env-key coverage.
- Added model discovery, capability probes, readiness checks, and registry snapshots.
- Added bounded provider-call retries with backoff, trace-visible failed attempts, per-request retry overrides, and fallback-compatible retry accounting.
- Added provider timeout handling for OpenAI, Anthropic Claude, Ollama, OpenRouter, and Google Gemini client configuration.
- Added OpenAI unsupported-parameter repair so models that reject optional parameters such as `temperature` can retry once without the unsupported field.
- Added structured output validation/repair, native JSON-schema request formatting for OpenAI/OpenRouter, Google, and Ollama, tool execution, multimodal file planning, and local trace storage.
- Added OpenAI-compatible client, optional autopatch, and HTTP proxy surfaces.
- Added project adoption audit, adoption doctor, production readiness gates, code comments, compare evals, and human feedback signals.
- Added adoption `review_contract` summaries that separate technical readiness from human approval and block auto-approval while human gates remain open.
- Hardened the OpenAI-compatible local server defaults so non-loopback binds require explicit opt-in and browser CORS is disabled unless an origin is configured.
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
- Added workplace-style offline examples for customer support triage, agentic pull-request review, multimodal claim review, drop-in AI boundary adoption, and a larger multi-workflow operations hub, plus fuller routing/model-compare datasets.
- Added adoption handoff reports for reviewer-facing rollout actions, decision templates, and artifacts.
- Added project-level adoption signoff records and production gates for human rollout approval/rejection.
- Added production doctor checks for human feedback applied to capability cards.
- Added release readiness checks and CI workflow.
- Reworked README.md as a developer-first guide covering installation, SDK usage, provider setup, routing, adoption paths, multimodal input, operations, and release gates.
- Clarified developer-facing adoption language so Crupier is positioned as a multi-provider AI orchestration layer, with OpenAI-compatible proxy/client support as an optional adoption path rather than the core scope.
- Clarified multimodal README guidance to include Google Gemini native image execution support alongside OpenAI, Anthropic Claude, and Ollama adapters.
- Kept the public repository and built distributions focused on package code, README, examples, changelog, issues, and onboarding artifacts rather than exposing internal planning documents or extra community policy files.
- Added release checks that block reintroducing internal `docs/`, extra community policy files, or blank public issue forms before publication.
- Added release checks for broken public Markdown code fences and relative links before publication.
- Added public GitHub YAML syntax checks for workflows, Dependabot, and issue templates before publication.
- Added a tracked-file secret pattern scan so provider-key shaped credentials cannot silently enter the public repository.
- Added a repository `.gitignore` release check so local keys, caches, builds, and generated Crupier artifacts stay protected before public release.
- Added a release check that keeps README links absolute for PyPI rendering.
- Added PyPI publish workflow tag/manual-version matching and documented the final public release order.
- Hardened the PyPI publish workflow so draft or non-final GitHub Releases cannot publish the final package.
- Hardened the PyPI publish workflow so GitHub Release and manual publish paths only run from `main`.
- Hardened the PyPI publish workflow so the checked-out release/manual commit must match `origin/main` before building or uploading distributions.
- Serialized PyPI publish workflow runs per ref so duplicate release/manual triggers cannot race each other.
- Scoped PyPI trusted-publishing permissions to the publish job and linked the `pypi` environment to the package page.
- Hardened the PyPI publish workflow so the first public upload requires an available project name while later releases allow the already-owned PyPI project.
- Aligned the repository development allowlist with currently verified OpenAI, Anthropic Claude, and Ollama Cloud models, and clarified the provider-readiness remediation path when real checks need capability probes.
- Aligned the repository development provider config with the final OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud readiness gate.
- Added public package discovery metadata for AI-oriented PyPI classification and GitHub repository topics.
- Added Trove classifier validation to the dev release path so invalid PyPI classifiers are caught before upload.
- Added a release guard requiring public PyPI classifiers for every Python version tested in CI.
- Added release-guide guards so final provider-readiness commands include Google Gemini alongside OpenAI, Anthropic Claude, and Ollama Cloud.
- Added a global `--env-file` CLI option for loading local ignored provider keys during real-provider checks without passing secrets as command arguments.
- Added a public model-example release guard so README and CLI examples stay aligned with the current default allowlist.
- Added public repository settings guidance for focused GitHub surfaces, squash-only merges, vulnerability reporting, and secret scanning before opening the project.
- Added public branch-protection guidance for `main` after the final single release commit, with required CI and no force pushes before accepting public changes.
- Tightened the PyPI publishing workflow so trusted publishing runs through the `pypi` environment and repeats the first-upload PyPI name preflight.
- Added CONTRIBUTING.md with local development, provider-key handling, PR, public onboarding, and release-gate expectations.
- Added MIT license metadata for package release readiness.
- Tightened public PyPI metadata checks for keywords and classifiers without using legacy license classifiers rejected by modern setuptools.
- Added a non-blocking project URL readiness warning so public PyPI uploads do not ship with placeholder or missing repository links by accident.
- Added sdist + wheel build, artifact content inspection, `twine check`, py.typed packaging, and wheel/sdist install/import/CLI/`crupier init`/dry-run route/Python SDK quickstart smoke validation to release checks.
- Added built wheel/sdist PyPI metadata inspection for name, version, summary, Python requirement, license, project URLs, classifiers, and extras.
- Excluded the repository-local `crupier.toml` from release artifacts and added a guard so local project config cannot leak into PyPI distributions.
- Excluded repository tests from PyPI distributions and added an artifact guard so package uploads stay focused on runtime code, public examples, and onboarding files.
- Kept Ollama Cloud/local REST support dependency-free in the base package and added release guards against incomplete provider extras or reintroducing the unused Ollama Python SDK dependency.
- Added `crupier release check --strict-public` so PyPI publishing blocks on warnings such as missing public project URLs and on skipped build/install smoke checks.
- Added opt-in `crupier release check --verify-providers` for final public release gates with real provider discovery, readiness, and smoke validation.
- Increased real-provider smoke output budget to avoid false negatives on reasoning-heavy models that spend some tokens before emitting text.
- Added release artifact inspection for the offline SDK example so the sdist keeps first-run onboarding examples.
- Added release smoke validation for public examples from the built sdist so packaged examples stay executable without provider keys.
- Added wheel/sdist install smoke validation for exported public API names in `crupier.__all__`.
- Added `crupier --version` and release smoke validation for installed CLI version output.
- Added release smoke validation for installed `python -m crupier --version` module execution.
- Added release checks and artifact inspection for CONTRIBUTING.md so public development guidance is shipped and maintained.
- Added a PyPI publish workflow for GitHub Release based publishing.
- Added a release-language guard so final package metadata and public onboarding files do not regress into non-final release labels.
- Added optional `crupier release check --check-pypi-name` with `--allow-existing-pypi-project` for first-upload and maintenance-release PyPI name preflights.
- Added `crupier release check --verify-project-urls` for public package-link reachability checks before PyPI upload.
- Added public collaboration templates and release checks for bug reports, feature requests, and pull requests.
- Added required issue-template safety confirmations so public bug reports and feature requests ask reporters to remove keys, prompts, provider responses, customer data, `.env`, and `.crupier/` artifacts before posting.
- Added Dependabot configuration for Python tooling and GitHub Actions plus release checks for dependency-update automation and minimal CI permissions.
- Added public repository guidance and release-check coverage for enabling Dependabot security updates before opening the repository.
- Updated CI/publish workflows to current GitHub Actions majors and added checked distribution artifact upload before PyPI publishing.
- Expanded SECURITY.md with scope, private reporting, supported versions, disclosure process, and secret-handling guidance, and added release checks for that content.
- Added `pip-audit` to the development workflow plus CI and publish dependency vulnerability audit steps.
- Added Ruff critical lint (`E9,F63,F7,F82`) to dev, CI, publish, and release-readiness checks.

# Crupier

Crupier is a Python orchestration SDK for AI agents and applications that need to choose, combine, audit, and update model routes across providers.

Long-term goal: Crupier should also work as a drop-in orchestration layer for existing AI projects, through an OpenAI-compatible proxy, Python autopatch, compatible clients, or the native SDK.

Current status: `0.1.0`. The current implementation focuses on the local core, drop-in adoption workflows, human review gates, and real execution for OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud. A local Ollama daemon can still be configured explicitly for development or private deployments.

## Installation

Install only the provider SDKs you need:

```bash
pip install crupier
pip install "crupier[openai]"
pip install "crupier[anthropic]"
pip install "crupier[google]"
pip install "crupier[ollama]"
```

For a new project:

```bash
crupier init
cp .env.example .env
crupier models discover --provider openai
crupier models allow openai:gpt-5.4-mini --replace
crupier update --online
crupier verify
crupier deal "Say hello in one sentence" --mode fast --no-dry-run
```

`crupier init` writes `crupier.toml`, `.env.example`, `.gitignore` entries for local secrets/artifacts, and the `.crupier/` project directories. For an existing OpenAI-compatible app, start with `crupier serve` or `crupier.compat.openai.OpenAI` before trying the global autopatch helper. Dry-run routing is the default; real provider calls require API keys in environment variables or a local `.env`.

## Python SDK Quickstart

The base package can plan routes without provider SDKs or API keys:

```python
from crupier import Crupier

crupier = Crupier.from_project()

result = crupier.deal(
    task="Choose a model route for a short support reply.",
    input={"priority": "normal", "message": "Where is my invoice?"},
    mode="agentic",
    dry_run=True,
    trace="summary",
)

print(result.route.strategy)
print(result.route.model_summary)
```

Set `dry_run=False` only after installing the relevant provider extra, setting API keys, and selecting allowed models for the project.

You can also run the repository example without provider SDKs or keys:

```bash
python examples/sdk_dry_run.py
```

Implemented now:

- `Crupier.from_project()`, `from_toml()`, `from_config()`
- `deal()`, `adeal()`, and basic `stream()`
- dry-run `RoutePlan` execution
- `DecisionTrace` summaries
- `crupier.toml` loading
- local seed `CapabilityCard` registry
- online model discovery and capability-card refresh
- registry snapshots with create/list/diff/use
- capability probes for text generation and JSON-instruction adherence
- provider readiness verification with OpenAI baseline
- model-kind classification for chat vs embedding models
- embedding capability probing and dimensions capture
- multimodal/file input planning with `FileAsset` metadata and representation plans
- real native image execution path for OpenAI, Anthropic Claude, and Ollama adapters
- real extracted text/code/PDF file execution path for local files
- `PlanningContext`, `Orchestrator` interface, and deterministic orchestrator baseline
- opt-in `ModelOrchestrator` with JSON route plans, validation, one repair attempt, and deterministic fallback
- sensitive-route guardrails that block model-orchestrator downgrades for high-risk agentic/tool requests
- strict `RoutePlan` shape validation before policy/provider execution
- pre-execution cost estimates and hard per-request budgets
- real structured-output execution with local JSON parsing, validation, and one repair attempt
- provider-agnostic local tool execution with approval guardrails and idempotency ledger
- routing eval runner with human-readable and JSON reports
- project adoption audit with human route-review prompts and optional real canaries
- one-command project adoption doctor with gates for audit, patches, eval history, and human feedback
- adoption handoff reports that collect human actions, commands, and recent review artifacts
- programmer code comments for AI integration hotspots
- project-local human feedback recording and application to model scoring
- human review packets for A/B compare reports with ready-to-run feedback commands
- editable human decision templates for reviewer-filled verdict import
- opt-in persistent trace storage, inspection, deletion, and replay
- package release readiness checks with sdist/wheel, `twine check`, and wheel/sdist install smoke validation
- CI workflow for tests, distribution build, and release readiness
- initial OpenAI-like compatibility client: `crupier.compat.openai.OpenAI`
- opt-in OpenAI SDK autopatch helper: `crupier.install("openai")`
- initial OpenAI-compatible HTTP server: `crupier serve`
- initial OpenAI-compatible embeddings endpoint/client method
- OpenAI-like streaming shapes for Responses and Chat Completions
- OpenAI-like HTTP error payloads and request IDs in the compatibility server
- `crupier init`, `update`, `models`, `registry snapshot`, `capabilities probe`, `verify`, `profiles list`, `route`, `deal`, and `smoke`
- real text provider calls with `dry_run=False` for OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud
- fallback, panel, fusion, and critique-repair execution paths
- typed errors

Roadmap after `0.1.0`:

- online price refresh
- production-calibrated model orchestrator evals
- production-calibrated eval datasets beyond the built-in routing checks
- provider-native structured-output parameter mapping beyond prompt+validate execution
- provider-native PDF/audio/video/document execution
- table-aware PDF extraction, OCR, audio/video transcription, and office-document parsing
- full SDK response compatibility matrix
- provider-native tool-calling execution optimizations
- provider-native streaming proxy
- docs website

## Real Provider Calls

Dry-run is the default. To call real providers, install the relevant extra, set API keys in environment variables, and pass `dry_run=False`:

```bash
pip install "crupier[openai]"
export OPENAI_API_KEY="..."
crupier deal "Say hello in one sentence" --mode fast --no-dry-run
```

For Anthropic Claude:

```bash
pip install "crupier[anthropic]"
export ANTHROPIC_API_KEY="..."
```

For Google Gemini:

```bash
pip install "crupier[google]"
export GOOGLE_API_KEY="..."  # GEMINI_API_KEY is also accepted.
```

For direct Ollama Cloud API access:

```toml
[providers.ollama]
enabled = true
host = "https://ollama.com/api"
env_key = "OLLAMA_API_KEY"
```

```bash
export OLLAMA_API_KEY="..."
```

Never pass API keys as CLI arguments or commit them to `crupier.toml`.

Crupier also loads a local `.env` file from the project directory when `crupier.toml` is loaded. Existing exported environment variables win over `.env` values, and `.env` is ignored by this repository's `.gitignore`.

For Ollama Cloud, set the host through config or environment:

```bash
export OLLAMA_HOST="https://ollama.com/api"
```

When `OLLAMA_HOST` is present, Crupier uses it for the Ollama adapter host.

For a local Ollama daemon, override the host explicitly:

```toml
[providers.ollama]
enabled = true
host = "http://localhost:11434"
```

## Discover and Select Models

Query models available to enabled providers:

```bash
crupier models discover --provider openai
crupier models discover --provider anthropic
crupier models discover --provider google
crupier models discover --provider ollama
```

Then select allowed project models:

```bash
crupier models allow openai:gpt-5.5 anthropic:claude-opus-4-8 --replace
crupier models allow ollama:gpt-oss:120b
```

`claude:...` is accepted as an alias and normalized to `anthropic:...`.

Refresh local capability cards from the models that exist and are available to your account at that moment:

```bash
crupier update --online --dry-run
crupier update --online
crupier update --online --provider openai
crupier update --online --json
```

The human output shows added, removed, modified, unchanged, and registry state counts. `models list` also labels models as `discovered`, `allowed`, `locked`, `stale`, `builtin`, or `local`.

This does not automatically enable every discovered model. Use `crupier models allow ...` to select the project allowlist. For production, keep using explicit model IDs and registry snapshots rather than dynamic aliases.

## Registry Snapshots

Freeze the registry state used by a project:

```bash
crupier registry snapshot create baseline --allowed-only
crupier registry snapshot list
crupier registry snapshot diff baseline
```

Restore local capability cards from a snapshot:

```bash
crupier registry snapshot use baseline
crupier registry snapshot use baseline --restore-allowlist
```

Snapshots live under `.crupier/registry/snapshots/` and include capability cards plus the allowlist active when they were created. Use `--allowed-only` for production route reproducibility.

## Capability Probes

Run capability probes against allowed models:

```bash
crupier capabilities probe --provider openai
crupier capabilities probe --model openai:gpt-5.4-mini
crupier capabilities probe --dry-run
```

Persist probe results into capability cards:

```bash
crupier capabilities probe --provider openai --apply
crupier capabilities probe --provider openai --probe text_basic --probe json_instruction --apply
crupier capabilities probe --provider openai --probe structured_output --probe tool_call --probe streaming --apply
crupier capabilities probe --model openai:text-embedding-3-small --probe embeddings --apply
```

Available probes: `text_basic`, `json_instruction`, `max_output_param`, `structured_output`, `tool_call`, `streaming`, and `embeddings`. OpenAI, Anthropic, and Ollama adapters expose provider-native probes where the underlying account/model supports them. `embeddings` is explicit rather than part of the default chat probe set, and readiness uses it for models marked `model_kind="embedding"`. Probe storage does not include raw prompts or responses.

Routing treats `verified` capability evidence as stronger than `inferred` metadata. If a request sets `constraints={"require_verified_capabilities": True}`, models with only inferred support are filtered out for required tools, structured output, streaming, embeddings, or multimodal requirements.

Ollama Cloud and explicit local Ollama hosts are not treated as "all models are vector models." Crupier marks dedicated embedding models by name and confirms them with `--probe embeddings`; chat models stay chat models unless a real probe proves otherwise.

Check production readiness for the allowlist:

```bash
crupier capabilities readiness
crupier capabilities readiness --strict
crupier capabilities readiness --provider openai --json
```

## Real Smoke Tests

After selecting allowed models and setting environment variables, run a real minimal call:

```bash
crupier smoke --provider openai
crupier smoke --provider anthropic
crupier smoke --provider ollama
```

Test exact models:

```bash
crupier smoke --model openai:gpt-5.5
crupier smoke --model anthropic:claude-opus-4-8
crupier smoke --model ollama:gpt-oss:120b
```

By default, smoke tests do not print model output. Add `--show-output` only for harmless prompts/debugging.

## Provider Readiness

Run one command to check config, environment variables, model discovery, capability-card readiness, and real smoke calls:

```bash
crupier verify
crupier verify --provider anthropic --provider google --provider ollama
crupier verify --json
```

OpenAI is included as a baseline by default, even when you ask to verify Anthropic, Google, or Ollama. Use `--no-openai-baseline` only when you explicitly want to skip that baseline.

Statuses:

- `blocked`: config, adapter, env, or allowlist is missing.
- `failed`: discovery, readiness, or smoke failed.
- `needs_probes`: real smoke can work, but capability cards still need verified probes.
- `ready`: selected models passed config/env/discovery/readiness/smoke checks.

## Release Readiness

Before publishing or handing a package build to another project, run:

```bash
python -m ruff check src tests --select E9,F63,F7,F82
python -m pip_audit --skip-editable --progress-spinner off
crupier release check
crupier release check --skip-build
crupier release check --json
crupier release check --strict-public
crupier release check --check-pypi-name
crupier release check --verify-providers --provider openai --provider anthropic --provider ollama
```

The check validates package metadata/classifiers, project URL readiness, final public version shape, non-final release language, version sync, typed package marker, README onboarding content, public collaboration templates, safe `crupier init` defaults, vulnerability reporting and secret-handling policy, changelog, CI workflow, critical lint wiring, dependency-update automation, dependency vulnerability audit wiring, provider extras, console script, license metadata, sdist + wheel build, `twine check`, and clean wheel/sdist install/import/CLI/init/dry-run route/Python SDK smokes in temporary virtual environments. Add `--strict-public` before a PyPI upload to fail on any remaining warning or skipped build. Add `--check-pypi-name` before a first upload to verify the configured package name against PyPI. Add `--verify-providers` when real provider keys are available; it appends a blocking provider readiness check with discovery, readiness, and smoke validation.

For the public release procedure, see [docs/crupier-publishing.md](docs/crupier-publishing.md).
For development and pull request expectations, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Routing Evals

Run product-facing routing checks:

```bash
crupier eval run
crupier eval run --orchestrator-mode hybrid
crupier eval run --dataset examples/routing-eval.json --write-report
crupier eval compare "Answer this support ticket" --model openai:gpt-4.1-mini --model anthropic:claude-sonnet-4-6
crupier eval compare-dataset --dataset examples/model-compare-eval.json --model openai:gpt-4.1-mini --model anthropic:claude-sonnet-4-6
```

These evals check human-relevant expectations such as strategy, number of models, providers, and route roles. Hybrid/model modes may call the configured orchestrator model, but still validate and fall back before execution.

`eval compare` compares variants for one task and prints a recommended winner based on deterministic checks, estimated/actual cost, latency, model count, and human review questions. It is dry-run by default; add `--no-dry-run` plus small budgets when you want real provider outputs:

```bash
crupier eval compare "Reply exactly crupier-ok" \
  --model openai:gpt-4.1-mini \
  --model anthropic:claude-sonnet-4-6 \
  --expect-contains crupier-ok \
  --max-cost-usd 0.02 \
  --no-dry-run
```

`eval compare-dataset` repeats the comparison over a JSON/JSONL eval dataset and aggregates scores by model and mode. Use `--record-history` to persist metadata-only aggregate results, then inspect trends with `eval history`. Add `--apply` when you want sufficiently confident aggregate scores written into capability cards as `eval:<mode>` signals for future routing:

```bash
crupier eval compare-dataset \
  --dataset examples/model-compare-eval.json \
  --model openai:gpt-4.1-mini \
  --model anthropic:claude-sonnet-4-6 \
  --record-history

crupier eval history

crupier eval history \
  --min-count 10 \
  --min-confidence high \
  --apply
```

The history file lives under `.crupier/evals/history/` and stores dataset/run/model metrics, not raw prompts or responses.

## Project Doctor

For a repo you just downloaded or a project you want to retrofit, start with the non-destructive doctor:

```bash
crupier adopt plan
crupier adopt patches --path recommended --write-report
crupier adopt package
crupier adopt handoff --write-report
crupier adopt doctor
crupier adopt doctor --real --provider anthropic --provider google --provider ollama
crupier adopt doctor --production --real --provider anthropic --provider google --provider ollama
crupier adopt handoff --production --real --provider anthropic --provider google --provider ollama --write-report
crupier adopt signoff --verdict approve --handoff .crupier/handoffs/adoption_handoff_YYYYMMDDTHHMMSSZ.md
crupier adopt doctor --write-report --json
```

`adopt package`, `adopt doctor`, `adopt plan`, `adopt patches`, and offline `adopt handoff --write-report` can run on a freshly cloned repo before `crupier init`; they infer a project name from `package.json`, `pyproject.toml`, or the directory name and write non-destructive review artifacts. `adopt package` is the one-command human handoff: it writes code comments, PR/review-comment packets, SARIF annotations, an editable programmer decision template, patch guidance, a doctor report, a handoff, and a persistent package index under `.crupier/packages/`. Real/production doctor, package, and handoff runs use the full Crupier project config because they combine provider readiness, eval history, human feedback gates, and production signoff.

The doctor combines adoption planning, patch suggestions, audit checks, real canary status, A/B eval history, human feedback signals, project-level adoption signoff, and programmer code comments into `pass` / `warn` / `fail` gates. It also emits a `review_contract` that separates technical evidence from human approval and sets `must_not_auto_approve=true` until the human gates are closed. Default `adoption` mode uses warnings for missing evidence so a maintainer can start safely. `--production` makes real canaries, recorded eval history, human feedback, and an explicit adoption signoff mandatory, because technically passing output is not enough if a human would reject the route/result.
`adopt handoff` wraps the doctor into a reviewer-facing JSON/Markdown package under `.crupier/handoffs/`, including recent compare reports, feedback review packets, human decision templates, code-comment reports, required human actions, and the next Crupier commands to run.
`adopt signoff` records the human adoption decision as metadata under `.crupier/handoffs/signoffs.jsonl`. Use `--verdict reject` or `--verdict needs_work` when the code path is green but the result, cost, maintainability, or rollout plan is not acceptable.

## Project Audit

Before putting Crupier in front of an existing project, run an adoption audit:

```bash
crupier audit
crupier audit --real --provider anthropic --provider google --provider ollama
crupier audit --real --write-report
```

The audit combines configuration checks, capability readiness, routing evals, human route-review prompts, optional real provider canaries, and source-code comments. It is deliberately stricter than "the code ran": it asks whether a maintainer can understand the route, whether privacy defaults are safe, and whether real calls work within small budgets.

Real audit canaries include text smoke, structured output, a local tool loop, local text-file context, and native image input on allowlisted chat/vision models. Outputs are not printed by default.

## Adoption Plan

Ask Crupier which integration path fits an existing project:

```bash
crupier adopt doctor
crupier adopt plan
crupier adopt plan src app.py --write-report
crupier adopt plan --json
crupier adopt patches --path compat_client
```

The plan recommends one of:

- `proxy`: run `crupier serve` and point OpenAI-compatible clients at it.
- `compat_client`: replace narrow Python OpenAI imports with `crupier.compat.openai.OpenAI`.
- `autopatch`: use `crupier.install("openai")` only in controlled experiments.
- `native_sdk`: wrap the project's AI boundary with `Crupier.from_project(...).deal(...)`.

It also reports blockers such as inline credentials, framework hints, a rollout checklist, and the programmer comments that justify the recommendation.

`adopt patches` generates suggested diffs, snippets, and commands, but does not modify project files:

```bash
crupier adopt patches --path recommended --write-report
crupier adopt patches --path proxy
crupier adopt patches --path native_sdk
```

Use these as review artifacts for another programmer; apply changes manually only after tests/evals pass.

Generate only programmer comments for a project:

```bash
crupier code comments
crupier code comments src app.py --write-report
crupier code comments src app.py --write-review-comments
crupier code comments src app.py --write-sarif
crupier code comments src app.py --write-decisions-template
crupier code comments src app.py --import-decisions .crupier/code-comments/decisions/code_comment_decisions_YYYYMMDDTHHMMSSZ.json
crupier code comments src app.py --ack-reviewed
```

Comments flag OpenAI/Anthropic/Ollama/Google call sites, hard-coded model choices, and plausible inline credentials so another programmer can decide between proxy, compatible client, autopatch, or native SDK integration. `--write-review-comments` writes PR/review-comment Markdown and JSONL under `.crupier/code-comments/`, with fingerprints and comment bodies but no source snippets. `--write-sarif` writes SARIF annotations for GitHub Code Scanning and similar tools. Generated/dependency directories such as `build`, `dist`, `.venv`, `node_modules`, and `*.egg-info` are skipped to keep review artifacts focused on source code; redaction-regex examples and short hyphenated identifiers are not treated as credentials. Credential-like values inside `tests/`, `fixtures/`, and test-named files are still shown to reviewers, but as P3 test fixtures instead of production P1 credential blockers.
`--write-decisions-template` creates an editable JSON checklist where each comment can be marked `accepted`, `false_positive`, `needs_change`, or `unresolved`; `--import-decisions` records only reviewed/resolved fingerprints so pending comments keep the doctor gate open. `--ack-reviewed` is the bulk path after a programmer has reviewed the full current set.

## Human Feedback

When a route technically passes but a human reviewer decides the result is not good enough, record that judgement:

```bash
crupier feedback record \
  --model openai:gpt-4.1-mini \
  --mode agentic \
  --rating 2 \
  --verdict needs_work \
  --tag weak_code_review

crupier feedback summary
crupier feedback apply
```

If the judgement comes from an eval comparison, write a report first and let Crupier infer models, mode, and strategy from the reviewed variant:

```bash
crupier eval compare "Answer this support ticket" \
  --mode fast \
  --model openai:gpt-4.1-mini \
  --model anthropic:claude-sonnet-4-6 \
  --no-dry-run \
  --write-report

crupier feedback review \
  --compare-report .crupier/evals/runs/compare_YYYYMMDDTHHMMSSZ.json \
  --write-report \
  --write-decisions-template

crupier feedback record \
  --compare-report .crupier/evals/runs/compare_YYYYMMDDTHHMMSSZ.json \
  --variant openai:gpt-4.1-mini \
  --rating 2 \
  --verdict needs_work \
  --tag weak_answer

crupier feedback import-decisions \
  --decisions .crupier/feedback/decisions/human_decisions_YYYYMMDDTHHMMSSZ.json \
  --apply-to-registry
```

`feedback review` creates JSON and Markdown review packets under `.crupier/feedback/reviews/`. Each review item includes route metadata, human checks, optional output preview from the compare report, and ready-to-run `feedback record` commands for `accept`, `needs_work`, or `reject`. `--write-decisions-template` also writes an editable metadata-only JSON file under `.crupier/feedback/decisions/`; a human sets `record=true`, `rating`, `verdict`, optional tags/note, then `feedback import-decisions --apply-to-registry` records and applies the judgement. Decision templates omit output previews and feedback storage still redacts notes. Feedback from dry-run compare reports is rejected by default for production; pass `--allow-dry-run-source` only for non-production calibration.

If the route was stored as a trace, Crupier can infer models, mode, and strategy without storing prompts or responses:

```bash
crupier deal "Plan this agent step" --store-trace --trace summary
crupier feedback record --trace-id trc_... --rating 1 --verdict reject --tag wrong_route
crupier feedback apply
```

Feedback lives under `.crupier/feedback/` as project-local metadata. `feedback apply` writes aggregate scores such as `human:agentic` into capability cards, and future `crupier route` output shows a `human_feedback` score term when that signal changes ranking.
Production `adopt doctor` checks both sides: feedback must be recorded and the resulting `human:<mode>` scores must be applied to capability cards, otherwise the selector has not learned from the human judgement yet.

## Trace Storage

Trace persistence is opt-in. By default, Crupier does not store prompts or responses.

```bash
crupier deal "Plan this route" --store-trace --trace summary
crupier trace list
crupier trace show trc_...
crupier trace delete trc_...
```

Metadata-only traces are inspectable but not replayable. To allow replay, explicitly store prompt/input data:

```bash
crupier deal "Plan this route" --store-prompt --store-response --trace summary
crupier trace replay trc_...
```

Stored traces live under `.crupier/traces/`. Secret-like values are redacted before writing.

## Explainable Model Selection

Inspect the routing decision without provider calls:

```bash
crupier route "Compare two agent architectures and critique risks" --mode research
crupier route "Extract invoice fields" --response-schema '{"type":"object","properties":{"total":{"type":"number"}},"required":["total"]}'
crupier route "Answer briefly" --max-cost-usd 0.001
```

The route output includes score terms such as quality tier, profile preferences, task signals, tool support, structured-output support, local eval scores, human feedback, and penalties for deprecation, preview/experimental models, high cost, or latency.

`RoutePlan.estimated_cost` is computed by Crupier, not trusted from the model orchestrator. Until online price refresh is implemented, models use capability-card pricing when available and conservative tier defaults otherwise. `max_cost_usd` blocks execution before provider calls.

## Structured Output

Python:

```python
schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "total": {"type": "number"}},
    "required": ["name", "total"],
    "additionalProperties": False,
}

result = crupier.deal(
    task="Extract invoice data",
    input="Invoice for Ada, total 12.50",
    response_schema=schema,
    dry_run=False,
)

print(result.output_json)
```

CLI:

```bash
crupier deal "Extract invoice data" \
  --input "Invoice for Ada, total 12.50" \
  --response-schema '{"type":"object","properties":{"name":{"type":"string"},"total":{"type":"number"}},"required":["name","total"]}' \
  --no-dry-run
```

Crupier validates returned JSON locally and makes one repair attempt with the same model before failing with `CrupierStructuredOutputError`.

## Tools

Python callables can be passed as local tools:

```python
def lookup_user(name: str):
    """Look up a user in the project database."""
    return {"name": name, "id": "usr_123"}

result = crupier.deal(
    task="Find Ada and summarize the result",
    tools=[lookup_user],
    dry_run=False,
)

print(result.output_text)
print(result.provider_metadata["tool_calls"])
```

Sensitive tools require approval:

```python
tool = {
    "name": "write_file",
    "description": "Write a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    "handler": write_file,
    "requires_approval": True,
}

result = crupier.deal(
    task="Write the changelog",
    tools=[tool],
    constraints={"approved_tools": ["write_file"]},
    dry_run=False,
)
```

Tool execution is provider-agnostic in `0.1.0`: Crupier asks the selected model for a JSON tool plan, executes approved local tools, deduplicates identical tool calls with idempotency keys, and sends tool results back for the final answer. Provider-native tool call execution is still planned as an optimization.

## File and Multimodal Planning

Plan how Crupier would route an image, PDF, audio/video file, spreadsheet, document, or code file:

```bash
crupier route "Extract totals from this invoice" --file invoice.png --json
crupier route "Summarize this contract" --file contract.pdf --json
```

Python:

```python
result = crupier.deal(
    task="Extract totals from this invoice.",
    files=["invoice.png"],
    trace="summary",
)

print(result.route.input_plan)
```

Images route to native vision-capable models and can execute with `dry_run=False` through the OpenAI, Anthropic Claude, and Ollama adapters when the selected model supports image input. Text, Markdown, JSON, YAML, HTML/CSS, code files, and PDFs can execute as extracted text context; PDF extraction uses `pypdf` from `crupier[pdf]` when installed or a local `pdftotext` binary when available. Use `constraints={"require_native_file_input": True}` when a project explicitly wants provider-native file input. Native PDF/audio/video/office-document execution, OCR, table-aware PDF extraction, and transcription are still pending.

## OpenAI-Like Compatibility

For projects that already look like OpenAI SDK code:

```python
from crupier.compat.openai import OpenAI

client = OpenAI(project=".")
response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Summarize this"}],
)

print(response.choices[0].message.content)
```

Opt-in monkeypatch for experiments:

```python
import crupier
crupier.install("openai")
```

Implemented now: `responses.create`, `chat.completions.create`, `embeddings.create`, attribute/dict-style response objects, OpenAI-like stream events/chunks, `model_dump()`, OpenAI content-part image/file extraction into Crupier file planning, `strict` vs `balanced` model behavior, and an initial local HTTP server.

Run a local OpenAI-compatible HTTP server:

```bash
crupier serve --port 8787
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
```

Implemented endpoints: `GET /health`, `GET /v1/models`, `POST /v1/responses`, `POST /v1/chat/completions`, and `POST /v1/embeddings`. The server returns OpenAI-like JSON errors, `x-request-id`, typed Responses SSE events, and Chat Completions chunks. Add `--no-dry-run` when you want the proxy to call real providers.

## Quick Start

```bash
crupier init
crupier update
crupier models list
crupier deal "Plan the best model route for this agent step" --mode agentic --trace summary
```

Python:

```python
from crupier import Crupier

crupier = Crupier.from_project()

result = crupier.deal(
    task="Review this agent plan and choose the best model route.",
    input={"step": "write files after analysis"},
    mode="agentic",
    trace="summary",
)

print(result.output_text)
print(result.route.summary)
```

## Design Docs

- `docs/crupier-product-discovery.md`
- `docs/crupier-architecture.md`
- `docs/crupier-api-design.md`
- `docs/crupier-drop-in-adoption.md`
- `docs/crupier-programmer-guide.md`

# Crupier

Crupier is a Python orchestration SDK for AI applications and agents that need to choose, combine, audit, and improve model routes across providers.

It is designed for two situations:

- New projects that want one AI boundary instead of hard-coding provider/model choices throughout the codebase.
- Existing AI projects, agents, SDK integrations, or provider-specific codebases that want to add model selection, provider fallback, audits, evals, and human review without a full rewrite.

Crupier is a BYOK orchestration layer: it runs with your own provider accounts across OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, optional OpenRouter BYOK, or your own integration boundary. It keeps prompts/responses out of persistent logs by default and routes each request toward the best available model or model family for the task, quality target, latency, cost budget, and project policy.

Current public package version: `0.3.0`.

## What It Does

Crupier gives a project a single place to answer:

- Which provider/model should handle this request?
- Should the route be single-model, fallback, panel, fusion, or critique-repair?
- Does the request need tools, structured JSON, image input, embeddings, low latency, low cost, or stronger reasoning?
- Are the allowed models still available to this account today?
- Do real provider calls, local capability probes, eval history, and human feedback agree with the route?
- Can an existing project adopt this without hiding the tradeoffs from reviewers?

The core idea is simple: your app sends a task to Crupier, Crupier produces a validated route plan, then either executes it or returns the dry-run decision for inspection.

## Installation

Install the base package first:

```bash
pip install crupier
```

Install only the provider SDKs and file helpers you use. Ollama Cloud and explicitly configured local Ollama use Crupier's built-in REST adapter, so they do not require an additional Python SDK:

```bash
pip install "crupier[openai]"
pip install "crupier[anthropic]"
pip install "crupier[google]"
pip install "crupier[openrouter]"
pip install "crupier[pdf]"
pip install "crupier[all]"
```

Optional provider extras:

| Extra | Purpose |
| --- | --- |
| `crupier[openai]` | Native OpenAI calls and OpenAI-compatible OpenRouter adapter support. |
| `crupier[anthropic]` | Native Anthropic Claude calls. |
| `crupier[google]` | Native Google Gemini calls. |
| `crupier[ollama]` | Compatibility extra; Ollama Cloud/local REST support ships in the base package. |
| `crupier[openrouter]` | Optional OpenRouter BYOK adapter through OpenAI-compatible SDK calls. |
| `crupier[pdf]` | PDF text extraction through `pypdf` for file-context execution. |
| `crupier[all]` | All runtime provider/file extras. |

## Create A Project

Initialize Crupier inside your app or agent repo:

```bash
crupier init
cp .env.example .env
```

`crupier init` writes:

- `crupier.toml`
- `.env.example`
- safe `.gitignore` entries for local keys and generated Crupier artifacts
- `.crupier/` project directories

Dry-run routing is the default, so you can inspect decisions before any provider call:

```bash
crupier route "Choose a model for a short support reply" --mode fast
crupier deal "Say hello in one sentence" --mode fast
```

Real execution requires provider SDKs, environment keys, and `--no-dry-run`:

```bash
crupier deal "Say hello in one sentence" --mode fast --no-dry-run
```

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

Run the offline quickstart:

```bash
python examples/sdk_dry_run.py
```

Run workplace-style dry-run examples without provider keys:

```bash
python examples/customer_support_triage.py
python examples/agentic_pr_review.py
python examples/multimodal_claim_review.py
python examples/drop_in_agent_boundary.py
python examples/workflow_operations_hub.py
```

When you are ready to call providers:

```python
result = crupier.deal(
    task="Write a concise customer reply.",
    input={"message": "Can you resend my invoice?"},
    mode="fast",
    dry_run=False,
)

print(result.output_text)
```

## Configure Providers

Crupier uses your provider keys from environment variables or a local `.env` file. Existing exported variables win over `.env` values.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GEMINI_API_KEY=
OLLAMA_API_KEY=
OLLAMA_HOST=https://ollama.com/api
```

Do not pass API keys as CLI arguments, commit `.env`, or put provider keys in `crupier.toml`.
For CLI checks, load a local ignored env file explicitly:

```bash
crupier --env-file .env verify --provider google
crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama
```

Relative `--env-file` paths resolve from `--project`. Existing exported variables are kept, so CI or shell-provided secrets are not overwritten by local files.
If a provider behaves differently than expected, check for stale exported keys first; for a one-off local check you can run `env -u OLLAMA_API_KEY -u OLLAMA_HOST crupier --env-file .env ...` so the ignored `.env` values are definitely used.

Default provider configuration looks like this:

```toml
[providers.openai]
enabled = true
env_key = "OPENAI_API_KEY"

[providers.anthropic]
enabled = false
env_key = "ANTHROPIC_API_KEY"

[providers.google]
enabled = false
env_key = "GOOGLE_API_KEY"

[providers.ollama]
enabled = false
host = "https://ollama.com/api"
env_key = "OLLAMA_API_KEY"

[providers.openrouter]
enabled = false
mode = "byok"
host = "https://openrouter.ai/api/v1"
env_key = "OPENROUTER_API_KEY"
```

Ollama means Ollama Cloud by default:

```bash
export OLLAMA_HOST="https://ollama.com/api"
```

Use a local Ollama daemon only when you explicitly choose it:

```toml
[providers.ollama]
enabled = true
host = "http://localhost:11434"
env_key = "OLLAMA_API_KEY"
```

## Discover And Allow Models

Crupier discovers the models your account can see at that moment:

```bash
crupier models discover --provider openai
crupier models discover --provider anthropic
crupier models discover --provider google
crupier models discover --provider ollama
```

Then you choose which models this project is allowed to use:

```bash
crupier models allow openai:gpt-5.5 openai:gpt-5.4-mini anthropic:claude-sonnet-4-6 --replace
crupier models allow ollama:gpt-oss:120b
```

`claude:...` is accepted as an alias and normalized to `anthropic:...`.

Refresh capability cards from currently available provider models:

```bash
crupier models refresh --dry-run
crupier models refresh
crupier models refresh --provider openai
crupier models refresh --json
```

Discovery does not automatically enable every model. Production projects should keep an explicit `[models].allow` list and use registry snapshots for repeatability.

Online discovery, online updates, probes, readiness checks, and real runtime routing only treat a provider/model as operational when the configured API key can actually query that provider. If a provider is enabled but the key is missing, invalid, rate-limited, or cannot see a selected model, Crupier reports that boundary and excludes those models from automatic routing. Explicit single-provider commands fail clearly; multi-provider discovery can skip non-operational providers with warnings so one bad key does not pollute the operational catalog. Dry-run planning remains offline by default; set `constraints={"require_operational_providers": True}` when a dry-run should also preflight live keys.

Model listings separate what the provider exposes from what Crupier recommends for default routing:

```bash
crupier models list --all
crupier models list --all --recommended
crupier models list --all --status specialized
crupier models show ollama:glm-5.2
```

Every capability card carries a decision profile with `routing_status`, `lifecycle`, `production_default`, `requires_opt_in`, task skills, modality support, and source evidence. Expensive or narrow models can remain visible without being selected by default. For example, OpenAI `o3`, `o3-pro`, and `o4-mini` are treated as explicit opt-in models rather than Crupier production-default choices.

For `0.3.0`, Crupier treats the provider catalog and the automatic routing set as different things. Provider discovery may produce hundreds of cards, but the production-default set stays intentionally small and source-backed: current OpenAI GPT defaults, current Claude Opus/Sonnet defaults, current Gemini Flash/Pro defaults, and selected Ollama Cloud defaults such as `ollama:glm-5.2` and `ollama:gpt-oss:120b`. Everything else remains selectable by the project owner through `[models].allow`, but is classified as `unknown`, `opt_in`, `specialized`, `legacy`, `deprecated`, or `shutdown` until there is enough evidence to promote it.

Refresh reports now separate added, removed, stale, pricing, and profile/capability changes so maintainers can review what changed before updating an allowlist.

## How Model Selection Works

Crupier selection is intentionally layered:

1. Load project policy, profiles, allowlist, denylist, and capability cards.
2. Classify the request with weighted signals: agentic, structured, fast, cheap, research, private, multimodal, file-based, embedding, tool-using, or constrained.
3. Filter out models that violate policy, stability rules, provider config, required capabilities, or budget constraints.
4. Score the remaining models using configurable weights for profile preferences, task signals, quality/cost/latency tiers, verified probes, eval results, budget fit, and human feedback.
5. Build a `RoutePlan` with a strategy such as single, fallback, cascade, panel, fusion, critique-repair, local-first, or delegate.
6. Validate the route shape before any provider call.
7. Execute the route, or return the plan when `dry_run=True`.

The default orchestrator is deterministic. You can opt into a model orchestrator for JSON route plans, one repair attempt, and deterministic fallback:

```toml
[orchestrator]
mode = "model"
model = "ollama:glm-5.2"
fallback_model = "anthropic:claude-opus-4-8"
fallback = "deterministic"
temperature = 0
require_validated_plan = true
max_repairs = 1
allow_prompt_summary_only = true
```

Configure it from the CLI or SDK:

```bash
crupier orchestrator show
crupier orchestrator set --model ollama:glm-5.2 --fallback-model anthropic:claude-opus-4-8
crupier orchestrator set --model openai:gpt-5.4-mini
```

```python
from crupier import Crupier

crupier = Crupier.from_project()
crupier.configure_orchestrator(
    mode="model",
    model="ollama:glm-5.2",
    fallback_model="anthropic:claude-opus-4-8",
    persist=True,
)
```

`ollama:glm-5.2` is a strong preset when Ollama Cloud is enabled for the project, not a lock-in. Any model visible to your enabled provider accounts can be used as the orchestrator model by setting `provider:model`.

Sensitive agentic/tool routes include guardrails so an orchestrator cannot silently downgrade into an unsafe or unsupported model path.

Scoring is project-configurable:

```toml
[scoring]
task_signal_weight = 2
profile_preference_weight = 3
local_eval_weight = 1
human_feedback_weight = 1
budget_over_penalty = -30
```

Use project evals and feedback to suggest conservative scoring updates:

```bash
crupier scoring suggest
crupier scoring suggest --apply
```

## Profiles And Strategies

Profiles are named routing intents:

```toml
[profiles.agentic]
prefer = ["tool_use", "coding", "long_horizon", "reliability"]
strategy = "orchestrated"

[profiles.cheap]
prefer = ["low_cost"]
strategy = "cascade"

[profiles.fast]
prefer = ["low_latency"]
strategy = "single"

[profiles.private]
prefer = ["local", "zdr", "no_prompt_logging"]
strategy = "local_first"

[profiles.research]
prefer = ["consensus", "critique"]
strategy = "fusion"

[profiles.structured]
prefer = ["structured_output", "schema_validity"]
strategy = "cascade"
```

Profiles can also live in `.crupier/profiles/*.toml` or `.json`, which lets teams share routing presets without editing the main `crupier.toml`. Advanced profiles can declare `strategy_rules` so, for example, a short tool request stays `single` while a longer high-risk tool workflow uses `critique_repair` or `delegate`.

Use a profile from Python:

```python
result = crupier.deal(
    task="Review this agent plan and identify risks.",
    input={"plan": "..."},
    mode="research",
    dry_run=True,
)
```

Or from the CLI:

```bash
crupier route "Extract invoice fields" --mode structured
crupier route "Answer this in one sentence" --mode fast
crupier route "Compare two implementation plans" --mode research
```

## Capability Probes And Readiness

Capability metadata starts as inferred provider/model information. Probes turn that into verified evidence:

```bash
crupier capabilities probe --provider openai
crupier capabilities probe --model openai:gpt-5.4-mini
crupier capabilities probe --provider openai --apply
crupier capabilities probe --provider openai --probe structured_output --probe tool_call --probe streaming --apply
crupier capabilities probe --model openai:text-embedding-3-small --probe embeddings --apply
```

Verified probe evidence wins over family inference. If a model family looks tool-capable but `tool_call` fails for the configured account, Crupier records that failed capability and keeps tool routes away from that model unless the evidence changes.

Available probes:

- `text_basic`
- `json_instruction`
- `max_output_param`
- `structured_output`
- `tool_call`
- `streaming`
- `embeddings`

Check readiness:

```bash
crupier capabilities readiness
crupier capabilities readiness --strict
crupier capabilities readiness --provider openai --json
```

Routing can require verified capability evidence:

```python
result = crupier.deal(
    task="Return strict JSON.",
    response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    constraints={"require_verified_capabilities": True},
    dry_run=True,
)
```

Ollama Cloud and local Ollama are not treated as "all models are embedding models." Crupier marks dedicated embedding models by name and confirms them with `--probe embeddings`; chat models stay chat models unless real probes prove otherwise.

## Real Provider Verification

Check config, environment variables, discovery, capability readiness, and real smoke calls:

```bash
crupier verify
crupier verify --provider openai --provider anthropic --provider ollama
crupier verify --provider anthropic --provider google --provider ollama
crupier verify --json
```

OpenAI is included as a baseline by default when verifying other providers. Use `--no-openai-baseline` only when you deliberately want to skip it.

Run minimal real calls against selected models:

```bash
crupier smoke --provider openai
crupier smoke --provider anthropic
crupier smoke --provider ollama
crupier smoke --model openai:gpt-5.4-mini
```

By default, smoke tests do not print model output. Use `--show-output` only for harmless debugging prompts.

## Structured Output

Python:

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "total": {"type": "number"},
    },
    "required": ["name", "total"],
    "additionalProperties": False,
}

result = crupier.deal(
    task="Extract invoice data.",
    input="Invoice for Ada, total 12.50",
    response_schema=schema,
    dry_run=False,
)

print(result.output_json)
```

Crupier passes native JSON-schema constraints to providers that support them in the active adapter, including OpenAI/OpenRouter Responses, Google Gemini, and Ollama REST. It still validates the parsed JSON locally and runs one repair attempt when a provider returns invalid output.

CLI:

```bash
crupier deal "Extract invoice data" \
  --input "Invoice for Ada, total 12.50" \
  --response-schema '{"type":"object","properties":{"name":{"type":"string"},"total":{"type":"number"}},"required":["name","total"]}' \
  --no-dry-run
```

Crupier validates returned JSON locally and makes one repair attempt with the same model before failing with `CrupierStructuredOutputError`.

## Local Tools

Python callables can be passed as local tools:

```python
def lookup_user(name: str):
    """Look up a user in the project database."""
    return {"name": name, "id": "usr_123"}

result = crupier.deal(
    task="Find Ada and summarize the result.",
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
    task="Write the changelog.",
    tools=[tool],
    constraints={"approved_tools": ["write_file"]},
    dry_run=False,
)
```

Tool execution is provider-agnostic: Crupier asks the selected model for a JSON tool plan, executes approved local tools, deduplicates identical tool calls with idempotency keys, and sends tool results back for the final answer. Provider-native tool-call execution is planned as an optimization.

Tool workflows can re-plan for multiple rounds with `max_tool_rounds`, so a model can call one safe local tool, inspect the result, then call a second tool before producing the final answer.

## Files, Images, PDFs, And Multimodal Requests

Plan a file route:

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
    dry_run=True,
)

print(result.route.input_plan)
```

Current execution behavior:

- Images can route to native vision-capable models and execute through OpenAI, Anthropic Claude, Google Gemini, and Ollama adapters when the selected model supports image input.
- Text, Markdown, JSON, YAML, HTML/CSS, code files, and PDFs can execute as extracted text context.
- PDF extraction uses `pypdf` from `crupier[pdf]` when installed or a local `pdftotext` binary when available.
- `constraints={"require_native_file_input": True}` forces a native file-capable route instead of extracted context.
- Native PDF/audio/video/office-document execution, OCR, table-aware PDF extraction, and transcription are explicitly blocked until those local or provider-native pipelines are implemented.

## Embeddings

Crupier marks embedding models separately from chat models. For projects that already use OpenAI-style embedding clients, the compatibility layer can route embedding calls through the same Crupier boundary:

```bash
crupier capabilities probe --model openai:text-embedding-3-small --probe embeddings --apply
```

OpenAI-compatible client:

```python
from crupier.compat.openai import OpenAI

client = OpenAI(project=".")
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Search text",
)

print(response.data[0].embedding[:3])
```

## Drop-In Adoption For Existing Projects

Crupier adoption is not limited to OpenAI-style apps. The native SDK path is the general route for any AI app or agent stack; the OpenAI-compatible paths are convenience routes for projects that already expose an OpenAI-like HTTP/client boundary.

Crupier supports four adoption paths:

| Path | Use when |
| --- | --- |
| `native_sdk` | Any app, agent, backend, notebook, or integration can wrap its AI boundary with `Crupier.from_project().deal(...)`. |
| `proxy` | The app can point an OpenAI-compatible SDK or HTTP client at a local base URL. |
| `compat_client` | Python code imports OpenAI client classes in a narrow boundary and can switch that import. |
| `autopatch` | You want a controlled OpenAI-client experiment without editing many imports. |

Ask Crupier to inspect a repo and recommend a path:

```bash
crupier adopt plan
crupier adopt plan src app.py --write-report
crupier adopt patches --path recommended --write-report
crupier adopt doctor
```

For a freshly cloned repo, these can run before `crupier init`:

```bash
crupier adopt plan
crupier adopt patches --path recommended --write-report
crupier adopt package
crupier adopt handoff --write-report
```

`adopt package` writes a reviewer bundle under `.crupier/packages/`: code comments, patch suggestions, SARIF, review-comment packets, doctor reports, and handoff notes. It is non-destructive.

Production-oriented gates:

```bash
crupier adopt doctor --production --real --provider openai --provider anthropic --provider ollama
crupier adopt signoff --verdict approve --handoff .crupier/handoffs/adoption_handoff_YYYYMMDDTHHMMSSZ.md
crupier adopt handoff --production --real --provider openai --provider anthropic --provider ollama --write-report
```

The doctor separates technical readiness from human approval. A technically passing route is not automatically approved while human review gates remain open.

## Optional OpenAI-Compatible Client And Server

This is one adoption surface, not the whole product. Use it when existing code already resembles OpenAI SDK usage:

```python
from crupier.compat.openai import OpenAI

client = OpenAI(project=".")
response = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Summarize this"}],
)

print(response.choices[0].message.content)
```

Opt-in monkeypatch for controlled experiments:

```python
import crupier

crupier.install("openai")
```

Run a local OpenAI-compatible HTTP server:

```bash
crupier serve --port 8787
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
```

`crupier serve` binds to `127.0.0.1` by default. Non-loopback binds such as `0.0.0.0` require `--allow-remote`, and should only be used behind your own network/auth boundary. Browser CORS is disabled by default; opt in with `--cors-origin http://localhost:3000` for trusted local browser experiments.

Implemented endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

The server returns OpenAI-like JSON errors, `x-request-id`, typed Responses SSE events, and Chat Completions chunks. Add `--no-dry-run` when you want the proxy to call real providers.

## Provider Retries And Fallback

Crupier separates per-call retries from route fallback:

```toml
[routing]
max_provider_retries = 1
retry_backoff_seconds = 0.2
retry_jitter_seconds = 0
circuit_breaker_failure_threshold = 3
circuit_breaker_cooldown_seconds = 60
require_operational_providers = true
```

Provider calls retry transient rate-limit and provider-unavailable errors before giving up or moving to the next fallback model. Crupier does not retry auth failures, missing optional dependencies, policy/budget failures, model-capability failures, non-transient provider setup errors, or tool approval blocks. Each failed provider attempt is recorded in `trace.errors` with provider, model, attempt, latency, error type, and retryability. Successful calls include the final `attempt` number in `trace.provider_calls`.

Per request, set `constraints={"max_provider_retries": 0}` to disable provider retries, `constraints={"retry_backoff_seconds": 0}` for zero-wait test runs, or `constraints={"require_operational_providers": False}` for offline route simulation. Circuit breakers mark repeatedly failing providers as degraded; when another provider remains available, degraded providers are removed before route selection. `constraints={"timeout_seconds": 30}` applies to provider calls where the adapter/SDK exposes request-level timeouts, including OpenAI, OpenRouter, Anthropic Claude, and Ollama. For Google Gemini, configure a client timeout in `[providers.google]` with `timeout_seconds = 30`.

## Evals And Human Feedback

Run routing evals:

```bash
crupier eval run
crupier eval run --orchestrator-mode hybrid
crupier eval run --dataset examples/routing-eval.json --write-report
crupier eval compare "Answer this support ticket" --model openai:gpt-5.4-mini --model anthropic:claude-opus-4-8
crupier eval compare-dataset --dataset examples/model-compare-eval.json --model openai:gpt-5.4-mini --model anthropic:claude-opus-4-8
```

Compare variants with real provider calls and small budgets:

```bash
crupier eval compare "Reply exactly crupier-ok" \
  --model openai:gpt-5.4-mini \
  --model anthropic:claude-opus-4-8 \
  --expect-contains crupier-ok \
  --max-cost-usd 0.02 \
  --no-dry-run
```

Record human judgement when a technically valid route is not good enough:

```bash
crupier feedback record \
  --model openai:gpt-5.4-mini \
  --mode agentic \
  --rating 2 \
  --verdict needs_work \
  --tag weak_code_review

crupier feedback summary
crupier feedback apply
crupier scoring suggest --apply
```

Feedback records route metadata, ratings, verdicts, tags, and redacted notes. It does not need to store prompts or responses.

## Trace Storage And Privacy Defaults

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

## CLI Map

| Command | Purpose |
| --- | --- |
| `crupier init` | Create `crupier.toml`, `.env.example`, `.gitignore` entries, and `.crupier/` directories. |
| `crupier models discover` | List models available to enabled providers. |
| `crupier models allow` | Update the project allowlist. |
| `crupier update --online` | Refresh local capability cards from provider discovery. |
| `crupier models refresh` | Alias for online provider refresh with added model/profile/pricing change reporting. |
| `crupier route` | Show a route decision without provider execution. |
| `crupier deal` | Route and optionally execute a task. |
| `crupier verify` | Check provider config, discovery, readiness, and real smoke calls. |
| `crupier smoke` | Run minimal real provider calls. |
| `crupier capabilities probe` | Verify model capabilities and optionally persist evidence. |
| `crupier eval` | Run routing evals and model comparisons. |
| `crupier feedback` | Record and apply human feedback signals. |
| `crupier adopt` | Inspect existing projects and produce adoption reports. |
| `crupier code comments` | Generate AI-integration review comments. |
| `crupier trace` | Inspect, delete, or replay stored traces. |
| `crupier serve` | Run an OpenAI-compatible local HTTP server. |
| `crupier release check` | Validate package readiness for release. |
| `crupier --env-file .env ...` | Load missing provider keys from a local ignored env file for real-provider commands. |

## Release Readiness

During development, run the fast local checks:

```bash
python -m pytest
python -m ruff check src tests --select E9,F63,F7,F82
python -m pip_audit --skip-editable --progress-spinner off
crupier release check
crupier release check --json
```

Use `crupier release check --skip-build` only for quick inner-loop diagnostics. Do not use it as publication evidence.

Before publishing or handing a package build to another project, run the full gates:

```bash
crupier release check --strict-public
crupier release check --strict-public --verify-project-urls --check-pypi-name
crupier capabilities probe --provider openai --apply
crupier capabilities probe --provider anthropic --apply
crupier capabilities probe --provider google --apply
crupier capabilities probe --provider ollama --apply
crupier release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama
crupier --version
```

The release check validates:

- package metadata, Trove classifiers for the tested Python versions, final public version, and version sync
- README, PyPI-safe README links, CONTRIBUTING, SECURITY, CHANGELOG, public Markdown link/fence health, GitHub YAML syntax, and community templates
- repository `.gitignore` coverage for local keys, caches, builds, and generated Crupier artifacts
- safe `crupier init` defaults, including Ollama Cloud and prompt/response storage opt-in
- CI, Dependabot, critical Ruff lint, and `pip-audit` wiring
- provider extras, console script, license metadata, and `py.typed`
- sdist and wheel build
- built wheel/sdist PyPI metadata for name, version, summary, Python requirement, license, project URLs, classifiers, and extras
- tracked-file secret pattern scanning plus artifact content inspection for secret/cache leaks
- public examples from the built sdist run offline without provider keys
- `twine check`
- wheel and sdist install/import/public-export/CLI and `python -m crupier` help/version/init/route/Python SDK smoke tests in clean virtual environments
- optional PyPI name availability
- optional public project URL reachability
- optional real provider readiness and smoke calls

If the provider gate reports `needs_probes`, apply capability probes for the
affected allowlist models, then rerun the provider gate. If a smoke call fails,
first resolve provider account, quota, permission, or regional availability
issues; if the model itself is unavailable or unsuitable for the project, remove
or replace it in `[models].allow` before publishing.

Final public release order:

1. Keep the repository private while preparing the single release commit.
2. Configure PyPI trusted publishing for this repository and the `pypi` GitHub environment.
3. Change repository visibility to public.
4. Rerun `crupier release check --strict-public --verify-project-urls --check-pypi-name`.
5. Rerun `crupier release check --strict-public --verify-providers --provider openai --provider anthropic --provider google --provider ollama`.
6. Confirm Dependabot security updates are enabled and unpaused.
7. Protect `main` with required CI, no force pushes, and pull-request review before accepting public changes.
8. Publish a final GitHub Release from the current `main` tip tagged `v0.3.0` or `0.3.0`; the publish workflow rejects draft/non-final releases, non-main targets, commits that do not match `origin/main`, and tags that do not match the package version.

Manual workflow dispatch is only for retrying an intentional release operation.
It must run from `main`, requires the `version` input to equal `0.3.0`, and
requires `confirm_publish=true` before any distribution is built or uploaded.
The publish workflow verifies that the configured PyPI project is available for
first uploads or already owned for maintenance releases, then requires the
package version, workflow input, release tag, and checked-out `main` commit to
match before building distributions.
Publish attempts are serialized per ref so duplicate release/manual triggers do
not race each other. PyPI publishing uses job-scoped `id-token: write`
permissions and links the `pypi` environment to the package page.

For development and pull request expectations, see [CONTRIBUTING.md](https://github.com/686f6c61/crupier/blob/main/CONTRIBUTING.md).

## What Is Implemented In 0.3.0

Implemented now:

- Python SDK: `Crupier.from_project()`, `from_toml()`, `from_config()`, `deal()`, `adeal()`, basic `stream()`
- CLI for init, update, models, registry snapshots, capabilities, profiles, route, deal, smoke, verify, eval, feedback, audit, adoption, trace, server, and release checks
- Real provider calls for OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud
- OpenRouter as an optional disabled-by-default BYOK OpenAI-compatible adapter
- deterministic route planning and opt-in model orchestrator with fallback orchestrator support
- configurable scoring weights plus `scoring suggest` from eval and human-feedback evidence
- weighted task-signal classification for model selection explanations
- fallback, cascade, panel, fusion, critique-repair, local-first, and delegate routes
- cascade validation/escalation, parallel panel/fusion execution, iterative tool planning, and bounded delegate sub-routes
- structured-output validation and one repair attempt
- provider-agnostic local tool execution with approval guardrails and idempotency
- multimodal/file planning, native image execution for supported adapters, and extracted text/PDF context
- explicit unsupported errors for OCR, audio/video transcription, spreadsheets, office documents, and native non-image execution paths
- model discovery, capability cards, provider refresh, capability/profile/pricing change reports, probes, readiness checks, and registry snapshots
- eval runner, compare reports, human review packets, human decision templates, and feedback application
- adoption audit, doctor, package, handoff, code comments, SARIF, and signoff workflows
- opt-in metadata traces with redaction and replay only when prompt/input storage is explicitly enabled
- OpenAI-like Python client, optional autopatch, and local HTTP server
- declarative policy rules and shared `.crupier/profiles/` routing presets
- provider retries with jitter, circuit breakers, and route-time degraded-provider exclusion
- typed errors and `py.typed`
- release gate with build, artifact, install smoke, PyPI name, project URL, provider readiness, CI, security, and dependency checks

Planned after `0.3.0`:

- production-calibrated model orchestrator evals
- larger production eval datasets
- provider-native structured-output parameter mapping beyond prompt+validate execution
- provider-native PDF/audio/video/document execution
- table-aware PDF extraction, OCR, audio/video transcription, and office-document parsing
- broader SDK compatibility matrix
- provider-native tool-calling optimizations
- provider-native streaming proxy

## License

MIT. See [LICENSE](https://github.com/686f6c61/crupier/blob/main/LICENSE).

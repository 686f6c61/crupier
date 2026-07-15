# Crupier

Crupier is a Python orchestration SDK for AI applications and agents that need to choose, combine, audit, and improve model routes across providers.

It is designed for two situations:

- New projects that want one AI boundary instead of hard-coding provider/model choices throughout the codebase.
- Existing AI projects, agents, SDK integrations, or provider-specific codebases that want to add model selection, provider fallback, audits, evals, and human review without a full rewrite.

Crupier is a BYOK orchestration layer: it runs with your own provider accounts across OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, configurable OpenAI-compatible inference servers, optional OpenRouter BYOK, or your own integration boundary. It keeps prompts/responses out of persistent logs by default and routes each request toward the best available model or model family for the task, quality target, latency, cost budget, and project policy.

Current public package version: `0.4.0`.

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
pip install "crupier[inference-server]"
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
| `crupier[inference-server]` | Configurable OpenAI-compatible chat, image-input, and embedding servers. |
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
INFERENCE_API_KEY=
```

Do not pass API keys as CLI arguments, commit `.env`, or put provider keys in `crupier.toml`.
For CLI checks, load a local ignored env file explicitly:

```bash
crupier --env-file .env verify --provider google
crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama --provider inference
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

[providers.inference]
enabled = false
mode = "openai_compatible"
host = "http://127.0.0.1:8000/v1"
auth = "none"
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
crupier models discover --provider inference
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

For `0.4.0`, Crupier treats the provider catalog and the automatic routing set as different things. Provider discovery may produce hundreds of cards, but the production-default set stays intentionally small and source-backed: current OpenAI GPT defaults, current Claude Opus/Sonnet defaults, current Gemini Flash/Pro defaults, and selected Ollama Cloud defaults such as `ollama:glm-5.2` and `ollama:gpt-oss:120b`. Models from a configurable inference server remain selectable by the project owner through `[models].allow`, but are classified as `unknown`, `opt_in`, `specialized`, `legacy`, `deprecated`, or `shutdown` until project probes and eval evidence justify promotion.

Refresh reports now separate added, removed, stale, pricing, and profile/capability changes so maintainers can review what changed before updating an allowlist.

## How Model Selection Works

Crupier selection is intentionally layered:

1. Load project policy, profiles, allowlist, denylist, and capability cards.
2. Classify the request with weighted signals: agentic, structured, fast, cheap, research, private, multimodal, file-based, embedding, tool-using, or constrained.
3. Filter out models that violate policy, stability rules, provider config, adapter transport support, required capabilities, or budget constraints.
4. Score the remaining models using configurable weights for profile preferences, task signals, quality/cost/latency tiers, verified probes, eval results, budget fit, and human feedback.
5. Build a `RoutePlan` with a strategy such as single, fallback, cascade, panel, fusion, critique-repair, local-first, or delegate.
6. Validate the route shape before any provider call.
7. Execute the route under one shared call/cost/latency budget that includes model-powered planning, retries, tools, panels, and delegated work; or return the plan when `dry_run=True`.

Projects created with `crupier init` default to a model-powered orchestrator for JSON route plans, one repair attempt, and deterministic fallback. The orchestrator receives deterministic scoring as a prior and cannot bypass policy or route validation. Set `mode = "deterministic"` when a project explicitly wants zero planning-model calls:

```toml
[orchestrator]
mode = "model"
model = "ollama:glm-5.2"
fallback_model = "anthropic:claude-opus-4-8"
fallback = "deterministic"
temperature = 0
require_validated_plan = true
max_repairs = 1
candidate_limit = 6
allow_prompt_summary_only = false
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

The model orchestrator receives bounded, redacted request content plus a compact, provider-diverse candidate pool: natural-language strengths and avoid-cases, modalities, context/output limits, tool and structured-output support, reasoning controls, pricing evidence, probe status, and deterministic scores as a calibrated prior. `candidate_limit` bounds that pool to control planning latency and cost. Its JSON plan is schema-validated and policy-checked before execution. Set `allow_prompt_summary_only = true` only when the orchestrator must choose from task metadata without seeing request content.

`force_model` is an explicit caller decision, so Crupier validates it and bypasses the model-orchestrator call. This avoids paying for a redundant routing decision and keeps a one-call budget genuinely to one provider call.

Crupier also bypasses model-powered planning when policy and capability filters leave exactly one candidate. The LLM orchestrator is reserved for real choices between viable routes.

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
strategy = "orchestrated"

[profiles.fast]
prefer = ["low_latency"]
strategy = "orchestrated"

[profiles.private]
prefer = ["local", "zdr", "no_prompt_logging"]
strategy = "local_first"

[profiles.research]
prefer = ["consensus", "critique"]
strategy = "orchestrated"

[profiles.structured]
prefer = ["structured_output", "schema_validity"]
strategy = "orchestrated"
```

`strategy = "orchestrated"` lets the configured LLM choose the route strategy from the request and candidate evidence. Setting `single`, `cascade`, `fusion`, or another concrete strategy turns it into a project policy that the model orchestrator must obey. Profiles can also live in `.crupier/profiles/*.toml` or `.json`, which lets teams share routing presets without editing the main `crupier.toml`. Advanced profiles can declare `strategy_rules` so, for example, a short tool request stays `single` while a longer high-risk tool workflow uses `critique_repair` or `delegate`.

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

## Live End-To-End Routing Validation

`verify` and `smoke` prove that provider credentials, discovery, and one minimal model call work. They do not prove that the LLM orchestrator can inspect a real task, choose a strategy, divide work between roles, execute those roles across providers, and preserve the result in a trace. The live validation suite exercises that complete path without `force_model`:

```bash
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  -u GOOGLE_API_KEY -u GEMINI_API_KEY \
  -u OLLAMA_API_KEY -u OLLAMA_HOST \
  python examples/live_routing_validation.py --real --project . --write-report
```

Use `--case fast`, `--case research`, or repeat `--case` to run only selected cases. The optional report is written to the ignored local path `.crupier/evals/live-routing-validation.json`. It contains the task, validated route, role/model allocation, provider-call metadata, latency, cost, checks, and a bounded output preview; it does not contain credentials.

Each validation case starts from a fresh Crupier runtime so a circuit opened by one experiment cannot invalidate an unrelated later case. Cross-request circuit persistence and degraded-provider exclusion are covered separately by the automated executor suite.

A provider or primary-orchestrator error is not silently discarded. The case passes only if the error is recovered by a later validated orchestrator plan, successful retry, role fallback, or fusion quorum; executor errors or an invalid final route still fail the suite. Recovered events remain in `trace.errors` and `trace.fallbacks` so fallback behavior is auditable.

The suite sends the following literal task strings and controlled inputs; these are the prompts used by the executable validation case definitions, not paraphrases:

1. `fast`: `Summarize the deployment event in exactly one sentence and include the incident id.` Input identifies `INC-42` and a canary rollback.
2. `structured`: `Extract claim fields from the supplied text; do not infer missing facts. Use a primary extraction, validate it against the response schema, and reserve a separate escalation model only if validation fails.` Input contains claim `CLM-2048`, its date, total, missing police report, and mandatory human review. The response must match an exact JSON Schema.
3. `research`: `Compare a single frontier model with provider fallback against a capability-aware multi-provider router for 100000 support tickets per month. Evaluate reliability, latency, cost, observability, failure modes, and migration risk. Obtain three independent provider-diverse analyses, have a separate judge reconcile consensus and disagreements, then use a final writer to recommend one.`
4. `agentic`: `Review a production payment-retry change. Produce a merge decision, identify rollback risks, challenge the first draft with an independent critic, and repair the recommendation.` Input includes the proposed retry policy, existing idempotency behavior, and missing tests.
5. `tools`: `Use lookup_billing_case for ticket SUP-LIVE-TOOL-1, then draft a concise reply. Do not claim a refund started or completed unless the tool says so. State the next action and ETA. Have an independent critic verify the draft against the tool result, then repair any unsupported claim.` The local tool returns an authoritative duplicate-charge state, `refund_status=not_started`, billing review, and a two-business-day ETA.
6. `delegate`: `Use delegate exactly once. Hand off this bounded subtask: identify three failure modes in a capability-aware router and one mitigation for each. Execute the subtask as single-model analysis.`
7. `image`: `Inspect the attached solid-color image and reply with exactly one lowercase color word.` The fixture is a generated solid red PNG sent through native vision.
8. `pdf`: `Read the attached PDF and reply with only the audit passphrase.` The generated PDF contains `The audit passphrase is zircon.` and is sent through native PDF input.

Each case asserts the work decomposition as well as the final answer:

| Case | Required route and subtasks | Success evidence |
| --- | --- | --- |
| `fast` | `single`: one primary answer | Valid model-authored plan, one sentence, `INC-42` preserved |
| `structured` | `cascade`: primary attempt, sufficiency/schema validation, conditional escalation | Exact typed JSON; escalation is skipped only when the primary output validates |
| `research` | `fusion`: three provider-diverse panel analyses, judge, final writer | Three panel members are planned; at least two providers must succeed to satisfy quorum before judge and writer execute |
| `agentic` | `critique_repair`: generator, independent critic, repair | All three roles execute under one shared budget |
| `tools` | Tool planner, approved local tool execution, tool-aware critic, repaired final answer | Tool completes; critic and repair receive the tool ledger; refund state and ETA remain factual |
| `delegate` | Outer delegate creates a bounded subtask; nested Crupier route selects and executes its own model | Nested strategy/models are traced and remaining depth decreases |
| `image` | Capability filtering plus native image delivery to a vision model | Adapter records one native image and output is exactly `red` |
| `pdf` | Capability filtering plus native PDF delivery to a document-capable model | Adapter records one native file and output is exactly `zircon` |

For multi-step routes, the executor turns the validated role plan into these bounded subtask prompts:

| Role | Subtask sent to that role |
| --- | --- |
| Fusion panel member | Receives the original task and controlled input independently; panel members do not see each other's drafts |
| Fusion judge | Receives every labeled panel output and must return consensus, contradictions, gaps, and risks without hidden reasoning |
| Fusion final writer | Receives the original request plus the judge synthesis and writes the direct user-facing answer with uncertainty stated |
| Critique-repair generator | Produces the first complete answer from the original request |
| Critique-repair critic | Receives the draft and checks correctness, missing constraints, cost/latency tradeoffs, and tool risk |
| Critique-repair repair | Receives both draft and critique and produces the corrected final answer |
| Tool planner | Receives the original task plus JSON tool schemas, proposes approved calls, observes the authoritative result ledger, and may plan another bounded round |
| Tool critic | Checks the draft against the actual tool ledger rather than relying on model memory |
| Tool repair | Preserves only facts supported by the request and tool ledger, then applies the critic's corrections |
| Delegate | Receives the bounded subtask in `RouteStep.params.task`; a nested Crupier call inherits the remaining call, cost, latency, and depth budget and selects its own validated subroute |

In the observed tool plan, the orchestrator made the subtasks explicit: the generator had to call `lookup_billing_case` and draft the reply; the critic had to verify that refund status, next action, and ETA were supported; the repair role had to remove unsupported refund claims or fill missing action/ETA. The delegate subtask was exactly `identify three failure modes in a capability-aware router and one mitigation for each`.

The following allocation was observed with the configured operational allowlist on `2026-07-15`:

| Case | Orchestrator decision | Executed model allocation |
| --- | --- | --- |
| `fast` | `ollama:glm-5.2` selected `single` | `google:gemini-3.5-flash` as primary |
| `structured` | `ollama:glm-5.2` selected `cascade` | `anthropic:claude-sonnet-4-6` as primary; `anthropic:claude-opus-4-8` reserved for escalation and not called after exact schema validation |
| `research` | `ollama:glm-5.2` selected `fusion` | Planned panel: `google:gemini-3.5-flash`, `openai:gpt-5.5`, and `anthropic:claude-sonnet-4-6`; Gemini and Claude succeeded, GPT-5.5 returned empty text twice, and the `2/3` quorum continued to `anthropic:claude-opus-4-8` as judge and `openai:gpt-5.5` as final writer |
| `agentic` | `ollama:glm-5.2` selected `critique_repair` | Generator: `anthropic:claude-opus-4-8`; critic: `openai:gpt-5.5`; repair: `anthropic:claude-opus-4-8` |
| `tools` | `ollama:glm-5.2` selected a tool-aware `critique_repair` route | Tool planner: `anthropic:claude-opus-4-8`; local tool; tool critic: `anthropic:claude-sonnet-4-6`; tool repair: `anthropic:claude-opus-4-8` |
| `delegate` | Requested `delegate`; `ollama:glm-5.2` authored the outer plan, bounded subtask, and `anthropic:claude-opus-4-8` anchor | Nested `ollama:glm-5.2` routing plan; `anthropic:claude-opus-4-8` as nested primary |
| `image` | `ollama:glm-5.2` selected a native-vision `single` route | `google:gemini-3.5-flash` received the PNG |
| `pdf` | `ollama:glm-5.2` selected a native-PDF `single` route | `openai:gpt-5.4-mini` received the PDF |

No case fixes an executor model. Seven cases let the model orchestrator choose both strategy and models; `delegate` fixes only the strategy because that case specifically validates recursive delegation. Exact model names may change as a project's allowlist, account availability, prices, probes, evals, and feedback change. The stable contract is the validated strategy, required roles, capability fit, budget enforcement, and trace evidence.

The repeated live run on `2026-07-15` passed `8/8` cases with no unrecovered trace errors. Its eight sequential routes used about `189.9 s` in aggregate and recorded an estimated provider cost of `$0.4234`; these are trace estimates, not provider invoices. The failed GPT-5.5 panel attempts remain cost-visible in the trace even though fusion recovered through quorum.

The complete executable cases live in [`examples/live_routing_validation.py`](https://github.com/686f6c61/crupier/blob/main/examples/live_routing_validation.py). Model-facing route and role prompts are versioned in [`src/crupier/prompts.py`](https://github.com/686f6c61/crupier/blob/main/src/crupier/prompts.py) instead of being duplicated in prose. The route prompt is `orchestrator.route_plan.v3`; it supplies the allowed strategies, exact role shape for each strategy, bounded request context, constraints, and a compact candidate-card corpus. Fusion plans prefer a provider-diverse three-member panel when candidates and budget allow, while `min_panel_size` and `max_panel_size` are enforced as hard request policy. A returned plan must pass schema, policy, capability, and budget validation before execution; malformed plans get one contract-aware repair attempt and then the configured fallback.

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
- OpenAI-compatible inference servers can receive native images when the selected model card and configured adapter both declare image support.
- Text, Markdown, JSON, YAML, HTML/CSS, code files, and PDFs can execute as extracted text context.
- PDF extraction uses `pypdf` from `crupier[pdf]` when installed or a local `pdftotext` binary when available.
- OpenAI Responses can receive native PDFs when `constraints={"require_native_file_input": True}` or `constraints={"file_strategy": "native"}` is set.
- The CLI exposes `--file-strategy auto|native|extract` on `crupier deal`.
- Crupier checks both model capability and adapter transport support before selecting a native-file route, so an unsupported adapter cannot silently drop a file.
- Native video/office-document execution, OCR, table-aware PDF extraction, and transcript-first audio/video preprocessing remain explicitly unsupported.

## Embeddings And Specialized Operations

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

Crupier can route dedicated embedding, reranking, transcription, text-to-speech, and image generation/editing models through capability-specific operations instead of pretending they are chat models. Built-in providers and custom adapters declare which operations they actually execute:

```python
from crupier import Crupier

crupier = Crupier.from_project()

ranking = crupier.rerank(
    query="capital of France",
    documents=["Berlin", "Paris"],
    top_n=1,
)
transcript = crupier.transcribe(file="meeting.wav")
speech = crupier.synthesize(input="Hola", voice="ef_dora")
image = crupier.generate_image(
    prompt="A precise product diagram on a white background",
    size="1024x1024",
)
```

Use `Crupier.run(...)` when the caller does not already know which operation is required. In `model` or `hybrid` orchestrator mode, Crupier first asks the configured orchestrator to classify the request as chat, embedding, reranking, transcription, TTS, or image generation, validates that choice against the executable allowlist, and then selects a capable model. The classifier and execution share one call/cost/latency budget and one decision trace:

```python
result = crupier.run(
    "Ordena estos textos por relevancia para la consulta",
    input={
        "query": "capital of France",
        "documents": ["Berlin", "Paris"],
    },
    constraints={"max_calls": 2},
    trace="debug",
)

print(result.operation)  # reranker
print(result.model)      # for example inference:rerank-model
```

Direct methods are preferable when application code already knows the operation: they avoid paying for classification while retaining policy, model selection, budgets, provider readiness, and traces. Specialized provider pricing may be quota-, request-, image-, or subscription-based instead of token-based; when the provider does not report billable cost, Crupier records the call and emits a warning but does not invent a USD estimate. Each adapter validates its own upload, dimension, output-count, and reference-image limits before dispatch.

### Live Operations And Compatibility Validation

The public operations suite validates the actual operation boundary, the Python compatibility client, and an ephemeral OpenAI-compatible HTTP server. It does not use fake adapters in real mode:

```bash
python examples/live_operations_validation.py \
  --real --project . --write-report
```

The project used for this command must allow at least one executable chat, embedding, reranking, transcription, TTS, and image-generation model. Those capabilities may come from built-in providers or a configured OpenAI-compatible inference server. Use `--case classifier`, `--case audio`, or repeat `--case` to narrow a run. The sanitized report is written to the ignored local path `.crupier/evals/live-operations-validation.json`; it records models, dimensions, byte counts, endpoint evidence, calls, budgets, and errors, but not vectors, audio, images, credentials, or raw provider responses.

The suite sends these exact controlled requests:

| Case | Request and input | Contract asserted |
| --- | --- | --- |
| `classifier` | `Rank these documents by relevance to the query.` with query `the exact token ZEBRA-991` and three documents | The model orchestrator classifies `reranker`, then a different operation step executes a capable model under the same budget and ranks the matching document first |
| `embeddings` | Embeds `Crupier live embedding` and `capability-aware model router` through every executable embedding model in the allowlist | Two non-zero vectors per model, equal positive dimensions, and requested dimensions honored where supported |
| `rerank` | Query `the exact token ZEBRA-991`; the second of three documents contains that token | Three descending scores and source index `1` ranked first |
| `audio` | Synthesizes `The secret phrase is blue ocean seven.` to WAV and sends those exact bytes to transcription | Valid RIFF audio and a transcript preserving `blue ocean` plus `seven` or `7` |
| `image` | Generates `A centered green circle on a white background, no text.` and edits a generated red fixture with `Change the red square to blue and keep the background white. No text.` | Generation and edit both return decodable image data; the edit trace records the edit endpoint |
| `compat` | Exact replies `COMPAT-OK`, `STREAM-OK`, and `NATIVE-STREAM-OK`, plus one embedding | Chat Completions shape, Responses event sequence, native route-event sequence, exact text reconstruction, and embedding shape |
| `http` | Repeats chat, streaming, embeddings, rerank, TTS/transcription, and image generation/edit through a loopback server, then sends an invalid chat request | `/health`, `/v1/models`, all operation endpoints, SSE `[DONE]`, request ids, binary response, multipart uploads, clean shutdown, and an OpenAI-shaped typed `400` error |

On `2026-07-15`, a clean-environment provider verification reported OpenAI, Anthropic Claude, Google Gemini, and Ollama Cloud ready, with `131`, `9`, `54`, and `18` models discovered respectively. Representative `structured_output`, `tool_call`, and `streaming` probes passed `12/12` across those four providers.

The repeated live operations run passed `7/7` cases. It produced two `128`-dimension vectors through each public embedding API and two `4096`-dimension vectors through the configured inference server; ranked the exact-token document first; produced a `136430`-byte WAV whose transcription preserved the test phrase; and returned image data for both generation and editing. The Python surface reconstructed all three exact compatibility replies. The HTTP surface returned healthy/model-list responses, a `32`-dimension embedding, the correct rerank index, audio and image bodies, and the expected typed `400` error with a request id.

`Crupier.stream()` currently emits route lifecycle events (`route_started`, `route_selected`, `final`). The OpenAI-compatible streaming surfaces reconstruct the expected SSE/Responses event contracts from the completed model result; they are not yet a provider-native token-forwarding proxy. Broader provider-native streaming remains listed under Planned Work.

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
- `POST /v1/rerank` and `POST /v2/rerank`
- `POST /v1/images/generations`
- `POST /v1/images/edits` (multipart)
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions` (multipart)

The server returns OpenAI-like JSON errors, `x-request-id`, typed Responses SSE events, Chat Completions chunks, and binary speech responses. Add `--no-dry-run` when you want the proxy to call real providers. Request bodies default to a 10 MB ceiling; raise it deliberately with `--max-request-bytes` for larger audio or image uploads, while still respecting the selected provider's own limits.

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
python -m pytest --cov=crupier --cov-fail-under=95
python -m ruff check src tests
python -m mypy src/crupier
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
crupier capabilities probe --provider inference --apply
crupier release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama --provider inference
crupier --version
```

The release check validates:

- package metadata, Trove classifiers for the tested Python versions, final public version, and version sync
- README, PyPI-safe README links, CONTRIBUTING, SECURITY, CHANGELOG, public Markdown link/fence health, GitHub YAML syntax, and community templates
- repository `.gitignore` coverage for local keys, caches, builds, and generated Crupier artifacts
- safe `crupier init` defaults, including Ollama Cloud and prompt/response storage opt-in
- CI, Dependabot, full Ruff lint, `mypy`, a 95% coverage floor, and `pip-audit` wiring
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
5. Rerun `crupier release check --strict-public --verify-providers --provider openai --provider anthropic --provider google --provider ollama --provider inference`.
6. Confirm Dependabot security updates are enabled and unpaused.
7. Protect `main` with required CI, no force pushes, and pull-request review before accepting public changes.
8. Publish a final GitHub Release from the current `main` tip tagged `v0.4.0` or `0.4.0`; the publish workflow rejects draft/non-final releases, non-main targets, commits that do not match `origin/main`, and tags that do not match the package version.

Manual workflow dispatch is only for retrying an intentional release operation.
It must run from `main`, requires the `version` input to equal `0.4.0`, and
requires `confirm_publish=true` before any distribution is built or uploaded.
The publish workflow verifies that the configured PyPI project is available for
first uploads or already owned for maintenance releases, then requires the
package version, workflow input, release tag, and checked-out `main` commit to
match before building distributions.
Publish attempts are serialized per ref so duplicate release/manual triggers do
not race each other. PyPI publishing uses job-scoped `id-token: write`
permissions and links the `pypi` environment to the package page.

For development and pull request expectations, see [CONTRIBUTING.md](https://github.com/686f6c61/crupier/blob/main/CONTRIBUTING.md).

## What Is Implemented In 0.4.0

Implemented now:

- Python SDK: `Crupier.from_project()`, `from_toml()`, `from_config()`, `deal()`, `adeal()`, basic `stream()`
- capability-aware operation SDK: `run()`, `embed()`, `rerank()`, `transcribe()`, `synthesize()`, `generate_image()`, and `edit_image()`
- CLI for init, update, models, registry snapshots, capabilities, profiles, route, deal, smoke, verify, eval, feedback, audit, adoption, trace, server, and release checks
- Real provider calls for OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, and configurable OpenAI-compatible inference servers
- OpenRouter as an optional disabled-by-default BYOK OpenAI-compatible adapter
- deterministic route planning and opt-in model orchestrator with fallback orchestrator support
- configurable scoring weights plus `scoring suggest` from eval and human-feedback evidence
- weighted task-signal classification for model selection explanations
- fallback, cascade, panel, fusion, critique-repair, local-first, and delegate routes
- cascade validation/escalation, parallel panel/fusion execution, iterative tool planning, and bounded delegate sub-routes
- structured-output validation and one repair attempt
- provider-agnostic local tool execution with approval guardrails and idempotency
- multimodal/file planning, adapter-aware native image execution, provider-specific native audio, OpenAI native PDF, and extracted text/PDF context
- explicit unsupported errors for OCR, transcript-first audio/video preprocessing, spreadsheets, office documents, and unimplemented native paths
- model discovery, capability cards, provider refresh, capability/profile/pricing change reports, probes, readiness checks, and registry snapshots
- model-powered operation classification with deterministic fallback and shared end-to-end budgets
- eval runner, compare reports, human review packets, human decision templates, and feedback application
- adoption audit, doctor, package, handoff, code comments, SARIF, and signoff workflows
- opt-in metadata traces with redaction and replay only when prompt/input storage is explicitly enabled
- OpenAI-like Python client, optional autopatch, and local HTTP server for chat, embeddings, reranking, images, speech, and transcription
- declarative policy rules and shared `.crupier/profiles/` routing presets
- provider retries with jitter, circuit breakers, and route-time degraded-provider exclusion
- typed errors and `py.typed`
- release gate with build, artifact, install smoke, PyPI name, project URL, provider readiness, CI, security, and dependency checks

Planned after `0.4.0`:

- production-calibrated model orchestrator evals
- larger production eval datasets
- provider-native structured-output parameter mapping beyond prompt+validate execution
- broader provider-native PDF/audio/video/document execution
- table-aware PDF extraction, OCR, transcript-first video processing, and office-document parsing
- broader SDK compatibility matrix
- provider-native tool-calling optimizations
- provider-native streaming proxy

## License

MIT. See [LICENSE](https://github.com/686f6c61/crupier/blob/main/LICENSE).

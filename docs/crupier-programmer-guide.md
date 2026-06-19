# Crupier Programmer Guide

This guide is for a developer who wants to add Crupier to an existing project and verify that the routing makes sense to a human, not only to the test suite.

## Project Setup

From a project root:

```bash
crupier init
```

Then configure `crupier.toml`:

- Enable only the providers the project is allowed to use.
- Keep API keys in environment variables or `.env`, never in `crupier.toml`.
- Put explicit model IDs in `[models].allow`.
- Keep `[orchestrator].mode = "deterministic"` until evals pass.
- Use `[orchestrator].mode = "hybrid"` when you want the model orchestrator to propose routes under guardrails.

For Ollama Cloud:

```toml
[providers.ollama]
enabled = true
host = "https://ollama.com/api"
env_key = "OLLAMA_API_KEY"
```

For Google Gemini, enable `[providers.google]` and set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in the environment.

## Package Readiness

When changing Crupier itself, run the release gate before handing the package to another project:

```bash
crupier release check
crupier release check --json
crupier release check --strict-public
crupier release check --verify-providers --provider openai --provider anthropic --provider ollama
```

Use `--skip-build` only inside CI steps that already build and check distributions separately.

The release check covers package metadata, final public version shape, version sync, typed package marker, README onboarding content, security policy, changelog, CI workflow, provider extras, console script, license metadata, safe public onboarding defaults, sdist + wheel build, `twine check`, and wheel/sdist install/import/CLI/`crupier init`/dry-run route/Python SDK smokes. Use `--strict-public` before publishing so warnings and skipped build checks fail the release. Use `--verify-providers` for the final local publish gate when real provider keys are loaded; it fails the release if discovery, readiness, or real smoke checks fail.

## Human Verification Loop

Run this loop before trusting a new project route:

```bash
crupier update --online
crupier verify --provider anthropic --provider google --provider ollama
crupier eval run
crupier eval run --orchestrator-mode hybrid
crupier eval compare "Answer this support ticket" --model openai:gpt-4.1-mini --model anthropic:claude-sonnet-4-6
crupier eval compare-dataset --dataset examples/model-compare-eval.json --model openai:gpt-4.1-mini --model anthropic:claude-sonnet-4-6
crupier audit
crupier audit --real --provider anthropic --provider google --provider ollama
crupier route "Compare two agent architectures and identify risks" --mode research
crupier route "Answer briefly" --max-cost-usd 0.001
crupier route "Probe exact model" --force-model openai:gpt-4.1-mini
```

What good looks like:

- `verify` reports `ready` for the providers/models you rely on.
- `eval run` passes the built-in or project dataset.
- `audit` shows no failing checks and gives route-review questions a human can answer.
- `route` prints a strategy, models and score reasons a human can inspect.
- `eval compare` shows a recommended variant plus the human questions a reviewer should answer.
- Hybrid evals do not weaken hard project profiles such as `private`.

## A/B Route Comparison

Compare candidate models or route variants before recording feedback:

```bash
crupier eval compare "Reply exactly crupier-ok" \
  --model openai:gpt-4.1-mini \
  --model anthropic:claude-sonnet-4-6 \
  --expect-contains crupier-ok \
  --max-cost-usd 0.02 \
  --no-dry-run
```

For non-model variants, pass JSON:

```bash
crupier eval compare "Review this code change" \
  --variant '{"name":"fast","mode":"fast"}' \
  --variant '{"name":"agentic","mode":"agentic","constraints":{"risk_level":"high"}}'
```

The winner is only a data-backed recommendation: deterministic checks first, then lower cost, lower latency, and fewer model calls. A maintainer should still inspect the preview and answer the human checks before using `crupier feedback record`.

When the compare output is the thing being reviewed, write a report, generate a human review packet, and record feedback from that report instead of copying route metadata by hand:

```bash
crupier eval compare "Reply exactly crupier-ok" \
  --mode fast \
  --model openai:gpt-4.1-mini \
  --model anthropic:claude-sonnet-4-6 \
  --expect-contains crupier-ok \
  --no-dry-run \
  --write-report

crupier feedback review \
  --compare-report .crupier/evals/runs/compare_YYYYMMDDTHHMMSSZ.json \
  --write-report \
  --write-decisions-template

crupier feedback record \
  --compare-report .crupier/evals/runs/compare_YYYYMMDDTHHMMSSZ.json \
  --variant anthropic:claude-sonnet-4-6 \
  --rating 2 \
  --verdict needs_work \
  --tag weak_answer

crupier feedback import-decisions \
  --decisions .crupier/feedback/decisions/human_decisions_YYYYMMDDTHHMMSSZ.json \
  --apply-to-registry
```

`feedback review` writes JSON and Markdown under `.crupier/feedback/reviews/`. A reviewer gets each variant, route metadata, human checks, optional output preview, and ready commands for `accept`, `needs_work`, or `reject`. `--write-decisions-template` writes an editable JSON under `.crupier/feedback/decisions/` so the reviewer can mark exactly which variants should become feedback. The template omits output previews; the imported feedback stores only route metadata, rating, verdict, tags and redacted note. Dry-run compare reports cannot close production feedback gates unless the reviewer explicitly passes `--allow-dry-run-source`, which is meant for non-production calibration only.

For repeated project cases, compare a dataset:

```bash
crupier eval compare-dataset \
  --dataset examples/model-compare-eval.json \
  --model openai:gpt-4.1-mini \
  --model anthropic:claude-sonnet-4-6 \
  --record-history \
  --write-report
```

Inspect history before applying:

```bash
crupier eval history
```

After the team trusts the dataset and the historical signal has enough sample size, apply aggregate scores:

```bash
crupier eval history \
  --min-count 10 \
  --min-confidence high \
  --apply
```

This writes `eval:<mode>` scores into local capability cards only when the apply gate passes. Use this for repeatable deterministic signals; use `crupier feedback record` for subjective human judgement after reviewing actual outputs. The history file stores aggregate model metrics, not raw prompts or responses.

## Human Feedback Loop

Use feedback when the route/result technically works but a maintainer would not ship it.

Manual model feedback:

```bash
crupier feedback record \
  --model anthropic:claude-sonnet-4-6 \
  --mode agentic \
  --rating 2 \
  --verdict needs_work \
  --tag missed_constraint \
  --note "Technically valid, but not maintainable enough."

crupier feedback summary
crupier feedback apply
```

Trace-based feedback:

```bash
crupier deal "Plan this code agent step" --store-trace --trace summary
crupier feedback record --trace-id trc_... --rating 1 --verdict reject --tag wrong_route
crupier feedback apply
```

The feedback file stores route metadata, rating, verdict and tags, not prompts or responses. `feedback apply` aggregates the reviews and writes `human:<mode>` scores into local capability cards. Future route output includes a `human_feedback` score term, so another programmer can see why a model was promoted or penalized.

For production, recording feedback is not enough. Run `crupier feedback apply` after review so the selector actually sees the human judgement; `adopt doctor --production` reports feedback as incomplete while recorded reviews have not been applied to capability cards.

When a reviewer used a decision template, `crupier feedback import-decisions --apply-to-registry` performs both steps: it imports only entries with `record=true`, then writes the aggregate `human:<mode>` scores into capability cards.

## Programmer Code Comments

Before editing a project, ask Crupier for an adoption path:

```bash
crupier adopt doctor
crupier adopt doctor --real --provider anthropic --provider google --provider ollama --write-report
crupier adopt doctor --production --real --provider anthropic --provider google --provider ollama --write-report
crupier adopt package
crupier adopt handoff --write-report
crupier adopt handoff --production --real --provider anthropic --provider google --provider ollama --write-report
crupier adopt signoff --verdict approve --handoff .crupier/handoffs/adoption_handoff_YYYYMMDDTHHMMSSZ.md
crupier adopt plan
crupier adopt plan src app.py --write-report
crupier adopt patches --path recommended --write-report
```

Use `adopt doctor` first when the project is unfamiliar. It rolls the adoption plan, non-applied patch suggestions, project audit, real canary status, eval history, human feedback, and programmer comments into one readiness report.

Use `adopt package`, `adopt doctor`, `adopt plan`, `adopt patches`, and offline `adopt handoff --write-report` when a repo does not have `crupier.toml` yet. They are non-destructive, infer the project name from `package.json`, `pyproject.toml`, or the directory name, and can produce review artifacts before full provider configuration exists. `adopt package` is the easiest first command for another programmer because it writes the doctor, patch suggestions, code-comment reports, PR/review-comment packets, SARIF annotations, editable programmer decision template, handoff, and `.crupier/packages/adoption_package_*.md` index in one run.

Every doctor, handoff, and package includes a `review_contract` section. Treat it as the human/automation boundary: technical gates can be ready while `human_status` is still `needs_review` or `blocked`, and `must_not_auto_approve=true` means the rollout still needs human judgement.

Use real/production `adopt handoff` when another person needs to review the rollout after configuration exists. It writes a compact reviewer package under `.crupier/handoffs/` with the doctor state, recent feedback/code-comment artifacts, human decision templates, required human actions, and commands to close the remaining gates.

Use `adopt signoff` after the reviewer reads the handoff. `approve` records the project-level rollout approval; `reject` and `needs_work` deliberately block adoption even when technical checks pass.

Use `adopt doctor --production --real` before rollout. Production mode blocks when real canaries were skipped, no compare history exists, no human feedback has been recorded, or no approving adoption signoff exists. This is the guard for cases where the code path is technically green but the route/result is not acceptable to a maintainer.

Use the recommendation as the integration decision:

- `proxy`: best when an existing app already speaks OpenAI-compatible HTTP or centralizes base URL config.
- `compat_client`: best for small Python OpenAI client changes.
- `autopatch`: useful for experiments and tests, not the first production choice.
- `native_sdk`: best for multi-provider, agentic, tool, structured output, or multimodal work.

If `adopt doctor` or `adopt plan` reports blockers, fix those before changing routing behavior. Inline credentials are treated as blockers because adoption should not spread secret handling across more surfaces.

`crupier adopt patches` produces suggested diffs/snippets only; it does not edit files. Use it to hand another programmer a reviewable patch artifact:

```bash
crupier adopt patches --path compat_client
crupier adopt patches --path proxy
crupier adopt patches --path native_sdk
```

For `compat_client`, Crupier only suggests narrow Python import diffs it can recognize safely. For proxy, autopatch, and native SDK, it emits commands or starter modules rather than touching app code.

When adding Crupier to an existing codebase, generate integration comments:

```bash
crupier code comments
crupier code comments src app.py --write-report
crupier code comments src app.py --write-review-comments
crupier code comments src app.py --write-sarif
crupier code comments src app.py --write-decisions-template
crupier code comments src app.py --import-decisions .crupier/code-comments/decisions/code_comment_decisions_YYYYMMDDTHHMMSSZ.json
crupier code comments src app.py --ack-reviewed --reviewer-hash dev-a
```

The comments are not source rewrites. They identify AI call sites, hard-coded model choices, and plausible inline credentials so a maintainer can decide the right adoption path. Use `--write-review-comments` to create PR/review-comment Markdown and JSONL under `.crupier/code-comments/` without storing source snippets, and `--write-sarif` when you want CI/GitHub-style annotations:

- proxy/base URL for OpenAI-compatible projects
- `crupier.compat.openai.OpenAI` for small import changes
- `crupier.install("openai")` for opt-in experiments
- native `Crupier.from_project(".").deal(...)` for deeper agent integrations

Generated/dependency directories such as `build`, `dist`, `.venv`, `node_modules`, and `*.egg-info` are skipped so reviewers do not waste time on copied package output. Redaction-regex examples and short hyphenated identifiers are not treated as credentials. Credential-like values inside `tests/`, `fixtures/`, and test-named files remain visible as P3 test fixtures, while production source credentials stay P1 blockers.

Use `--write-decisions-template` for granular human review. A programmer can mark each comment as `accepted`, `false_positive`, `not_applicable`, `reviewed`, `resolved`, `needs_change`, `rejected`, or `unresolved`; only reviewed/resolved verdicts close those fingerprints. Use `--ack-reviewed` only after a programmer has inspected and accepted the full current comment set. Crupier stores comment fingerprints and verdict metadata, not source snippets, so `adopt doctor` can pass the programmer-comment gate until new or changed hotspots appear.

For a complete adoption report:

```bash
crupier adopt doctor --write-report
```

Reports are written under `.crupier/audits/` as JSON and Markdown.

## Trace Storage And Replay

Store a trace only when a project needs auditability:

```bash
crupier deal "Plan this agent step" --store-trace --trace summary
crupier trace list
crupier trace show trc_...
crupier trace delete trc_...
```

`--store-trace` saves route/cost/provider metadata without raw prompt or response text. This is enough for human inspection, but not for replay.

Replay requires explicit prompt storage:

```bash
crupier deal "Plan this agent step" --store-prompt --store-response --trace summary
crupier trace replay trc_...
```

Use this only for non-sensitive test cases or controlled debugging. Secret-like values are redacted before writing, and stored traces live under `.crupier/traces/`.

## Project Evals

Create a JSON or JSONL dataset when the defaults are too generic.

```json
{
  "name": "my-project-routing",
  "cases": [
    {
      "id": "research_decision",
      "task": "Compare two database migration plans and identify risks.",
      "mode": "research",
      "expect": {
        "strategy": "fusion",
        "min_models": 2,
        "max_models": 4
      }
    }
  ]
}
```

Run it:

```bash
crupier eval run --dataset examples/routing-eval.json --write-report
```

Eval expectations currently support:

- `strategy`
- `strategy_in`
- `risk_level`
- `models_include`
- `models_exclude`
- `providers_include`
- `providers_exclude`
- `min_models`
- `max_models`
- `roles_include`
- `roles_exclude`

## Integration Options

Native SDK:

```python
from crupier import Crupier

crupier = Crupier.from_project(".")
result = crupier.deal(
    task="Route this agent step",
    input={"step": "modify files and run tests"},
    mode="agentic",
    trace="summary",
)
print(result.output_text)
print(result.route.summary)
```

Structured output:

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
    constraints={"max_cost_usd": 0.01},
    dry_run=False,
)
print(result.output_json)
```

File context:

```python
result = crupier.deal(
    task="Summarize the attached design notes.",
    files=["notes.md", "agent.py"],
    constraints={"max_file_context_chars": 40000},
    dry_run=False,
)

print(result.output_text)
```

Images are sent as native vision input when the selected model supports images. Text-like files and code are extracted into bounded text context. PDFs are extracted into text chunks when `pypdf` is installed via `crupier[pdf]` or a local `pdftotext` binary is available. Native PDF provider upload, OCR, table-aware extraction, audio/video transcription, and office-document parsing are still separate roadmap items.

Budget behavior:

- Crupier estimates route cost before provider calls.
- `constraints={"max_cost_usd": ...}` blocks over-budget routes before spend.
- Actual cost is estimated from provider usage metadata when available.
- The model orchestrator cannot set its own trusted cost; Crupier recalculates it.

Local tools:

```python
def lookup_user(name: str):
    """Look up a user by name."""
    return {"name": name, "id": "usr_123"}

result = crupier.deal(
    task="Find Ada and summarize the result",
    tools=[lookup_user],
    constraints={"max_cost_usd": 0.01},
    dry_run=False,
)

print(result.provider_metadata["tool_calls"])
```

Tools with side effects should require approval:

```python
tool = {
    "name": "write_file",
    "description": "Write a file to disk.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    "handler": write_file,
    "requires_approval": True,
}

result = crupier.deal(
    task="Write the report file.",
    tools=[tool],
    constraints={"approved_tools": ["write_file"]},
    dry_run=False,
)
```

Crupier writes a tool-call ledger in `result.provider_metadata["tool_calls"]`. Duplicate tool calls with the same name and arguments are skipped by idempotency key.

OpenAI-like client:

```python
from crupier.compat.openai import OpenAI

client = OpenAI(project=".")
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Summarize this"}],
)
print(response.choices[0].message.content)
```

OpenAI-compatible server:

```bash
crupier serve --port 8787
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
```

## Extension Points

- `CapabilityCard`: provider/model capability metadata.
- `PolicyEngine`: hard filtering and post-plan route validation.
- `validate_route_plan_shape`: strict plan shape checks before policy/provider execution.
- `estimate_route_cost`: pre-execution budget estimate.
- `parse_and_validate_json`: structured-output validation.
- `normalize_tools` / `execute_tool_plan`: provider-agnostic local tool execution.
- `ProjectAuditRunner`: project adoption checks, human route reviews, and real canaries.
- `scan_code_comments`: programmer-facing comments for integration hotspots.
- `ModelSelector`: deterministic scoring.
- `Orchestrator`: interface for deterministic, model, hybrid or locked planners.
- `RoutingEvalRunner`: project-level evals for human-relevant routing expectations.

Important guardrail: model-powered orchestration can propose a route, but it cannot use models outside the policy-filtered candidates, cannot violate explicit profile strategies, and cannot bypass `PolicyEngine.validate_route`.

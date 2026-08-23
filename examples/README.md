# Crupier 0.6.0 examples

These examples are integration blueprints, not provider benchmarks. The default path is offline: it builds the real capability registry, policy filters, scoring terms, route plan, cost estimate, and decision trace without spending tokens or requiring API keys.

Run commands from the repository root.

## Start here

| Example | What it demonstrates | Provider calls |
| --- | --- | --- |
| `sdk_dry_run.py` | Smallest useful SDK route with a cost and latency budget | No |
| `drop_in_agent_boundary.py` | One AI boundary for an existing app, queue worker, or agent framework | No |
| `routing_tradeoffs.py` | Why fast, structured, research, agentic, and delegated work produce different routes | No |
| `workflow_operations_hub.py` | Multiple business workflows sharing one configured Crupier client | No |
| `approval_workflow.py` | Frozen route hash, durable reviewer decision, and one-use token without executing the sensitive tool | No |
| `session_contract_review.py` | Route stickiness across turns and automatic replan when a contract file appears | No |
| `shadow_canary_rollout.py` | Sampled shadow planning, external quality evidence, and promotion gates | No |

```bash
python examples/sdk_dry_run.py
python examples/drop_in_agent_boundary.py
python examples/routing_tradeoffs.py
python examples/workflow_operations_hub.py
python examples/approval_workflow.py
python examples/session_contract_review.py
python examples/shadow_canary_rollout.py
```

The output includes the chosen strategy, models by role, leading score and terms, estimated cost and latency, policy filters, approval requirements, candidate counts, exclusions, planned versus real provider calls, warnings, trace errors, and trace ID. It does not print prompts or model responses.

The shared offline client loads the same named profile preferences demonstrated
by the project template. Constraints such as `requires_tools`,
`requires_human_approval`, and `allow_parallel` are enforced by the SDK; they
are not application-only labels.

## Focused workflows

| Example | Production question it answers |
| --- | --- |
| `customer_support_triage.py` | Can a cheap, low-latency route still satisfy a strict support schema? |
| `agentic_pr_review.py` | Which route should review a high-risk code change with approved tools and human review? |
| `multimodal_claim_review.py` | Which files go native, how bounded CSV rows execute, and which richer pipelines remain explicit boundaries? |
| `specialized_operations.py` | How are embeddings, reranking, transcription, speech, and image generation kept away from chat-only models? |
| `eval_feedback_loop.py` | How do route comparisons and human judgement become project-local model signals? |
| `fail_closed_safety.py` | What stops a credential, a malformed policy rule, or a pasted secret from reaching a provider or a stored artifact? |

```bash
python examples/customer_support_triage.py
python examples/agentic_pr_review.py
python examples/multimodal_claim_review.py
python examples/specialized_operations.py
python examples/eval_feedback_loop.py
python examples/fail_closed_safety.py
```

`specialized_operations.py` uses OpenAI, Google, and NaN capability cards in `dry_run` mode. It proves operation filtering and selection; it does not claim those providers are configured or healthy for your account. Its routes report `planned_provider_calls=0`: operation planning does not stage dry-run calls in the trace, so `real_provider_calls=0` is the line that proves nothing was sent.

`fail_closed_safety.py` covers the three safety contracts of 0.6.0 that the other examples take for granted: canonical credentials are refused on unofficial hosts unless the project opts in with `allow_custom_host` over HTTPS, a malformed `[policy]` table raises `CrupierConfigError` instead of degrading to an allow-all policy, and secrets are redacted before they reach a trace observation, a stored feedback note, or the tool error text returned to a model. Its closing route shows three fail-closed filters at once: `stable_models_only`, `openrouter_byok`, and a declarative deny rule. The credential it uses is synthetic and only ever printed redacted.

`eval_feedback_loop.py` deliberately shows a cheap route winning deterministic dry-run checks and then receiving negative human feedback for insufficient review depth. Dry-run comparison validates route shape and estimated economics, not answer quality. Its feedback JSONL is created in a temporary directory and removed on exit.

## Real validation

The validation harnesses are the examples that call configured providers. They load your project `crupier.toml`, use its allowlist and environment variables, and fail with a non-zero exit code when a check fails.

Sanitization applies to the recorded provider calls: each one is reduced to an allowlist of route metadata (role, provider, model, attempt, status, latency, and the operation or multimodal counters of that case) and never carries prompts, headers, or credentials. It does not apply to the answers under test. `live_routing_validation.py` stores an `output_preview` of up to 1000 characters plus any structured `output_json`, and `live_operations_validation.py` stores transcription text, because that content is the evidence a real check needs. Treat the generated report as local evidence, keep it inside `.crupier/`, and never commit or publish it.

```bash
python examples/live_routing_validation.py --real --project . --write-report
python examples/live_operations_validation.py --real --project . --write-report
```

Narrow a run while diagnosing a provider or capability:

```bash
python examples/live_routing_validation.py --real --project . --case tools --case delegate
python examples/live_operations_validation.py --real --project . --case embeddings --case rerank
```

`live_routing_validation.py` covers single, cascade, fusion, critique/repair, iterative tools, delegation, native image input, and PDF extraction. `live_operations_validation.py` covers operation classification, embeddings, reranking, audio, image generation, the OpenAI-compatible Python surface, and the optional HTTP server.

## Eval datasets

- `routing-eval.json` checks expected strategies, roles, and model counts; its local-only case also pins which providers a route may and may not use.
- `model-compare-eval.json` compares route variants over repeatable project cases.

These datasets are intentionally small enough to inspect in code review. Replace their tasks and expectations with representative production cases before applying eval or feedback scores to a registry.

## Verify the folder

```bash
python -m pytest -q tests/test_examples.py
ruff check examples
```

The examples test runs every public Python script without provider keys and
verifies route evidence, warnings, approval and multimodal boundaries, dataset
paths, and that no `.crupier` directory is left behind.

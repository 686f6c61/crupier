# Crupier Real Capability Probes - 2026-06-18

Date: 2026-06-18

Scope: real provider capability probes using transient environment variables. No API keys, prompts, raw model responses, or provider raw payloads were written to this report.

## Discovery

- OpenAI: 126 models discovered. Selected `openai:gpt-5.4-mini`.
- Anthropic: 8 models discovered. Selected `anthropic:claude-haiku-4-5-20251001`.
- Ollama Cloud: 35 models discovered. Selected `ollama:gpt-oss:120b`.

## Initial Full Probe Run

Summary: 17 verified, 1 failed.

| Provider | Model | Probe | Status | Latency ms | Note |
| --- | --- | --- | --- | ---: | --- |
| OpenAI | `openai:gpt-5.4-mini` | `text_basic` | verified | 3578 |  |
| OpenAI | `openai:gpt-5.4-mini` | `json_instruction` | verified | 1481 |  |
| OpenAI | `openai:gpt-5.4-mini` | `max_output_param` | failed | 200 | OpenAI minimum `max_output_tokens` was 16; probe requested 8. |
| OpenAI | `openai:gpt-5.4-mini` | `structured_output` | verified | 1091 |  |
| OpenAI | `openai:gpt-5.4-mini` | `tool_call` | verified | 1790 |  |
| OpenAI | `openai:gpt-5.4-mini` | `streaming` | verified | 1317 |  |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `text_basic` | verified | 2517 |  |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `json_instruction` | verified | 855 |  |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `max_output_param` | verified | 1073 |  |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `structured_output` | verified | 871 |  |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `tool_call` | verified | 732 |  |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `streaming` | verified | 587 |  |
| Ollama Cloud | `ollama:gpt-oss:120b` | `text_basic` | verified | 1028 |  |
| Ollama Cloud | `ollama:gpt-oss:120b` | `json_instruction` | verified | 1436 |  |
| Ollama Cloud | `ollama:gpt-oss:120b` | `max_output_param` | verified | 1433 |  |
| Ollama Cloud | `ollama:gpt-oss:120b` | `structured_output` | verified | 1535 |  |
| Ollama Cloud | `ollama:gpt-oss:120b` | `tool_call` | verified | 2433 |  |
| Ollama Cloud | `ollama:gpt-oss:120b` | `streaming` | verified | 878 |  |

Initial readiness summary: 2 ready, 1 failed.

## Fix and Re-Test

Fix: changed `max_output_param` probe from `max_output_tokens=8` to `max_output_tokens=16`.

OpenAI re-test summary: 6 verified.

| Provider | Model | Probe | Status | Latency ms |
| --- | --- | --- | --- | ---: |
| OpenAI | `openai:gpt-5.4-mini` | `text_basic` | verified | 2236 |
| OpenAI | `openai:gpt-5.4-mini` | `json_instruction` | verified | 1014 |
| OpenAI | `openai:gpt-5.4-mini` | `max_output_param` | verified | 923 |
| OpenAI | `openai:gpt-5.4-mini` | `structured_output` | verified | 756 |
| OpenAI | `openai:gpt-5.4-mini` | `tool_call` | verified | 3398 |
| OpenAI | `openai:gpt-5.4-mini` | `streaming` | verified | 870 |

Strict readiness after re-test:

- `openai:gpt-5.4-mini`: ready
- `anthropic:claude-haiku-4-5-20251001`: ready in initial full run
- `ollama:gpt-oss:120b`: ready in initial full run

Privacy check: no sentinel prompt/output strings were found in persisted temporary capability cards.

# Security Policy

Crupier 0.1.0 is the first public package release line. Please do not include
API keys, prompts, responses, customer data, or other secrets in public issues
or reports.

## Scope

Security-sensitive areas include:

- provider credential handling
- prompt, response, trace, eval, and feedback persistence
- `.env`, `.crupier/`, and release artifact contents
- OpenAI-compatible proxy behavior
- adapter error handling for OpenAI, Anthropic Claude, Google Gemini, Ollama Cloud, and OpenRouter
- local file, PDF, image, tool, and multimodal routing paths

## Reporting A Vulnerability

Report suspected vulnerabilities privately to the project maintainers. Use
GitHub private vulnerability reporting once the public repository enables it.
Until a public private-reporting channel is configured, keep reports out of
public issues, pull requests, discussions, screenshots, and chat logs.

Include:

- affected version or commit
- reproduction steps
- expected and actual impact
- whether provider credentials, prompts, responses, traces, or local files are involved
- a minimal redacted reproduction that avoids real provider keys or customer data

Do not include:

- API keys, bearer tokens, or provider account IDs
- raw prompts, raw responses, customer files, or private traces
- `.env` files
- full `.crupier/` directories
- screenshots that reveal secrets or provider dashboards

If a credential is accidentally shared, rotate it immediately before continuing
the report.

## Supported Versions

| Version line | Supported |
| --- | --- |
| Latest `0.x` release | Yes |
| Older `0.x` releases | Best effort until the next patch is available |
| Unreleased local builds | No public security support |

## Secret Handling Expectations

Crupier should receive provider credentials through environment variables or a
local `.env` file excluded from source control. Do not pass API keys as CLI
arguments, write them into `crupier.toml`, or include them in eval datasets,
feedback notes, trace metadata, bug reports, or release artifacts.

By default, Crupier should not persist prompts or responses. Any storage of
prompts, responses, traces, or replayable inputs must stay opt-in and should be
redacted before it is shared outside the local project.

## Disclosure And Fix Process

Maintainers should acknowledge private reports, reproduce the issue with
redacted inputs, prepare a patch, and publish release notes that describe the
impact without exposing exploit details or secrets. Public disclosure should
happen only after a fix or mitigation is available, unless active exploitation
requires earlier notice.

## Non-Security Issues

Model quality problems, routing preferences, provider outages, cost surprises,
and documentation gaps are usually normal issues unless they expose secrets,
bypass configured policy, route to a forbidden provider/model, or persist data
that should not be stored.

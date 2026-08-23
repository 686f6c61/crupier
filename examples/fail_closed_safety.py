"""Prove the fail-closed safety contracts without provider keys or provider calls.

    python examples/fail_closed_safety.py

Three boundaries are demonstrated, in this order:

1. Credentials never reach an unofficial host by accident. A canonical provider
   key pointed at a custom host is rejected; `allow_custom_host = true` over
   HTTPS is the explicit opt-in, plain HTTP outside loopback stays rejected even
   with the opt-in, and a generic inference endpoint may not reuse a canonical
   credential at all.
2. A malformed policy rule raises `CrupierConfigError` instead of degrading to an
   empty policy that would allow every model.
3. Secrets are redacted centrally before they reach a trace observation, a stored
   feedback note, or the tool error text that is handed back to a model.

The closing route runs the same hardened project and shows three fail-closed
filters at once: preview models, the disabled OpenRouter BYOK provider, and a
declarative deny rule. Nothing is printed in clear text: the synthetic credential
below only ever appears redacted.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Any

from _example_support import offline_client, print_route

from crupier import CrupierConfigError, CrupierResult, HumanFeedbackStore
from crupier.config import CrupierConfig
from crupier.redaction import redact_text, redact_value

# Synthetic credential assembled at runtime, so the file holds no literal shaped
# like a real key that a secret scanner (the one in `crupier release check`, for
# instance) could flag.
EXAMPLE_SECRET = "s" + "k-" + "proj-" + "EXAMPLE" + "0" * 24

CUSTOM_HOST = "https://models.internal.example/v1"
INSECURE_HOST = "http://models.internal.example/v1"

# Well-formed declarative rule: the contrast with the malformed ones shows that the
# same table that fails closed when it is invalid still filters when it is valid.
NO_LOCAL_DAEMON_RULE = {
    "name": "no_local_daemon_for_customer_data",
    "effect": "deny",
    "providers": ["ollama"],
    "reason": "Customer data may not be sent to the local daemon in this project",
}


def lookup_invoice(invoice_id: str) -> dict[str, str]:
    """Example tool whose failure leaks a credential into the error message.

    Crupier runs tools behind a boundary that applies ``redact_text`` to the
    error detail before it reaches the model or the trace, which is exactly what
    this example reproduces.
    """

    raise RuntimeError(
        f"Vendor API rejected the call for {invoice_id} with credential {EXAMPLE_SECRET}"
    )


def main() -> None:
    credential_boundary = {
        "canonical_key_on_custom_host": _endpoint_verdict(
            "openai",
            {"enabled": True, "env_key": "OPENAI_API_KEY", "host": CUSTOM_HOST},
            expect="is not an official endpoint",
        ),
        "custom_host_with_explicit_optin": _endpoint_verdict(
            "openai",
            {
                "enabled": True,
                "env_key": "OPENAI_API_KEY",
                "host": CUSTOM_HOST,
                "allow_custom_host": True,
            },
        ),
        "custom_host_without_https": _endpoint_verdict(
            "openai",
            {
                "enabled": True,
                "env_key": "OPENAI_API_KEY",
                "host": INSECURE_HOST,
                "allow_custom_host": True,
            },
            expect="must use HTTPS outside loopback",
        ),
        "generic_endpoint_reusing_canonical_key": _endpoint_verdict(
            "inference",
            {
                "enabled": True,
                "mode": "openai_compatible",
                "env_key": "OPENAI_API_KEY",
                "host": CUSTOM_HOST,
            },
            expect="cannot reuse the canonical credential",
        ),
        "generic_endpoint_with_own_key": _endpoint_verdict(
            "inference",
            {
                "enabled": True,
                "mode": "openai_compatible",
                "env_key": "INFERENCE_API_KEY",
                "host": CUSTOM_HOST,
            },
        ),
    }

    policy_boundary = {
        "policy_table_of_wrong_type": _policy_verdict(
            "deny-everything",
            expect="policy must be a table/object",
        ),
        "rule_that_is_not_a_table": _policy_verdict(
            {"rules": ["deny"]},
            expect="must be a table/object, not str",
        ),
        "rule_with_unsupported_effect": _policy_verdict(
            {"rules": [{"name": "allow_all", "effect": "allow"}]},
            expect="unsupported effect",
        ),
        "well_formed_deny_rule": _policy_verdict(
            {"rules": [NO_LOCAL_DAEMON_RULE]},
            expected_rules=1,
        ),
    }

    try:
        lookup_invoice("INV-2098")
    except RuntimeError as exc:
        redacted_tool_error = redact_text(str(exc))
    else:  # pragma: no cover - la herramienta de ejemplo siempre falla
        raise RuntimeError("The example tool was expected to fail")

    redacted_observation = redact_value(
        {
            "provider": "openai",
            "authorization": f"Bearer {EXAMPLE_SECRET}",
            "environment": f"OPENAI_API_KEY={EXAMPLE_SECRET}",
        }
    )

    crupier = offline_client(
        project="fail-closed-safety",
        profile="fast",
        allow=[
            "openai:gpt-5.4-mini",
            "anthropic:claude-sonnet-4-6",
            # Excluded by stable_models_only: preview models are turned off.
            "google:gemini-3.1-pro-preview",
            # Excluded by openrouter_byok: OpenRouter is optional BYOK and is not
            # enabled, so it is never even evaluated against the policy rules.
            "openrouter:openai/gpt-5.5",
            # Excluded by the declarative NO_LOCAL_DAEMON_RULE.
            "ollama:gpt-oss:120b",
        ],
        policy_rules=[NO_LOCAL_DAEMON_RULE],
    )
    result = crupier.deal(
        "Summarize the vendor security questionnaire for the compliance tracker.",
        input={"vendor": "example-payments", "questionnaire_items": 42},
        mode="fast",
        dry_run=True,
        trace="summary",
    )
    if result.route is None:
        raise RuntimeError("fail_closed_safety did not produce a route plan")

    with TemporaryDirectory(prefix="crupier-fail-closed-") as root:
        store = HumanFeedbackStore(root)
        record = store.record(
            project="fail-closed-safety",
            rating=3,
            verdict="needs_work",
            models=result.route.models,
            mode="fast",
            strategy=result.route.strategy,
            tags=["dry_run_source", "credential_pasted_by_mistake"],
            note=f"Reviewer pasted OPENAI_API_KEY={EXAMPLE_SECRET} into the review note.",
        )
        stored_note = record.note

    print_route(
        "fail_closed_safety",
        result,
        extra={
            **credential_boundary,
            **policy_boundary,
            "excluded_models": _format_exclusions(result),
            "redacted_tool_error": redacted_tool_error,
            "redacted_trace_observation": redacted_observation,
            "redacted_feedback_note": stored_note,
            "feedback_persistence": "temporary",
            "secret_printed_in_clear": any(
                EXAMPLE_SECRET in str(value)
                for value in (
                    *credential_boundary.values(),
                    *policy_boundary.values(),
                    redacted_tool_error,
                    redacted_observation,
                    stored_note,
                )
            ),
        },
    )
    crupier.close()


def _endpoint_verdict(
    provider: str,
    settings: dict[str, Any],
    *,
    expect: str | None = None,
) -> str:
    """Report whether Crupier would send credentials to that endpoint.

    ``expect`` is the fragment the message must contain when the contract rejects
    the configuration. Without that check the verdict would lie in both
    directions: ``CrupierConfig.from_dict`` turns any ``TypeError`` or
    ``ValueError`` into ``CrupierConfigError``, so a bug in this example would
    read as "the boundary held"; and ``validate_provider_endpoint`` only runs for
    enabled providers, so a provider left disabled by mistake would return
    "accepted" without anything having been validated.
    """

    try:
        config = _single_provider_config(provider, settings)
    except CrupierConfigError as exc:
        return f"rejected:{_rejection_reason(exc, expect)}"
    if not config.providers[provider].enabled:
        raise RuntimeError(
            f"Provider {provider!r} ended up disabled, so the credential boundary "
            "never ran and this verdict would prove nothing"
        )
    return "accepted"


def _policy_verdict(policy: Any, *, expect: str | None = None, expected_rules: int = 0) -> str:
    """Report whether a `[policy]` table is accepted or fails closed.

    ``expected_rules`` gives the accepted path positive evidence: it checks that the
    rule really parsed, not merely that no exception was raised.
    """

    try:
        config = _single_provider_config(
            "openai",
            {"enabled": True, "env_key": "OPENAI_API_KEY"},
            policy=policy,
        )
    except CrupierConfigError as exc:
        return f"rejected:{_rejection_reason(exc, expect)}"
    if len(config.policy.rules) != expected_rules:
        raise RuntimeError(
            f"Expected {expected_rules} parsed policy rule(s), got {len(config.policy.rules)}"
        )
    return "accepted"


def _single_provider_config(
    provider: str,
    settings: dict[str, Any],
    *,
    policy: Any = None,
) -> CrupierConfig:
    """Build the smallest configuration needed to validate one contract."""

    data: dict[str, Any] = {
        "project": {"name": "fail-closed-safety", "default_profile": "fast"},
        "providers": {provider: settings},
        "models": {"allow": ["openai:gpt-5.4-mini"]},
        "profiles": {"fast": {"prefer": ["low_latency"], "strategy": "orchestrated"}},
    }
    if policy is not None:
        data["policy"] = policy
    return CrupierConfig.from_dict(data)


def _rejection_reason(exc: CrupierConfigError, expect: str | None) -> str:
    """Reduce the rejection to one redacted line and check it is the expected one."""

    message = redact_text(" ".join(str(exc).split()))
    if expect is not None and expect not in message:
        raise RuntimeError(
            f"Expected the configuration to be rejected because it {expect!r}, "
            f"but it failed for another reason: {message}"
        )
    head, separator, _ = message.partition(". ")
    if not separator:
        return head
    return f"{head}."


def _format_exclusions(result: CrupierResult) -> str:
    """List every discarded model with its reason, and nothing else."""

    trace = result.trace
    if trace is None:
        return "trace=disabled"
    return ";".join(
        f"{item.get('model')}:{item.get('reason')}" for item in trace.excluded_models
    ) or "none"


if __name__ == "__main__":
    main()

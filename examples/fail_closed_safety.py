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

from crupier import CrupierConfigError, HumanFeedbackStore
from crupier.config import CrupierConfig
from crupier.redaction import redact_text, redact_value

# Credencial sintética compuesta en tiempo de ejecución: así el fichero no
# contiene ningún literal con formato de clave real que un escáner de secretos
# (el del propio `crupier release check`, por ejemplo) pueda marcar.
EXAMPLE_SECRET = "s" + "k-" + "proj-" + "EXAMPLE" + "0" * 24

CUSTOM_HOST = "https://models.internal.example/v1"
INSECURE_HOST = "http://models.internal.example/v1"

# Regla declarativa bien formada: el contraste con la malformada demuestra que la
# misma tabla que falla cerrada cuando es inválida sigue filtrando cuando es válida.
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
        ),
        "generic_endpoint_reusing_canonical_key": _endpoint_verdict(
            "inference",
            {
                "enabled": True,
                "mode": "openai_compatible",
                "env_key": "OPENAI_API_KEY",
                "host": CUSTOM_HOST,
            },
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
        "policy_table_of_wrong_type": _policy_verdict("deny-everything"),
        "rule_that_is_not_a_table": _policy_verdict({"rules": ["deny"]}),
        "rule_with_unsupported_effect": _policy_verdict(
            {"rules": [{"name": "allow_all", "effect": "allow"}]}
        ),
        "well_formed_deny_rule": _policy_verdict({"rules": [NO_LOCAL_DAEMON_RULE]}),
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
            # Excluido por stable_models_only: los modelos preview están apagados.
            "google:gemini-3.1-pro-preview",
            # Excluido por openrouter_byok: OpenRouter es BYOK opcional y no está
            # habilitado, así que ni siquiera llega a evaluarse contra las reglas.
            "openrouter:openai/gpt-5.5",
            # Excluido por la regla declarativa NO_LOCAL_DAEMON_RULE.
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

    with TemporaryDirectory(prefix="crupier-fail-closed-") as root:
        store = HumanFeedbackStore(root)
        record = store.record(
            project="fail-closed-safety",
            rating=3,
            verdict="needs_work",
            models=result.route.models if result.route else [],
            mode="fast",
            strategy=result.route.strategy if result.route else None,
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
            "redacted_tool_error": redacted_tool_error,
            "redacted_trace_observation": redacted_observation,
            "redacted_feedback_note": stored_note,
            "feedback_persistence": "temporary",
            "secret_printed_in_clear": any(
                EXAMPLE_SECRET in str(value)
                for value in (redacted_tool_error, redacted_observation, stored_note)
            ),
        },
    )
    crupier.close()


def _endpoint_verdict(provider: str, settings: dict[str, Any]) -> str:
    """Indica si Crupier aceptaría enviar credenciales a ese endpoint."""

    try:
        _single_provider_config(provider, settings)
    except CrupierConfigError as exc:
        return f"rejected:{_first_sentence(exc)}"
    return "accepted"


def _policy_verdict(policy: Any) -> str:
    """Indica si una tabla `[policy]` es aceptada o falla cerrada."""

    try:
        _single_provider_config(
            "openai",
            {"enabled": True, "env_key": "OPENAI_API_KEY"},
            policy=policy,
        )
    except CrupierConfigError as exc:
        return f"rejected:{_first_sentence(exc)}"
    return "accepted"


def _single_provider_config(
    provider: str,
    settings: dict[str, Any],
    *,
    policy: Any = None,
) -> CrupierConfig:
    """Construye la configuración mínima necesaria para validar un contrato."""

    data: dict[str, Any] = {
        "project": {"name": "fail-closed-safety", "default_profile": "fast"},
        "providers": {provider: settings},
        "models": {"allow": ["openai:gpt-5.4-mini"]},
        "profiles": {"fast": {"prefer": ["low_latency"], "strategy": "orchestrated"}},
    }
    if policy is not None:
        data["policy"] = policy
    return CrupierConfig.from_dict(data)


def _first_sentence(exc: Exception) -> str:
    """Reduce el mensaje a una línea para que la evidencia siga siendo legible."""

    message = " ".join(str(exc).split())
    head, separator, _ = message.partition(". ")
    if not separator:
        return head
    return f"{head}."


if __name__ == "__main__":
    main()

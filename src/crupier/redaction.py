"""Redacción central de secretos para persistencia, errores y prompts internos."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

REPLACEMENT = "[redacted]"

# Nombres de campo comparados sin mayúsculas y sin guiones.
_SECRET_FIELD_TOKENS = frozenset(
    {
        "authorization",
        "xapikey",
        "apikey",
        "password",
        "passwd",
        "cookie",
        "setcookie",
        "secret",
        "accesstoken",
        "refreshtoken",
        "awssecretaccesskey",
        "awsaccesskeyid",
        "privatekey",
        "clientsecret",
    }
)

_SK_PREFIX = "s" + "k-"
_DEFAULT_PATTERN_SPECS: tuple[tuple[str, str, int], ...] = (
    (_SK_PREFIX + r"[A-Za-z0-9_\-]{10,}", REPLACEMENT, 0),
    (r"AIza[0-9A-Za-z\-_]{20,}", REPLACEMENT, 0),
    (r"AKIA[0-9A-Z]{16}", REPLACEMENT, 0),
    (r"ollama_[A-Za-z0-9_\-]{16,}", REPLACEMENT, 0),
    (r"nan_[A-Za-z0-9_\-]{16,}", REPLACEMENT, 0),
    (r"(Bearer\s+)[A-Za-z0-9._\-]{12,}", r"\1" + REPLACEMENT, re.IGNORECASE),
    (r"([A-Z][A-Z0-9_]*_API_KEY=)[^\s]+", r"\1" + REPLACEMENT, 0),
    (r"(?i)(aws_secret_access_key\s*[=:]\s*)[A-Za-z0-9/+=]{30,}", r"\1" + REPLACEMENT, 0),
)
_DEFAULT_PATTERNS = tuple(
    (re.compile(pattern, flags), replacement) for pattern, replacement, flags in _DEFAULT_PATTERN_SPECS
)


def is_secret_field(name: str) -> bool:
    """Indica si el nombre de un campo debe redactarse por completo."""

    token = re.sub(r"[^a-z0-9]", "", str(name).lower())
    if token in _SECRET_FIELD_TOKENS:
        return True
    return token.endswith(("password", "secret", "apikey")) and len(token) >= 6


def redact_text(
    text: str,
    *,
    extra_patterns: Sequence[tuple[str | re.Pattern[str], str]] | None = None,
) -> str:
    """Redacta secretos textuales de proveedores y asignaciones de claves."""

    if not text:
        return text
    redacted = text
    for pattern, replacement in _iter_patterns(extra_patterns):
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_value(
    value: Any,
    *,
    extra_patterns: Sequence[tuple[str | re.Pattern[str], str]] | None = None,
) -> Any:
    """Redacta estructuras anidadas y aplica también la redacción textual."""

    if isinstance(value, str):
        return redact_text(value, extra_patterns=extra_patterns)
    if isinstance(value, list):
        return [redact_value(item, extra_patterns=extra_patterns) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, extra_patterns=extra_patterns) for item in value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if is_secret_field(str(key)):
                redacted[key] = REPLACEMENT
            else:
                redacted[key] = redact_value(item, extra_patterns=extra_patterns)
        return redacted
    return value


def _iter_patterns(
    extra_patterns: Sequence[tuple[str | re.Pattern[str], str]] | None,
) -> tuple[tuple[re.Pattern[str], str], ...]:
    extra: list[tuple[re.Pattern[str], str]] = []
    for pattern, replacement in extra_patterns or ():
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        extra.append((compiled, replacement))
    return _DEFAULT_PATTERNS + tuple(extra)

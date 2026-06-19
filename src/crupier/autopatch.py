"""Opt-in SDK monkeypatch helpers for drop-in adoption."""

from __future__ import annotations

from typing import Any


def install(
    providers: list[str] | tuple[str, ...] | str | None = None,
    **client_kwargs: Any,
) -> list[str]:
    """Patch supported SDK entrypoints in-process.

    The function is intentionally opt-in and conservative. Missing third-party
    SDKs are skipped instead of becoming hard Crupier dependencies.
    """

    if providers is None:
        providers = ["openai"]
    if isinstance(providers, str):
        providers = [item.strip() for item in providers.split(",") if item.strip()]

    patched: list[str] = []
    for provider in providers:
        if provider == "openai" and _patch_openai(**client_kwargs):
            patched.append("openai")
    return patched


def _patch_openai(**client_kwargs: Any) -> bool:
    try:
        import openai  # type: ignore[import-not-found]
    except Exception:
        return False

    from .compat.openai import OpenAI

    class PatchedOpenAI(OpenAI):
        def __init__(self, **kwargs: Any):
            merged = {**client_kwargs, **kwargs}
            super().__init__(**merged)

    openai.OpenAI = PatchedOpenAI
    return True

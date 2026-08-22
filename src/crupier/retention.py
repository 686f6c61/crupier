"""Shared retention helpers for project-local JSON artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .state import private_write_text


def is_expired(
    created_at: Any,
    ttl_days: int | None,
    *,
    fallback_path: Path | None = None,
    now: datetime | None = None,
) -> bool:
    if ttl_days is None:
        return False
    timestamp = _parse_timestamp(created_at)
    if timestamp is None and fallback_path is not None:
        try:
            timestamp = datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=UTC)
        except OSError:
            return False
    if timestamp is None:
        return False
    cutoff = (now or datetime.now(UTC)) - timedelta(days=ttl_days)
    return timestamp < cutoff


def prune_jsonl(path: Path, ttl_days: int | None) -> int:
    if ttl_days is None or not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return 0
    kept: list[str] = []
    removed = 0
    now = datetime.now(UTC)
    for line in lines:
        if not line.strip():
            kept.append(line)
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if isinstance(data, dict) and is_expired(data.get("created_at"), ttl_days, now=now):
            removed += 1
        else:
            kept.append(line)
    if removed:
        text = "\n".join(kept)
        private_write_text(path, text + ("\n" if text else ""))
    return removed


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)

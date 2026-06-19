"""Capability evidence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import CapabilityCard


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    capability: str
    supported: bool
    status: str
    source: str
    detail: dict[str, Any]


def capability_evidence(card: CapabilityCard, capability: str, *, declared: bool = False) -> CapabilityEvidence:
    """Resolve verified/inferred/failed support for a card capability."""

    data = card.capability_status.get(capability)
    if isinstance(data, dict):
        status = str(data.get("status", "unknown"))
        source = str(data.get("source", "capability_status"))
        if status == "verified":
            return CapabilityEvidence(capability, True, status, source, dict(data))
        if status == "failed":
            return CapabilityEvidence(capability, False, status, source, dict(data))
        if status == "inferred":
            return CapabilityEvidence(capability, declared or bool(data.get("declared", False)), status, source, dict(data))
        return CapabilityEvidence(capability, False, status, source, dict(data))

    if declared:
        return CapabilityEvidence(
            capability=capability,
            supported=True,
            status="inferred",
            source="capability_card",
            detail={"declared": True},
        )
    return CapabilityEvidence(
        capability=capability,
        supported=False,
        status="unknown",
        source="capability_card",
        detail={"declared": False},
    )


def capability_reason(evidence: CapabilityEvidence) -> str:
    return f"{evidence.capability} support is {evidence.status} via {evidence.source}"

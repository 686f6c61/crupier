import pytest

from crupier.capabilities import (
    CapabilityEvidence,
    capability_evidence,
    capability_reason,
)
from crupier.models import CapabilityCard, ModelRef


@pytest.mark.parametrize(
    ("evidence_declared", "declared", "supported"),
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, True),
    ],
)
def test_capability_evidence_inferred_status_respects_declared_flags(
    evidence_declared, declared, supported
):
    card = CapabilityCard(
        model_ref=ModelRef.parse("openai:test"),
        last_updated="2026-08-23",
        capability_status={
            "tool_call": {
                "status": "inferred",
                "source": "probe",
                "declared": evidence_declared,
            }
        },
    )

    evidence = capability_evidence(card, "tool_call", declared=declared)

    assert evidence.supported is supported
    assert evidence.status == "inferred"
    assert evidence.detail["declared"] is evidence_declared


def test_capability_reason_formats_evidence():
    evidence = CapabilityEvidence(
        capability="tool_call",
        supported=True,
        status="inferred",
        source="probe",
        detail={},
    )

    assert capability_reason(evidence) == "tool_call support is inferred via probe"

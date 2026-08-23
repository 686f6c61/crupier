from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

import crupier.approvals as approvals_module
from crupier.approvals import ApprovalManager, plan_hash, request_fingerprint
from crupier.errors import CrupierError
from crupier.models import (
    DecisionTrace,
    FileAsset,
    PreparedDeal,
    RequestEnvelope,
    RoutePlan,
    RouteStep,
)


def prepared_deal(*, files=None, tools=None):
    request = RequestEnvelope(
        task="Apply the reviewed change",
        files=list(files or []),
        tools=list(tools or []),
        mode="agentic",
        constraints={"requires_human_approval": True},
    )
    plan = RoutePlan(
        strategy="single",
        steps=[RouteStep(role="primary", model="openai:gpt-5.4-mini")],
        requires_user_confirmation=True,
    )
    trace = DecisionTrace(
        trace_id="trc_approval_test",
        request_summary="Apply the reviewed change",
        route_plan=plan,
    )
    return PreparedDeal(request=request, plan=plan, trace=trace, dry_run=False)


def test_approval_lifecycle_binds_plan_request_and_single_use_token(tmp_path):
    manager = ApprovalManager(
        tmp_path / "state.sqlite3",
        reviewer_verifier=lambda reviewer, context: {
            "verified": True,
            "subject": reviewer,
            "issuer": context["issuer"],
        },
    )
    prepared = prepared_deal()

    pending = manager.create(prepared, ttl_s=300)
    granted = manager.grant(
        pending.approval_id,
        reviewer="ana",
        ttl_s=120,
        reason="Rollback is explicit.",
        reviewer_context={"issuer": "test-oidc"},
    )

    assert pending.status == "pending"
    assert pending.plan_hash == plan_hash(prepared.plan)
    assert pending.request_fingerprint == request_fingerprint(prepared.request)
    assert granted.status == "granted"
    assert granted.reviewer_verified is True
    assert granted.reviewer_attestation == {
        "verified": True,
        "subject": "ana",
        "issuer": "test-oidc",
    }
    assert granted.token is not None
    assert granted.token not in str(manager.store.get("approval", pending.approval_id).payload)

    restored = manager.consume(granted.token)

    assert restored.plan.to_dict() == prepared.plan.to_dict()
    assert restored.request.metadata["_crupier_approval"]["reviewer"] == "ana"
    assert manager.get(pending.approval_id).status == "consumed"
    assert [event["event"] for event in manager.events(pending.approval_id)] == [
        "requested",
        "granted",
        "consumed",
    ]
    with pytest.raises(CrupierError, match="consumed"):
        manager.consume(granted.token)


def test_approval_reject_revoke_validation_and_filters(tmp_path):
    manager = ApprovalManager(tmp_path / "state.sqlite3")
    rejected = manager.create(prepared_deal())
    revoked = manager.create(prepared_deal())

    rejected_result = manager.reject(
        rejected.approval_id,
        reviewer="ana",
        reason="Missing rollback.",
    )
    granted = manager.grant(revoked.approval_id, reviewer="ana")
    revoked_result = manager.revoke(
        revoked.approval_id,
        reviewer="ana",
        reason="Environment changed.",
    )

    assert rejected_result.status == "rejected"
    assert revoked_result.status == "revoked"
    assert [item.approval_id for item in manager.list(status="rejected")] == [
        rejected.approval_id
    ]
    with pytest.raises(CrupierError, match="only pending"):
        manager.grant(rejected.approval_id, reviewer="ana")
    with pytest.raises(CrupierError, match="cannot be empty"):
        manager.grant(manager.create(prepared_deal()).approval_id, reviewer="")
    with pytest.raises(CrupierError, match="Unknown approval status"):
        manager.list(status="mystery")
    assert granted.token is not None
    with pytest.raises(CrupierError, match="revoked"):
        manager.consume(granted.token)


def test_approval_detects_changed_file_before_consuming_token(tmp_path):
    document = tmp_path / "contract.txt"
    document.write_text("version one", encoding="utf-8")
    manager = ApprovalManager(tmp_path / "state.sqlite3")
    pending = manager.create(
        prepared_deal(files=[FileAsset(kind="text", name="contract.txt", uri=str(document))])
    )
    granted = manager.grant(pending.approval_id, reviewer="ana")
    assert granted.token is not None
    document.write_text("version two", encoding="utf-8")

    with pytest.raises(CrupierError, match="fingerprint"):
        manager.consume(granted.token)

    assert manager.get(pending.approval_id).status == "granted"


def test_approved_tools_require_the_original_runtime_catalog_and_handlers(tmp_path):
    def lookup_ticket(ticket_id: str) -> dict[str, str]:
        """Look up one ticket."""

        return {"ticket_id": ticket_id}

    manager = ApprovalManager(tmp_path / "state.sqlite3")
    pending = manager.create(prepared_deal(tools=[lookup_ticket]))
    granted = manager.grant(pending.approval_id, reviewer="ana")
    assert granted.token is not None

    with pytest.raises(CrupierError, match="runtime tool handlers"):
        manager.consume(granted.token)

    restored = manager.consume(granted.token, tools=[lookup_ticket])

    assert restored.request.tools == [lookup_ticket]


def test_pending_approval_expires_and_cannot_be_decided(tmp_path, monkeypatch):
    from crupier import approvals

    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(approvals, "_after", lambda seconds: past)
    manager = ApprovalManager(tmp_path / "state.sqlite3")
    pending = manager.create(prepared_deal(), ttl_s=10)

    assert manager.get(pending.approval_id).status == "expired"
    with pytest.raises(CrupierError, match="only pending"):
        manager.grant(pending.approval_id, reviewer="ana")


def test_invalid_tokens_and_non_approvable_routes_fail_closed(tmp_path):
    manager = ApprovalManager(tmp_path / "state.sqlite3")

    with pytest.raises(CrupierError, match="invalid"):
        manager.consume("bad-token")
    with pytest.raises(CrupierError, match="greater than zero"):
        manager.create(prepared_deal(), ttl_s=0)


def test_reviewer_verification_cannot_be_self_asserted(tmp_path):
    unverified = ApprovalManager(tmp_path / "unverified.sqlite3")
    pending = unverified.create(prepared_deal())
    granted = unverified.grant(pending.approval_id, reviewer="ana")
    assert granted.reviewer_verified is False

    invalid = ApprovalManager(
        tmp_path / "invalid.sqlite3",
        reviewer_verifier=lambda reviewer, context: {"verified": False},
    )
    pending = invalid.create(prepared_deal())
    with pytest.raises(CrupierError, match="could not be verified"):
        invalid.grant(pending.approval_id, reviewer="ana")
    assert invalid.get(pending.approval_id).status == "pending"


def test_remote_files_require_content_digest_and_payloads_are_strictly_serializable(tmp_path):
    manager = ApprovalManager(tmp_path / "state.sqlite3")
    remote = FileAsset(
        kind="pdf",
        name="contract.pdf",
        uri="https://files.example.test/contract.pdf",
    )
    with pytest.raises(CrupierError, match="metadata.sha256"):
        manager.create(prepared_deal(files=[remote]))

    remote.metadata["sha256"] = "a" * 64
    pending = manager.create(prepared_deal(files=[remote]))
    assert pending.status == "pending"

    unsupported = prepared_deal()
    unsupported.request.input = object()
    with pytest.raises(CrupierError, match="unsupported value type"):
        manager.create(unsupported)


def test_approval_resolution_scope_and_reviewer_inputs_fail_closed(tmp_path):
    manager = ApprovalManager(tmp_path / "state.sqlite3")

    with pytest.raises(CrupierError, match="scope"):
        manager.create(prepared_deal(), scope="tool")
    with pytest.raises(CrupierError, match="No approval"):
        manager.resolve("trc_missing")

    first = manager.create(prepared_deal())
    second = manager.create(prepared_deal())
    with pytest.raises(CrupierError, match="Multiple approvals"):
        manager.resolve(first.trace_id)
    with pytest.raises(CrupierError, match="greater than zero"):
        manager.grant(first.approval_id, reviewer="ana", ttl_s=0)
    with pytest.raises(CrupierError, match="cannot be empty"):
        manager.reject(second.approval_id, reviewer="", reason="")

    separate = ApprovalManager(tmp_path / "single.sqlite3")
    only = separate.create(prepared_deal())
    separate.reject(only.approval_id, reviewer="ana", reason="")
    assert separate.resolve(only.trace_id).approval_id == only.approval_id


def test_approval_detects_token_plan_and_tool_catalog_tampering(tmp_path):
    manager = ApprovalManager(tmp_path / "state.sqlite3")
    granted = manager.grant(
        manager.create(prepared_deal()).approval_id,
        reviewer="ana",
    )
    assert granted.token is not None
    altered = granted.token[:-1] + ("a" if granted.token[-1] != "a" else "b")
    with pytest.raises(CrupierError, match="token is invalid"):
        manager.consume(altered)

    current = manager.store.get("approval", granted.approval_id)
    payload = dict(current.payload)
    payload["plan_hash"] = "0" * 64
    manager.store.transition(
        kind="approval",
        record_id=granted.approval_id,
        expected_statuses={"granted"},
        status="granted",
        payload=payload,
        expires_at=current.expires_at,
        event="test_tamper",
    )
    with pytest.raises(CrupierError, match="plan hash"):
        manager.consume(granted.token)

    def lookup_ticket(ticket_id: str):
        return ticket_id

    def write_ticket(ticket_id: str):
        return ticket_id

    tool_manager = ApprovalManager(tmp_path / "tools.sqlite3")
    tool_grant = tool_manager.grant(
        tool_manager.create(prepared_deal(tools=[lookup_ticket])).approval_id,
        reviewer="ana",
    )
    assert tool_grant.token is not None
    with pytest.raises(CrupierError, match="tool catalog"):
        tool_manager.consume(tool_grant.token, tools=[write_ticket])


def test_reviewer_attestation_and_file_identity_validation(tmp_path):
    invalid_verifier = ApprovalManager(
        tmp_path / "verifier.sqlite3",
        reviewer_verifier=lambda reviewer, context: "verified",
    )
    pending = invalid_verifier.create(prepared_deal())
    with pytest.raises(CrupierError, match="return a dictionary"):
        invalid_verifier.grant(pending.approval_id, reviewer="ana")

    manager = ApprovalManager(tmp_path / "files.sqlite3")
    with pytest.raises(CrupierError, match="stable URI"):
        manager.create(
            prepared_deal(files=[FileAsset(kind="text", name="missing-uri")])
        )
    with pytest.raises(CrupierError, match="must exist"):
        manager.create(
            prepared_deal(
                files=[
                    FileAsset(
                        kind="text",
                        name="missing.txt",
                        uri=str(tmp_path / "missing.txt"),
                    )
                ]
            )
        )

    serializable = prepared_deal(
        files=[
            FileAsset(
                kind="image",
                name="inline.png",
                uri="data:image/png;base64,aGVsbG8=",
            )
        ]
    )
    serializable.request.input = {
        "path": Path("contract.txt"),
        "date": date(2026, 8, 2),
    }
    assert manager.create(serializable).status == "pending"


def test_approval_json_helpers_support_to_dict_and_missing_expiry():
    class Serializable:
        def to_dict(self):
            return {"at": date(2026, 8, 23)}

    assert approvals_module._jsonable(Serializable()) == {"at": "2026-08-23"}
    assert approvals_module._parse(None) == datetime.max.replace(tzinfo=UTC)

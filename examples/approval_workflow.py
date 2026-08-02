"""Freeze, inspect, and approve a sensitive route without provider calls."""

from __future__ import annotations

from tempfile import TemporaryDirectory

from _example_support import offline_client, print_route


def write_release_note(text: str) -> dict[str, int]:
    """Example side-effecting handler; this script never executes it."""

    return {"chars": len(text)}


def main() -> None:
    tool = {
        "name": "write_release_note",
        "description": "Write the approved release note.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "handler": write_release_note,
        "side_effects": True,
    }
    with TemporaryDirectory(prefix="crupier-approval-example-") as temporary:
        crupier = offline_client(
            project="approval-workflow",
            profile="agentic",
            allow=["openai:gpt-5.4-mini", "anthropic:claude-sonnet-4-6"],
            root=temporary,
        )
        dry_run = crupier.deal(
            "Draft and write the release note after review.",
            tools=[tool],
            dry_run=True,
            trace="summary",
        )
        prepared = crupier.prepare(
            "Draft and write the release note after review.",
            tools=[tool],
            dry_run=True,
        )
        pending = crupier.request_approval(prepared, ttl_s=900)
        granted = crupier.approvals.grant(
            pending.approval_id,
            reviewer="ana",
            ttl_s=300,
            reason="Route and rollback reviewed.",
        )

        if granted.token is None:
            raise RuntimeError("Approval token was not issued")
        print_route(
            "approval_workflow",
            dry_run,
            extra={
                "approval_status": granted.status,
                "approval_scope": granted.scope,
                "plan_hash_bound": bool(granted.plan_hash),
                "reviewer_verified": granted.reviewer_verified,
                "token_printed": False,
                "execution_note": (
                    "Use execute_approved(token, tools=[tool]) in the authorized process; "
                    "the token is one-use"
                ),
            },
        )
        crupier.close()


if __name__ == "__main__":
    main()

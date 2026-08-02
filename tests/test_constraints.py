import ast
from pathlib import Path

import pytest

from crupier.config import CrupierConfig
from crupier.constraints import (
    SUPPORTED_CONSTRAINTS,
    request_allows_parallel,
    validate_request_constraints,
)
from crupier.errors import CrupierRouteValidationError
from crupier.models import RequestEnvelope


def test_constraint_contract_accepts_supported_controls():
    warnings = validate_request_constraints(
        {
            "requires_tools": True,
            "requires_human_approval": True,
            "allow_parallel": False,
            "max_cost_usd": 0.1,
            "top_p": 0.9,
        },
        has_tools=True,
    )

    assert warnings == []


def test_orchestrator_candidate_limit_is_supported_in_strict_mode():
    warnings = validate_request_constraints(
        {
            "orchestrator_candidate_limit": 4,
            "strict_constraints": True,
        },
        has_tools=False,
    )

    assert warnings == []


def test_consumed_request_constraints_are_declared():
    source_root = Path(__file__).resolve().parents[1] / "src" / "crupier"
    consumed: set[str] = set()

    def is_constraint_mapping(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "constraints"
            or isinstance(node, ast.Name)
            and node.id == "constraints"
        )

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            owner: ast.expr | None = None
            key: str | None = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
            ):
                owner = node.func.value
                candidate = node.args[0]
                if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                    key = candidate.value
            elif isinstance(node, ast.Subscript):
                owner = node.value
                candidate = node.slice
                if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                    key = candidate.value

            if owner is not None and key is not None and is_constraint_mapping(owner):
                consumed.add(key)

    undeclared = consumed.difference(SUPPORTED_CONSTRAINTS)
    assert not undeclared, (
        "Request constraints consumed by the implementation must be declared in "
        f"SUPPORTED_CONSTRAINTS: {sorted(undeclared)}"
    )


def test_unknown_constraints_warn_or_fail_in_strict_mode():
    warnings = validate_request_constraints({"application_gate": True}, has_tools=False)

    assert len(warnings) == 1
    assert "'application_gate'" in warnings[0]
    assert "metadata" in warnings[0]

    with pytest.raises(CrupierRouteValidationError, match="application_gate"):
        validate_request_constraints(
            {"application_gate": True, "strict_constraints": True},
            has_tools=False,
        )


def test_boolean_constraints_require_boolean_values():
    with pytest.raises(CrupierRouteValidationError, match="allow_parallel"):
        validate_request_constraints({"allow_parallel": "no"}, has_tools=False)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "10"])
def test_extraction_limits_require_positive_integers(value):
    with pytest.raises(CrupierRouteValidationError, match="max_pdf_pages"):
        validate_request_constraints({"max_pdf_pages": value}, has_tools=False)


def test_requires_tools_rejects_an_empty_tool_catalog():
    with pytest.raises(CrupierRouteValidationError, match="requires at least one tool"):
        validate_request_constraints({"requires_tools": True}, has_tools=False)


def test_parallel_override_resolves_against_project_default():
    disabled = CrupierConfig.from_dict({"routing": {"allow_parallel": False}})
    enabled = CrupierConfig.from_dict({"routing": {"allow_parallel": True}})

    assert request_allows_parallel(disabled, RequestEnvelope(task="default")) is False
    assert (
        request_allows_parallel(
            disabled,
            RequestEnvelope(task="override", constraints={"allow_parallel": True}),
        )
        is False
    )
    assert (
        request_allows_parallel(
            enabled,
            RequestEnvelope(task="request-limit", constraints={"allow_parallel": False}),
        )
        is False
    )

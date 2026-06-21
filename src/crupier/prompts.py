"""Versioned prompt templates used by Crupier planners and executors."""

from __future__ import annotations

import json
from typing import Any


ORCHESTRATOR_ROUTE_PLAN_PROMPT_VERSION = "orchestrator.route_plan.v1"


def build_orchestrator_planning_prompt(payload: dict[str, Any]) -> str:
    return (
        f"Prompt-Version: {ORCHESTRATOR_ROUTE_PLAN_PROMPT_VERSION}\n"
        "You are Crupier's model-routing orchestrator. Return only one JSON object, no markdown.\n"
        "The JSON must match this shape:\n"
        "{\n"
        '  "strategy": "single|fallback|cascade|panel|fusion|critique_repair|local_first|delegate",\n'
        '  "steps": [{"role": "primary", "model": "provider:model"}],\n'
        '  "estimated_cost": {"estimated_usd": 0.0},\n'
        '  "estimated_latency_ms": 6000,\n'
        '  "reason": "short explanation without hidden reasoning",\n'
        '  "risk_level": "low|medium|high",\n'
        '  "summary": "short route summary"\n'
        "}\n"
        "Rules:\n"
        "- Use only candidate_models exactly as provided.\n"
        "- Do not invent capabilities, providers, prices, tools, or model IDs.\n"
        "- Valid role shapes: single/local_first use primary plus optional fallback; "
        "cascade uses primary plus optional escalation; fallback uses one fallback step with models; "
        "fusion requires panel, judge, final_writer; critique_repair requires generator, critic, repair; "
        "delegate uses one delegate step with an anchor model and optional params.task/mode/strategy.\n"
        "- Prefer single/cascade/fallback unless uncertainty or risk justifies panel/fusion/critique_repair.\n"
        "- Respect max_calls, modality requirements, strategy constraints, and deterministic_scores.\n"
        "- Do not include raw prompt/input content or chain-of-thought.\n"
        "Planning context JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_orchestrator_repair_prompt(payload: dict[str, Any], *, raw_text: str, error: str) -> str:
    return (
        f"Prompt-Version: {ORCHESTRATOR_ROUTE_PLAN_PROMPT_VERSION}\n"
        "Repair the previous Crupier RoutePlan. Return only one valid JSON object, no markdown.\n"
        f"Validation error: {error}\n"
        f"Previous output: {_truncate(raw_text, 4000)}\n"
        "Planning context JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."

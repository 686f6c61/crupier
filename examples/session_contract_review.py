"""Keep a compatible route across turns and replan when a file appears."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from _example_support import offline_client, print_route


def main() -> None:
    with TemporaryDirectory(prefix="crupier-session-example-") as temporary:
        crupier = offline_client(
            project="contract-session",
            profile="agentic",
            # Dos modelos estables y seleccionables: un modelo preview quedaría
            # siempre excluido por stable_models_only y no aportaría nada a la
            # decisión de la sesión. Ese filtro se demuestra en fail_closed_safety.py.
            allow=["openai:gpt-5.4-mini", "anthropic:claude-sonnet-4-6"],
            root=temporary,
        )
        session = crupier.session(
            mode="agentic",
            sticky=True,
            persist=True,
            max_turns=10,
            max_session_cost_usd=0.25,
        )
        session.deal(
            "Summarize ticket LEG-42.",
            input={"title": "Renewal review", "priority": "high"},
            dry_run=True,
        )
        session.deal(
            "Draft the internal review checklist.",
            dry_run=True,
        )
        contract = Path(temporary) / "contract.csv"
        contract.write_text(
            "clause,risk\nrenewal,high\ntermination,medium\n",
            encoding="utf-8",
        )
        replanned = session.deal(
            "Now review the attached contract table.",
            files=[contract],
            dry_run=True,
            trace="summary",
        )

        print_route(
            "session_contract_review",
            replanned,
            extra={
                "session_id_prefix": session.session_id.split("_", 1)[0],
                "turns": session.turns,
                "retained_route": session.route_history[1].reused,
                "route_reasons": ",".join(item.reason for item in session.route_history),
                "last_replanned": session.route_history[-1].reused is False,
            },
        )
        session.close()
        crupier.close()


if __name__ == "__main__":
    main()

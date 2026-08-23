import json
import os
import re
import subprocess
import sys
from pathlib import Path

from crupier.evals import EXPECTATION_KEYS

SHOWCASE_SCRIPTS = {
    "approval_workflow.py",
    "agentic_pr_review.py",
    "customer_support_triage.py",
    "drop_in_agent_boundary.py",
    "eval_feedback_loop.py",
    "fail_closed_safety.py",
    "multimodal_claim_review.py",
    "routing_tradeoffs.py",
    "sdk_dry_run.py",
    "session_contract_review.py",
    "shadow_canary_rollout.py",
    "specialized_operations.py",
    "workflow_operations_hub.py",
}


def test_public_examples_run_without_provider_keys(tmp_path):
    root = Path(__file__).resolve().parents[1]
    examples_dir = root / "examples"
    scripts = sorted(path for path in examples_dir.glob("*.py") if not path.name.startswith("_"))
    env = dict(os.environ)
    pythonpath = [str(root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    assert scripts

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

        assert result.returncode == 0, f"{script.name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "strategy=" in result.stdout, script.name
        assert "models=" in result.stdout, script.name
        assert not (tmp_path / ".crupier").exists(), script.name
        assert "sk-" not in result.stdout, script.name

        if script.name in SHOWCASE_SCRIPTS:
            assert "warnings=" in result.stdout, script.name
            assert "trace_errors=none" in result.stdout, script.name
            assert "planned_provider_calls=" in result.stdout, script.name
            assert "real_provider_calls=0" in result.stdout, script.name
            assert "Unknown request constraint" not in result.stdout, script.name


def test_examples_demonstrate_enforced_contracts_and_boundaries():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    outputs = _run_showcase_examples(examples_dir)

    assert "human_approval_required=True" in outputs["agentic_pr_review.py"]
    assert "approval_status=granted" in outputs["approval_workflow.py"]
    assert "human_review=True" in outputs["workflow_operations_hub.py"]
    assert "sequential_panel_execution" not in outputs["drop_in_agent_boundary.py"]
    assert "csv_rows_extracted=2" in outputs["multimodal_claim_review.py"]
    assert "last_replanned=True" in outputs["session_contract_review.py"]
    assert "retained_route=True" in outputs["session_contract_review.py"]
    assert "live_execution_gate=False" in outputs["shadow_canary_rollout.py"]
    assert "report_ok=True" in outputs["eval_feedback_loop.py"]
    assert "failed_checks=none" in outputs["eval_feedback_loop.py"]

    fail_closed = outputs["fail_closed_safety.py"]
    assert "canonical_key_on_custom_host=rejected" in fail_closed
    assert "custom_host_with_explicit_optin=accepted" in fail_closed
    assert "custom_host_without_https=rejected" in fail_closed
    assert "generic_endpoint_reusing_canonical_key=rejected" in fail_closed
    assert "rule_with_unsupported_effect=rejected" in fail_closed
    assert "well_formed_deny_rule=accepted" in fail_closed
    assert "openrouter_byok" in fail_closed
    assert "policy_rule:no_local_daemon_for_customer_data" in fail_closed
    assert "secret_printed_in_clear=False" in fail_closed
    assert "OpenRouter is optional BYOK and not enabled" in fail_closed
    assert _example_secret() not in fail_closed

    assert '"provider_calls": 0' not in (examples_dir / "sdk_dry_run.py").read_text(encoding="utf-8")
    assert '"provider_calls": 0' not in (examples_dir / "specialized_operations.py").read_text(
        encoding="utf-8"
    )


def test_example_readme_dataset_paths_exist():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    readme = (examples_dir / "README.md").read_text(encoding="utf-8")
    dataset_paths = re.findall(r"`([^`]+-eval\.json)`", readme)

    assert dataset_paths
    assert all((examples_dir / path).is_file() for path in dataset_paths)


def test_offline_example_client_loads_real_scoring_profiles():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        from _example_support import offline_client

        client = offline_client(project="profile-check", allow=["openai:gpt-5.4-mini"])
    finally:
        sys.path.remove(str(examples_dir))

    assert {"agentic", "cheap", "fast", "private", "quality", "research", "structured"} <= set(
        client.config.profiles
    )
    assert client.config.profiles["fast"].prefer == ["low_latency"]


def test_offline_example_client_declares_openrouter_as_disabled_byok():
    """El filtro openrouter_byok también salta si el proveedor falta, así que hay que
    comprobar la declaración: apagado, en modo BYOK y con su propia variable."""

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        from _example_support import offline_client

        client = offline_client(project="byok-check", allow=["openai:gpt-5.4-mini"])
    finally:
        sys.path.remove(str(examples_dir))

    openrouter = client.config.providers["openrouter"]

    assert openrouter.enabled is False
    assert openrouter.mode == "byok"
    assert openrouter.env_key == "OPENROUTER_API_KEY"


def test_routing_eval_dataset_only_uses_supported_expectations():
    """El dataset público no puede arrastrar claves que el runner ahora rechaza."""

    dataset_path = Path(__file__).resolve().parents[1] / "examples" / "routing-eval.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    used = {key for case in dataset["cases"] for key in case.get("expect", {})}

    assert used
    assert sorted(used - EXPECTATION_KEYS) == []


def test_operations_harness_keeps_partial_http_observations():
    """Si un endpoint HTTP revienta, el informe debe conservar lo ya observado."""

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        import live_operations_validation as harness
    finally:
        sys.path.remove(str(examples_dir))

    def boom(_client):
        raise harness.PartialCaseFailure(
            "embeddings",
            {"health": {"status": 200, "ok": True}, "models": {"status": 200, "count": 3}},
            json.JSONDecodeError("Expecting value", "", 0),
        )

    original = harness.CASE_RUNNERS["http"]
    harness.CASE_RUNNERS["http"] = boom
    try:
        payload = harness._run_case("http", object())
    finally:
        harness.CASE_RUNNERS["http"] = original

    assert payload["status"] == "fail"
    assert payload["failed_step"] == "embeddings"
    assert payload["error_type"] == "JSONDecodeError"
    assert payload["endpoints"]["health"] == {"status": 200, "ok": True}
    assert payload["endpoints"]["models"]["count"] == 3


def _example_secret() -> str:
    """Lee la credencial sintética del ejemplo para asertarla contra su propia salida."""

    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        from fail_closed_safety import EXAMPLE_SECRET

        return EXAMPLE_SECRET
    finally:
        sys.path.remove(str(examples_dir))


def _run_showcase_examples(examples_dir: Path) -> dict[str, str]:
    root = examples_dir.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), *(value for value in [env.get("PYTHONPATH")] if value)]
    )
    outputs: dict[str, str] = {}
    for name in sorted(SHOWCASE_SCRIPTS):
        result = subprocess.run(
            [sys.executable, str(examples_dir / name)],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        assert result.returncode == 0, f"{name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        outputs[name] = result.stdout
    return outputs

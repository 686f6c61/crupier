import json
from pathlib import Path

from crupier import Crupier
from crupier.adapters import AdapterResponse
from crupier.cli import main
from crupier.config import CrupierConfig, write_default_project
from crupier.project_audit import (
    build_adoption_patches,
    build_adoption_plan,
    build_project_doctor,
    format_code_comments_sarif,
    record_adoption_signoff,
    scan_code_comments,
)


class FakeAuditAdapter:
    provider = "openai"

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append({"model": model, "prompt": prompt, "mode": request.mode})
        text = self.outputs.pop(0) if self.outputs else "crupier-audit-ok"
        return AdapterResponse(
            text=text,
            usage={"input_tokens": 2, "output_tokens": 2},
            metadata={"provider": "openai", "model": model},
        )


def make_audit_client(tmp_path, *, outputs=None):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "audit-test", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.4-mini"]},
            "routing": {"default_strategy": "single", "allow_fusion": True, "allow_parallel": True},
            "profiles": {
                "agentic": {"prefer": ["tool_use", "coding"], "strategy": "single"},
                "fast": {"prefer": ["low_latency"], "strategy": "single"},
                "structured": {"prefer": ["structured_output"], "strategy": "single"},
                "research": {"prefer": ["consensus"], "strategy": "fusion"},
            },
        }
    )
    config.root = tmp_path
    return Crupier(config, adapters={"openai": FakeAuditAdapter(outputs)})


def write_small_eval_dataset(tmp_path):
    path = tmp_path / "audit-eval.json"
    path.write_text(
        json.dumps(
            {
                "name": "audit-smoke",
                "cases": [
                    {
                        "id": "fast_single",
                        "task": "Answer briefly.",
                        "mode": "fast",
                        "expect": {"strategy": "single", "max_models": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_project_audit_generates_human_reviews_and_code_comments(tmp_path):
    (tmp_path / "app.py").write_text(
        """
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(model="gpt-4o-mini", messages=[])
""",
        encoding="utf-8",
    )
    client = make_audit_client(tmp_path)
    dataset = write_small_eval_dataset(tmp_path)

    report = client.audit.run(dataset=dataset, real=False, include_code_comments=True)

    assert report.ok is True
    assert {review.id for review in report.route_reviews} == {
        "fast_short_answer",
        "structured_invoice",
        "agentic_code_change",
        "research_tradeoffs",
    }
    assert any(comment.title == "OpenAI integration point" for comment in report.code_comments)
    assert any(comment.title == "Hard-coded model choice" for comment in report.code_comments)
    assert any(check.id == "real_canaries" and check.status == "warn" for check in report.checks)


def test_project_audit_real_canaries_with_fake_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = make_audit_client(
        tmp_path,
        outputs=[
            "crupier-audit-ok",
            '{"name": "Ada", "total": 12.5}',
            '{"tool_calls":[{"name":"lookup_user","arguments":{"name":"Ada"}}]}',
            "usr_audit pro",
            "citrine",
            "red",
        ],
    )
    dataset = write_small_eval_dataset(tmp_path)

    report = client.audit.run(dataset=dataset, real=True, include_code_comments=False)

    assert report.ok is True
    assert [item["kind"] for item in report.real_canaries] == [
        "text_smoke",
        "structured",
        "tool",
        "text_file",
        "image",
    ]
    assert all(item["ok"] for item in report.real_canaries)
    assert len(client.adapters["openai"].calls) == 6


def test_scan_code_comments_detects_secret_like_literal_without_returning_source(tmp_path):
    path = tmp_path / "service.py"
    fake_secret = "s" + "k-test-secret-value"
    path.write_text(f'OPENAI_API_KEY = "{fake_secret}"\n', encoding="utf-8")

    comments = scan_code_comments(tmp_path)

    assert comments[0].title == "Possible inline credential"
    assert comments[0].priority == 1
    assert comments[0].category == "security"
    assert fake_secret not in json.dumps([comment.to_dict() for comment in comments])


def test_scan_code_comments_credential_pattern_avoids_redaction_regex_noise(tmp_path):
    (tmp_path / "parser.py").write_text(
        '\n'.join(
            [
                'fields = ("ta' + "s" + 'k-id", "tool-use-id")',
                'SECRET_PATTERN = r"' + "s" + 'k-[a-zA-Z0-9]{20,}"',
                'description = "Clave API con prefijo ' + "s" + 'k- (OpenAI, Stripe u otro)"',
            ]
        ),
        encoding="utf-8",
    )

    comments = scan_code_comments(tmp_path)

    assert comments == []


def test_scan_code_comments_ignores_generated_and_dependency_dirs(tmp_path):
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")
    generated = tmp_path / "build" / "lib" / "app.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")
    dependency = tmp_path / "node_modules" / "pkg" / "client.js"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("const client = new OpenAI();\n", encoding="utf-8")
    egg_info = tmp_path / "src" / "demo.egg-info" / "generated.py"
    egg_info.parent.mkdir(parents=True)
    egg_info.write_text("from anthropic import Anthropic\n", encoding="utf-8")

    comments = scan_code_comments(tmp_path)

    assert comments
    assert {comment.file for comment in comments} == {"app.py"}


def test_adoption_plan_recommends_compat_client_for_small_python_openai_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        """
from openai import OpenAI

client = OpenAI()
client.responses.create(model="gpt-4o-mini", input="hello")
""",
        encoding="utf-8",
    )

    plan = build_adoption_plan(tmp_path, project="demo")

    assert plan.ready is True
    assert plan.recommended_path == "compat_client"
    assert plan.confidence == "high"
    assert any(option.path == "compat_client" and option.status == "recommended" for option in plan.options)
    assert any("crupier.compat.openai.OpenAI" in item for item in plan.checklist)


def test_adoption_plan_blocks_inline_credentials(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    fake_secret = "s" + "k-test-secret-value"
    (tmp_path / "app.py").write_text(
        f"""
import openai
OPENAI_API_KEY = "{fake_secret}"
model = "gpt-4o-mini"
""",
        encoding="utf-8",
    )

    plan = build_adoption_plan(tmp_path, project="demo")

    assert plan.ready is False
    assert plan.recommended_path == "fix_blockers_first"
    assert plan.blockers
    assert fake_secret not in json.dumps(plan.to_dict())


def test_adoption_plan_warns_but_does_not_block_test_fixture_credentials(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_config.py").write_text("OPENAI_API_KEY = 'quoted-value'\n", encoding="utf-8")

    plan = build_adoption_plan(tmp_path, project="demo")
    sarif = format_code_comments_sarif(plan.code_comments)

    assert plan.ready is True
    assert not plan.blockers
    assert any(comment.title == "Credential-like test fixture" for comment in plan.code_comments)
    assert all(comment.priority == 3 for comment in plan.code_comments)
    assert any("Credential-like test fixtures" in warning for warning in plan.warnings)
    assert sarif["runs"][0]["results"][0]["level"] == "note"


def test_adoption_patches_suggest_compat_client_without_modifying_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    source = "from openai import OpenAI\nclient = OpenAI()\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    report = build_adoption_patches(tmp_path, project="demo", adoption_path="compat_client")

    assert report.ready is True
    assert report.adoption_path == "compat_client"
    assert report.patches
    assert "from crupier.compat.openai import OpenAI" in report.patches[0].diff
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == source


def test_adoption_patches_block_when_inline_credentials_exist(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    fake_secret = "s" + "k-test-secret-value"
    (tmp_path / "app.py").write_text(f'OPENAI_API_KEY = "{fake_secret}"\n', encoding="utf-8")

    report = build_adoption_patches(tmp_path, project="demo", adoption_path="compat_client")

    assert report.ready is False
    assert report.patches[0].status == "blocked"


def test_project_doctor_combines_adoption_audit_and_human_gates(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        """
from openai import OpenAI

client = OpenAI()
client.responses.create(model="gpt-4o-mini", input="hello")
""",
        encoding="utf-8",
    )
    client = make_audit_client(tmp_path)
    dataset = write_small_eval_dataset(tmp_path)

    report = build_project_doctor(client, paths=["app.py"], dataset=dataset, real=False)
    gates = {gate.id: gate.status for gate in report.gates}

    assert report.ready is True
    assert report.status == "ready"
    assert report.readiness_mode == "adoption"
    assert report.recommended_path == "compat_client"
    assert gates["adoption_path"] == "pass"
    assert gates["patch_suggestions"] == "pass"
    assert gates["real_canaries"] == "warn"
    assert gates["eval_history"] == "warn"
    assert gates["human_feedback"] == "warn"
    assert gates["adoption_signoff"] == "warn"
    assert report.review_contract["overall_status"] == "needs-human-review"
    assert report.review_contract["human_status"] == "needs_review"
    assert report.review_contract["must_not_auto_approve"] is True


def test_project_doctor_production_requires_human_evidence(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")
    client = make_audit_client(tmp_path)
    dataset = write_small_eval_dataset(tmp_path)

    report = build_project_doctor(client, paths=["app.py"], dataset=dataset, real=False, production=True)
    gates = {gate.id: gate.status for gate in report.gates}

    assert report.ready is False
    assert report.status == "blocked"
    assert report.readiness_mode == "production"
    assert gates["adoption_path"] == "pass"
    assert gates["real_canaries"] == "fail"
    assert gates["eval_history"] == "fail"
    assert gates["human_feedback"] == "fail"
    assert gates["adoption_signoff"] == "fail"
    assert report.review_contract["overall_status"] == "blocked"
    assert report.review_contract["human_blockers"]
    assert report.review_contract["requires_human_signoff"] is True


def test_project_doctor_tracks_human_adoption_signoff(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")
    client = make_audit_client(tmp_path)

    before = build_project_doctor(client, paths=["app.py"], real=False)
    before_gate = {gate.id: gate for gate in before.gates}["adoption_signoff"]

    record = record_adoption_signoff(
        tmp_path,
        project="audit-test",
        verdict="approve",
        reviewer_hash="dev-a",
        note="approved after handoff",
        adoption_path="compat_client",
    )
    after = build_project_doctor(client, paths=["app.py"], real=False)
    after_gate = {gate.id: gate for gate in after.gates}["adoption_signoff"]

    assert before_gate.status == "warn"
    assert record["verdict"] == "approve"
    assert after_gate.status == "pass"
    assert after.adoption_signoff_summary["status"] == "approved"


def test_project_doctor_blocks_rejected_adoption_signoff(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")
    client = make_audit_client(tmp_path)

    record_adoption_signoff(tmp_path, project="audit-test", verdict="reject", note="output was not useful enough")

    report = build_project_doctor(client, paths=["app.py"], real=False)
    gate = {gate.id: gate for gate in report.gates}["adoption_signoff"]

    assert report.ready is False
    assert report.status == "blocked"
    assert gate.status == "fail"
    assert "reject" in gate.summary


def test_project_doctor_requires_human_feedback_apply(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    record_status = main(
        [
            "--project",
            str(tmp_path),
            "feedback",
            "record",
            "--model",
            "openai:gpt-5.4-mini",
            "--mode",
            "fast",
            "--rating",
            "5",
            "--verdict",
            "accept",
            "--json",
        ]
    )
    capsys.readouterr()

    client = make_audit_client(tmp_path)
    before = build_project_doctor(client, paths=["app.py"], real=False)
    before_gate = {gate.id: gate for gate in before.gates}["human_feedback"]

    apply_status = main(["--project", str(tmp_path), "feedback", "apply", "--json"])
    apply_payload = json.loads(capsys.readouterr().out)

    after_client = make_audit_client(tmp_path)
    after = build_project_doctor(after_client, paths=["app.py"], real=False)
    after_gate = {gate.id: gate for gate in after.gates}["human_feedback"]

    assert record_status == 0
    assert before_gate.status == "warn"
    assert "not applied" in before_gate.summary
    assert "crupier feedback apply" in before_gate.actions[0]
    assert apply_status == 0
    assert apply_payload["updated"][0]["score_key"] == "human:fast"
    assert after_gate.status == "pass"
    assert after.applied_feedback_summary["pending_count"] == 0


def test_project_doctor_blocks_inline_credentials_without_leaking_secret(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    fake_secret = "s" + "k-test-secret-value"
    (tmp_path / "app.py").write_text(f'OPENAI_API_KEY = "{fake_secret}"\n', encoding="utf-8")
    client = make_audit_client(tmp_path)

    report = build_project_doctor(client, paths=["app.py"], real=False)
    gates = {gate.id: gate.status for gate in report.gates}

    assert report.ready is False
    assert report.status == "blocked"
    assert gates["adoption_blockers"] == "fail"
    assert gates["patch_suggestions"] == "fail"
    assert fake_secret not in json.dumps(report.to_dict())


def test_cli_audit_json_and_code_comments(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "app.py").write_text("import openai\nmodel = \"gpt-4o-mini\"\n", encoding="utf-8")

    audit_status = main(["--project", str(tmp_path), "audit", "--no-code-comments", "--json"])
    audit_payload = json.loads(capsys.readouterr().out)

    comments_status = main(["--project", str(tmp_path), "code", "comments", "app.py", "--json"])
    comments_payload = json.loads(capsys.readouterr().out)

    assert audit_status == 0
    assert audit_payload["ok"] is True
    assert comments_status == 0
    assert comments_payload["count"] >= 1


def test_cli_code_comments_ack_reviewed_satisfies_doctor_gate(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    comments_status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--ack-reviewed",
            "--reviewer-hash",
            "dev-a",
            "--note",
            "Reviewed integration path.",
            "--json",
        ]
    )
    comments_payload = json.loads(capsys.readouterr().out)

    doctor_status = main(["--project", str(tmp_path), "adopt", "doctor", "app.py", "--json"])
    doctor_payload = json.loads(capsys.readouterr().out)
    gates = {gate["id"]: gate for gate in doctor_payload["gates"]}

    assert comments_status == 0
    assert comments_payload["count"] >= 1
    assert comments_payload["acknowledged"]["comment_count"] == comments_payload["count"]
    assert comments_payload["review"]["pending_count"] == 0
    assert doctor_status == 0
    assert gates["programmer_code_comments"]["status"] == "pass"


def test_cli_code_comments_writes_review_comment_packet(tmp_path, capsys):
    write_default_project(tmp_path)
    source = "from openai import OpenAI\nclient = OpenAI()\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--write-review-comments",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    review_files = [Path(path) for path in payload["review_comment_files"]]
    markdown = next(path for path in review_files if path.suffix == ".md").read_text(encoding="utf-8")
    jsonl = next(path for path in review_files if path.suffix == ".jsonl").read_text(encoding="utf-8")

    assert status == 0
    assert payload["count"] >= 1
    assert len(review_files) == 2
    assert "# Crupier Code Review Comments" in markdown
    assert "app.py:1" in markdown
    assert "review_comment" in jsonl
    assert source.strip() not in markdown
    assert source.strip() not in jsonl


def test_cli_code_comments_writes_sarif_annotations(tmp_path, capsys):
    write_default_project(tmp_path)
    source = "from openai import OpenAI\nclient = OpenAI()\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--write-sarif",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    sarif_path = Path(payload["sarif_files"][0])
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    sarif_text = json.dumps(sarif)

    assert status == 0
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["tool"]["driver"]["name"] == "Crupier"
    assert sarif["runs"][0]["results"]
    assert sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "app.py"
    assert source.strip() not in sarif_text


def test_cli_code_comments_imports_granular_decisions(tmp_path, capsys):
    write_default_project(tmp_path)
    source = "from openai import OpenAI\nclient = OpenAI()\nmodel = \"gpt-4o-mini\"\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")

    template_status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--write-decisions-template",
            "--json",
        ]
    )
    template_payload = json.loads(capsys.readouterr().out)
    template_path = Path(template_payload["decision_template_files"][0])
    template_text = template_path.read_text(encoding="utf-8")
    decisions = json.loads(template_text)
    decisions["comments"][0]["verdict"] = "accepted"
    decisions["comments"][1]["verdict"] = "false_positive"
    decisions["comments"][2]["verdict"] = "needs_change"
    template_path.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import_status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--import-decisions",
            str(template_path),
            "--json",
        ]
    )
    import_payload = json.loads(capsys.readouterr().out)
    gates_status = main(["--project", str(tmp_path), "adopt", "doctor", "app.py", "--json"])
    gates_payload = json.loads(capsys.readouterr().out)
    gates = {gate["id"]: gate for gate in gates_payload["gates"]}

    assert template_status == 0
    assert import_status == 0
    assert gates_status == 0
    assert source.strip() not in template_text
    assert import_payload["imported_decisions"]["comment_count"] == 2
    assert import_payload["review"]["reviewed_count"] == 2
    assert import_payload["review"]["pending_count"] == 1
    assert gates["programmer_code_comments"]["status"] == "warn"

    for item in decisions["comments"]:
        item["verdict"] = "reviewed"
    template_path.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    complete_status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--import-decisions",
            str(template_path),
            "--json",
        ]
    )
    complete_payload = json.loads(capsys.readouterr().out)
    doctor_status = main(["--project", str(tmp_path), "adopt", "doctor", "app.py", "--json"])
    doctor_payload = json.loads(capsys.readouterr().out)
    gates = {gate["id"]: gate for gate in doctor_payload["gates"]}

    assert complete_status == 0
    assert complete_payload["review"]["pending_count"] == 0
    assert doctor_status == 0
    assert gates["programmer_code_comments"]["status"] == "pass"


def test_code_comments_ack_redacts_review_note(tmp_path, capsys):
    write_default_project(tmp_path)
    fake_secret = "s" + "k-test-secret-value"
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(
        [
            "--project",
            str(tmp_path),
            "code",
            "comments",
            "app.py",
            "--ack-reviewed",
            "--note",
            f"reviewed with {fake_secret}",
            "--json",
        ]
    )
    payload = capsys.readouterr().out

    assert status == 0
    assert fake_secret not in payload
    assert "[redacted]" in payload


def test_cli_adopt_plan_json_and_report(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "plan", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["recommended_path"] == "compat_client"
    assert payload["written_files"]
    assert (tmp_path / ".crupier" / "audits").exists()


def test_cli_adopt_plan_without_crupier_config_uses_package_name(tmp_path, capsys):
    (tmp_path / "package.json").write_text('{"name":"demo-node"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "plan", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["project"] == "demo-node"
    assert payload["recommended_path"] == "proxy"
    assert payload["written_files"]


def test_cli_adopt_patches_json_and_report(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "patches", "--path", "compat_client", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["adoption_path"] == "compat_client"
    assert "from crupier.compat.openai import OpenAI" in payload["patches"][0]["diff"]
    assert payload["written_files"]
    assert "from openai import OpenAI" in (tmp_path / "app.py").read_text(encoding="utf-8")


def test_cli_adopt_patches_without_crupier_config(tmp_path, capsys):
    (tmp_path / "package.json").write_text('{"name":"demo-node"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "patches", "--path", "compat_client", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["project"] == "demo-node"
    assert payload["adoption_path"] == "compat_client"
    assert "from crupier.compat.openai import OpenAI" in payload["patches"][0]["diff"]


def test_cli_adopt_doctor_json_and_report(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "doctor", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["status"] == "ready"
    assert payload["readiness_mode"] == "adoption"
    assert payload["recommended_path"] == "compat_client"
    assert any(gate["id"] == "project_audit" for gate in payload["gates"])
    assert payload["written_files"]
    assert (tmp_path / ".crupier" / "audits").exists()


def test_cli_adopt_doctor_without_crupier_config(tmp_path, capsys):
    (tmp_path / "package.json").write_text('{"name":"demo-node"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "doctor", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["project"] == "demo-node"
    assert payload["readiness_mode"] == "config_free_adoption"
    assert payload["recommended_path"] == "proxy"
    assert any(gate["id"] == "configuration" and gate["status"] == "warn" for gate in payload["gates"])
    assert any(gate["id"] == "adoption_signoff" and gate["status"] == "warn" for gate in payload["gates"])
    assert payload["written_files"]
    assert (tmp_path / ".crupier" / "audits").exists()


def test_cli_adopt_doctor_without_config_rejects_real_mode(tmp_path, capsys):
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "doctor", "--production", "--real", "--json"])
    captured = capsys.readouterr()

    assert status == 1
    assert "Config-free doctor only supports offline adoption review" in captured.err


def test_cli_adopt_package_without_crupier_config_writes_full_review_bundle(tmp_path, capsys):
    (tmp_path / "package.json").write_text('{"name":"demo-node"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "package", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["project"] == "demo-node"
    assert payload["readiness_mode"] == "config_free_adoption"
    assert payload["recommended_path"] == "proxy"
    assert payload["status"] == "needs-human-review"
    assert payload["review_contract"]["overall_status"] == "needs-human-review"
    assert payload["review_contract"]["human_open_gates"]
    assert payload["review_contract"]["must_not_auto_approve"] is True
    assert set(payload["artifact_groups"]) == {
        "adoption_package",
        "adoption_handoff",
        "adoption_patches",
        "code_comments",
        "code_comment_decisions",
        "code_review_comments",
        "code_sarif",
        "project_doctor",
    }
    assert all(Path(path).exists() for path in payload["written_files"])
    assert any(path.endswith(".jsonl") for path in payload["artifact_groups"]["code_review_comments"])
    assert any(path.endswith(".sarif") for path in payload["artifact_groups"]["code_sarif"])
    assert any("code_comment_decisions_" in Path(path).name for path in payload["artifact_groups"]["code_comment_decisions"])
    package_markdown = next(path for path in payload["artifact_groups"]["adoption_package"] if path.endswith(".md"))
    package_text = Path(package_markdown).read_text(encoding="utf-8")
    assert "## Review Contract" in package_text
    assert "Auto-approval blocked: true" in package_text
    assert "## Open First" in package_text
    assert "Programmer decision template" in package_text
    assert any("adopt signoff --verdict approve" in command for command in payload["suggested_commands"])


def test_cli_adopt_package_without_config_rejects_real_mode(tmp_path, capsys):
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "package", "--real", "--json"])
    captured = capsys.readouterr()

    assert status == 1
    assert "Config-free package only supports offline adoption review" in captured.err


def test_cli_adopt_handoff_json_and_report(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")
    main(["--project", str(tmp_path), "code", "comments", "app.py", "--write-report", "--write-review-comments", "--json"])
    capsys.readouterr()
    decisions_dir = tmp_path / ".crupier" / "feedback" / "decisions"
    decisions_dir.mkdir(parents=True)
    (decisions_dir / "human_decisions_test.json").write_text(
        '{"source_dry_run": false, "decisions":[]}\n',
        encoding="utf-8",
    )

    status = main(["--project", str(tmp_path), "adopt", "handoff", "app.py", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["status"] == "needs-human-review"
    assert payload["doctor"]["status"] == "ready"
    assert payload["review_contract"]["overall_status"] == "needs-human-review"
    assert payload["review_contract"]["must_not_auto_approve"] is True
    assert payload["human_signoff_checklist"]
    assert any("technically valid" in item for item in payload["human_signoff_checklist"])
    assert any("adoption signoff" in item.lower() for item in payload["human_signoff_checklist"])
    assert payload["required_human_actions"]
    assert any("feedback" in action.lower() or "verdict" in action.lower() for action in payload["required_human_actions"])
    assert any("adoption signoff" in action.lower() for action in payload["required_human_actions"])
    assert any("import-decisions" in command for command in payload["suggested_commands"])
    assert any("adopt signoff --verdict approve" in command for command in payload["suggested_commands"])
    assert "crupier feedback apply" not in payload["suggested_commands"]
    assert any("ack-reviewed" in command for command in payload["suggested_commands"])
    assert payload["artifacts"]["code_comment_reports"]
    assert payload["artifacts"]["code_review_comment_packets"]
    assert payload["artifacts"]["feedback_decision_templates"]
    assert payload["written_files"]
    assert (tmp_path / ".crupier" / "handoffs").exists()
    markdown_path = next(path for path in payload["written_files"] if path.endswith(".md"))
    markdown = open(markdown_path, encoding="utf-8").read()
    assert "## Review Contract" in markdown
    assert "## Human Signoff Checklist" in markdown
    assert "## Pending Programmer Code Comments" in markdown


def test_cli_adopt_handoff_without_crupier_config(tmp_path, capsys):
    (tmp_path / "package.json").write_text('{"name":"demo-node"}\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "handoff", "--write-report", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["project"] == "demo-node"
    assert payload["doctor"]["readiness_mode"] == "config_free_adoption"
    assert payload["doctor"]["recommended_path"] == "proxy"
    assert any("Initialize Crupier configuration" in item for item in payload["human_signoff_checklist"])
    assert any(gate["id"] == "configuration" and gate["status"] == "warn" for gate in payload["doctor"]["gates"])
    assert any(gate["id"] == "adoption_signoff" and gate["status"] == "warn" for gate in payload["doctor"]["gates"])
    assert any(command == "crupier init" for command in payload["suggested_commands"])
    assert payload["written_files"]
    assert (tmp_path / ".crupier" / "handoffs").exists()


def test_cli_adopt_signoff_records_rejection_without_leaking_secret(tmp_path, capsys):
    write_default_project(tmp_path)

    fake_secret = "s" + "k-test-secret-value"
    status = main(
        [
            "--project",
            str(tmp_path),
            "adopt",
            "signoff",
            "--verdict",
            "reject",
            "--reviewer-hash",
            "dev-a",
            "--note",
            f"not acceptable {fake_secret}",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    client = Crupier.from_project(tmp_path)
    report = build_project_doctor(client, production=False)
    gate = {item.id: item for item in report.gates}["adoption_signoff"]

    assert status == 0
    assert payload["verdict"] == "reject"
    assert payload["reviewer_hash"] == "dev-a"
    assert fake_secret not in json.dumps(payload)
    assert gate.status == "fail"
    assert (tmp_path / ".crupier" / "handoffs" / "signoffs.jsonl").exists()


def test_adopt_doctor_production_rejects_only_dry_run_human_feedback(tmp_path):
    write_default_project(tmp_path)
    client = Crupier.from_project(tmp_path)
    client.feedback.record(
        project=client.config.project.name,
        models=["openai:gpt-5.4-mini"],
        mode="fast",
        rating=5,
        verdict="accept",
        tags=["dry_run_source"],
    )
    client.feedback.apply_to_registry(client.registry)

    report = build_project_doctor(client, production=True)
    gate = next(gate for gate in report.gates if gate.id == "human_feedback")

    assert gate.status == "fail"
    assert "dry-run" in gate.summary
    assert report.feedback_summary["dry_run_source_count"] == 1
    assert report.feedback_summary["production_feedback_count"] == 0


def test_cli_adopt_doctor_production_blocks_missing_human_evidence(tmp_path, capsys):
    write_default_project(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("from openai import OpenAI\nclient = OpenAI()\n", encoding="utf-8")

    status = main(["--project", str(tmp_path), "adopt", "doctor", "--production", "--json"])
    payload = json.loads(capsys.readouterr().out)
    gates = {gate["id"]: gate["status"] for gate in payload["gates"]}

    assert status == 1
    assert payload["status"] == "blocked"
    assert payload["readiness_mode"] == "production"
    assert gates["real_canaries"] == "fail"
    assert gates["eval_history"] == "fail"
    assert gates["human_feedback"] == "fail"

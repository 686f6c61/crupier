import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError
import zipfile

from crupier.cli import main
from crupier.release import (
    ReleaseCheck,
    ReleaseCheckReport,
    _artifact_content_check,
    _default_config_check,
    _public_release_language_check,
    _sdist_install_smoke,
    _wheel_install_smoke,
    check_pypi_project_name,
    run_release_checks,
)


def write_release_project(root):
    (root / "src" / "crupier").mkdir(parents=True)
    (root / "src" / "crupier" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "crupier" / "py.typed").write_text("", encoding="utf-8")
    (root / "src" / "crupier" / "version.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (root / "README.md").write_text(
        "# Demo\n\n"
        "## Installation\n\n"
        "```bash\n"
        "pip install demo\n"
        "crupier init\n"
        "crupier verify\n"
        "crupier release check\n"
        "crupier release check --strict-public\n"
        "crupier release check --check-pypi-name\n"
        "```\n\n"
        "## Python SDK Quickstart\n\n"
        "```python\n"
        "from crupier import Crupier\n"
        "client = Crupier.from_project()\n"
        "result = client.deal('Route this', dry_run=True)\n"
        "print(result.route.strategy)\n"
        "```\n\n"
        + "Install and use this package.\n" * 40,
        encoding="utf-8",
    )
    (root / "CONTRIBUTING.md").write_text(
        "# Contributing\n\n"
        "Never commit provider keys.\n\n"
        "```bash\n"
        "python -m pytest\n"
        "crupier release check\n"
        "crupier release check --strict-public\n"
        "crupier release check --check-pypi-name\n"
        "crupier release check --verify-providers\n"
        "```\n",
        encoding="utf-8",
    )
    (root / "CODE_OF_CONDUCT.md").write_text(
        "# Code of Conduct\n\n"
        "## Expected Behavior\n\nBe respectful.\n\n"
        "## Unacceptable Behavior\n\nDo not harass people.\n\n"
        "## Enforcement\n\nMaintainers may moderate.\n",
        encoding="utf-8",
    )
    (root / "SECURITY.md").write_text(
        "# Security Policy\n\n"
        "## Scope\n\nProvider credential handling and release artifacts.\n\n"
        "## Reporting A Vulnerability\n\n"
        "Use GitHub private vulnerability reporting. Do not include:\n"
        "- API keys\n"
        "- .env\n"
        "- .crupier/\n\n"
        "## Supported Versions\n\nLatest 0.x release.\n\n"
        "## Secret Handling Expectations\n\nNever post provider keys.\n\n"
        "## Disclosure And Fix Process\n\nPatch before public disclosure.\n",
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## 0.1.0\n", encoding="utf-8")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "ISSUE_TEMPLATE").mkdir(parents=True)
    (root / ".github" / "PULL_REQUEST_TEMPLATE.md").write_text(
        "# Pull Request\n\n"
        "## Validation\n\n"
        "- Release/readiness\n"
        "- No API keys\n",
        encoding="utf-8",
    )
    (root / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").write_text(
        "name: Bug report\nbody:\n"
        "  - attributes:\n"
        "      label: Reproduction\n"
        "  - attributes:\n"
        "      label: Environment\n",
        encoding="utf-8",
    )
    (root / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").write_text(
        "name: Feature request\nbody:\n"
        "  - attributes:\n"
        "      label: Use case\n"
        "  - attributes:\n"
        "      label: Constraints\n",
        encoding="utf-8",
    )
    (root / ".github" / "ISSUE_TEMPLATE" / "config.yml").write_text(
        "blank_issues_enabled: true\n",
        encoding="utf-8",
    )
    (root / ".github" / "dependabot.yml").write_text(
        "version: 2\n"
        "updates:\n"
        "  - package-ecosystem: pip\n"
        "    directory: /\n"
        "    schedule:\n"
        "      interval: weekly\n"
        "  - package-ecosystem: github-actions\n"
        "    directory: /\n"
        "    schedule:\n"
        "      interval: weekly\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "name: CI\n"
        "permissions:\n"
        "  contents: read\n"
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v6\n"
        "      - uses: actions/setup-python@v6\n"
        "      - run: python -m pytest\n"
        "      - run: python -m ruff check src tests --select E9,F63,F7,F82\n"
        "      - run: python -m pip_audit --skip-editable --progress-spinner off\n"
        "      - run: crupier release check\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "publish.yml").write_text(
        "name: Publish\n"
        "permissions:\n"
        "  contents: read\n"
        "  id-token: write\n"
        "jobs:\n"
        "  publish:\n"
        "    environment: pypi\n"
        "    steps:\n"
        "      - uses: actions/checkout@v6\n"
        "      - uses: actions/setup-python@v6\n"
        "      - run: crupier release check --strict-public\n"
        "      - run: python -m ruff check src tests --select E9,F63,F7,F82\n"
        "      - run: python -m pip_audit --skip-editable --progress-spinner off\n"
        "      - run: python -m build --sdist --wheel --outdir dist\n"
        "      - uses: actions/upload-artifact@v6\n"
        "        with:\n"
        "          if-no-files-found: error\n"
        "      - uses: pypa/gh-action-pypi-publish@release/v1\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "demo"
version = "0.1.0"
description = "Demo package"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Demo" }]
keywords = ["ai", "llm", "routing"]
classifiers = [
  "Intended Audience :: Developers",
  "Operating System :: OS Independent",
  "Topic :: Software Development :: Libraries :: Python Modules",
  "Typing :: Typed",
]

[project.optional-dependencies]
openai = ["openai>=1"]
anthropic = ["anthropic>=0.40"]
google = ["google-genai>=1"]
ollama = ["ollama>=0.4"]
openrouter = ["openai>=1"]
pdf = ["pypdf>=5"]
all = ["openai>=1"]
dev = ["pytest>=8", "build>=1", "twine>=5", "pip-audit>=2", "ruff>=0.14"]

[project.scripts]
crupier = "crupier.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
crupier = ["py.typed"]
""",
        encoding="utf-8",
    )


def test_release_check_reports_ready_with_license_warning(tmp_path):
    write_release_project(tmp_path)

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check.status for check in report.checks}

    assert report.ok is True
    assert report.project == "demo"
    assert report.version == "0.1.0"
    assert checks["pyproject_metadata"] == "pass"
    assert checks["version_sync"] == "pass"
    assert checks["public_version"] == "pass"
    assert checks["typing_marker"] == "pass"
    assert checks["license"] == "warn"
    assert checks["project_urls"] == "warn"
    assert checks["default_config"] == "pass"
    assert report.build == {"skipped": True}


def test_release_check_passes_with_license_metadata(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'requires-python = ">=3.11"\n',
            'requires-python = ">=3.11"\nlicense = "MIT"\n',
        ),
        encoding="utf-8",
    )
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[project.urls]\nRepository = "https://github.com/crupier-dev/crupier"\nIssues = "https://github.com/crupier-dev/crupier/issues"\n',
        encoding="utf-8",
    )
    (tmp_path / "LICENSE").write_text("MIT License\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check.status for check in report.checks}

    assert report.ok is True
    assert checks["license"] == "pass"
    assert checks["project_urls"] == "pass"
    assert report.summary.get("warn", 0) == 0


def test_release_check_fails_without_strict_publish_workflow(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".github" / "workflows" / "publish.yml").write_text(
        "name: Publish\n"
        "permissions:\n"
        "  contents: read\n"
        "jobs:\n"
        "  publish:\n"
        "    steps:\n"
        "      - run: crupier release check\n",
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["publish_workflow"].status == "fail"
    assert "crupier release check --strict-public" in checks["publish_workflow"].evidence["missing_markers"]
    assert "actions/checkout@v6" in checks["publish_workflow"].evidence["missing_markers"]
    assert "actions/upload-artifact@v6" in checks["publish_workflow"].evidence["missing_markers"]
    assert "if-no-files-found: error" in checks["publish_workflow"].evidence["missing_markers"]
    assert "python -m ruff check src tests --select E9,F63,F7,F82" in checks["publish_workflow"].evidence["missing_markers"]
    assert "python -m pip_audit --skip-editable --progress-spinner off" in checks["publish_workflow"].evidence["missing_markers"]


def test_release_check_warns_without_contributing_guide(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / "CONTRIBUTING.md").unlink()

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["contributing"].status == "warn"


def test_release_check_warns_without_public_collaboration_templates(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / "CODE_OF_CONDUCT.md").unlink()
    (tmp_path / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").unlink()

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["community_files"].status == "warn"
    assert "CODE_OF_CONDUCT.md" in checks["community_files"].evidence["missing_files"]
    assert ".github/ISSUE_TEMPLATE/feature_request.yml" in checks["community_files"].evidence["missing_files"]


def test_release_check_warns_without_complete_security_policy(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / "SECURITY.md").write_text("# Security\n\nReport privately.\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["security_policy"].status == "warn"
    assert "## Supported Versions" in checks["security_policy"].evidence["missing_markers"]
    assert "## Disclosure And Fix Process" in checks["security_policy"].evidence["missing_markers"]


def test_release_check_warns_without_dependabot_or_minimal_ci_permissions(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".github" / "dependabot.yml").unlink()
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "name: CI\njobs:\n  test:\n    steps:\n      - run: python -m pytest\n",
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["dependency_updates"].status == "warn"
    assert checks["ci"].status == "warn"
    assert "contents: read" in checks["ci"].evidence["missing_markers"]
    assert "actions/checkout@v6" in checks["ci"].evidence["missing_markers"]
    assert "actions/setup-python@v6" in checks["ci"].evidence["missing_markers"]
    assert "python -m ruff check src tests --select E9,F63,F7,F82" in checks["ci"].evidence["missing_markers"]
    assert "python -m pip_audit --skip-editable --progress-spinner off" in checks["ci"].evidence["missing_markers"]
    assert checks["dependency_updates"].evidence["exists"] is False


def test_release_check_warns_when_dev_extra_missing_pip_audit(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(', "pip-audit>=2"', ""),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["optional_dependencies"].status == "warn"
    assert "dev:pip-audit" in checks["optional_dependencies"].evidence["missing"]
    assert checks["optional_dependencies"].evidence["dev_has_pip_audit"] is False


def test_release_check_warns_when_dev_extra_missing_ruff(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(', "ruff>=0.14"', ""),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["optional_dependencies"].status == "warn"
    assert "dev:ruff" in checks["optional_dependencies"].evidence["missing"]
    assert checks["optional_dependencies"].evidence["dev_has_ruff"] is False


def test_release_check_cli_can_require_provider_readiness(tmp_path, monkeypatch, capsys):
    write_release_project(tmp_path)
    (tmp_path / "crupier.toml").write_text(
        '[project]\nname = "demo"\n\n[models]\nallow = []\n',
        encoding="utf-8",
    )

    def fake_verify_report(*args, **kwargs):
        return {
            "ok": True,
            "openai_baseline": True,
            "providers": ["openai"],
            "summary": {"ready": 1},
            "items": [{"provider": "openai", "status": "ready"}],
        }

    monkeypatch.setattr("crupier.cli._build_verify_report", fake_verify_report)

    code = main(["--project", str(tmp_path), "release", "check", "--skip-build", "--verify-providers", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["provider_readiness"]["status"] == "pass"
    assert payload["ok"] is True


def test_release_check_cli_fails_when_provider_readiness_fails(tmp_path, monkeypatch, capsys):
    write_release_project(tmp_path)
    (tmp_path / "crupier.toml").write_text(
        '[project]\nname = "demo"\n\n[models]\nallow = []\n',
        encoding="utf-8",
    )

    def fake_verify_report(*args, **kwargs):
        return {
            "ok": False,
            "openai_baseline": True,
            "providers": ["openai"],
            "summary": {"failed": 1},
            "items": [{"provider": "openai", "status": "failed", "issues": ["Missing required environment variable."]}],
        }

    monkeypatch.setattr("crupier.cli._build_verify_report", fake_verify_report)

    code = main(["--project", str(tmp_path), "release", "check", "--skip-build", "--verify-providers", "--json"])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["provider_readiness"]["status"] == "fail"
    assert payload["ok"] is False


def test_release_check_strict_public_blocks_warnings_and_skipped_build(tmp_path, capsys):
    write_release_project(tmp_path)

    code = main(["--project", str(tmp_path), "release", "check", "--skip-build", "--strict-public", "--json"])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["strict_public"]["status"] == "fail"
    assert checks["strict_public"]["evidence"]["warning_ids"] == ["project_urls", "license"]
    assert checks["strict_public"]["evidence"]["build_skipped"] is True


def test_release_check_strict_public_passes_without_warnings_or_skipped_build(monkeypatch, capsys):
    def fake_release_checks(*args, **kwargs):
        return ReleaseCheckReport(
            project="demo",
            version="0.1.0",
            checks=[
                ReleaseCheck(
                    id="pyproject_metadata",
                    status="pass",
                    summary="pyproject.toml has required package metadata.",
                )
            ],
            build={"skipped": False, "ok": True},
        )

    monkeypatch.setattr("crupier.cli.run_release_checks", fake_release_checks)

    code = main(["release", "check", "--strict-public", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["strict_public"]["status"] == "pass"
    assert payload["ok"] is True


def test_wheel_install_smoke_runs_install_import_and_cli(tmp_path, monkeypatch):
    wheel = tmp_path / "demo-0.1.0-py3-none-any.whl"
    wheel.write_text("wheel", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append([str(part) for part in command])
        if command[:3] == [sys.executable, "-m", "venv"]:
            bin_dir = tmp_path / "install-smoke-venv" / ("Scripts" if sys.platform == "win32" else "bin")
            bin_dir.mkdir(parents=True)
            (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")
            (bin_dir / ("crupier.exe" if sys.platform == "win32" else "crupier")).write_text("", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "-c" in command:
            code = str(command[command.index("-c") + 1])
            if "Crupier.from_project" in code:
                return SimpleNamespace(returncode=0, stdout="single\nopenai:gpt-5.4-mini\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="0.1.0\n", stderr="")
        if str(command[0]).endswith(("crupier", "crupier.exe")):
            if "init" in command:
                project_dir = Path(command[command.index("--project") + 1])
                project_dir.mkdir(parents=True)
                (project_dir / "crupier.toml").write_text('host = "https://ollama.com/api"\n', encoding="utf-8")
                (project_dir / ".env.example").write_text(
                    "OLLAMA_HOST=https://ollama.com/api\n", encoding="utf-8"
                )
                (project_dir / ".gitignore").write_text(".env\n!.env.example\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="Created crupier.toml\n", stderr="")
            if "route" in command:
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"strategy": "single", "steps": [{"role": "primary", "model": "openai:gpt-5.4-mini"}]}\n',
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="usage: crupier\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("crupier.release.subprocess.run", fake_run)

    check, smoke = _wheel_install_smoke(wheel, tmp_path)

    assert check.status == "pass"
    assert smoke["ok"] is True
    assert smoke["import_version"] == "0.1.0"
    assert [step["name"] for step in smoke["steps"]] == [
        "create_venv",
        "install_wheel",
        "import_crupier",
        "cli_help",
        "init_project",
    ]
    assert any("pip" in call and "install" in call for call in calls)


def test_sdist_install_smoke_runs_install_import_and_cli(tmp_path, monkeypatch):
    sdist = tmp_path / "demo-0.1.0.tar.gz"
    sdist.write_text("sdist", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append([str(part) for part in command])
        if command[:3] == [sys.executable, "-m", "venv"]:
            bin_dir = tmp_path / "sdist-install-smoke-venv" / ("Scripts" if sys.platform == "win32" else "bin")
            bin_dir.mkdir(parents=True)
            (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")
            (bin_dir / ("crupier.exe" if sys.platform == "win32" else "crupier")).write_text("", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "-c" in command:
            code = str(command[command.index("-c") + 1])
            if "Crupier.from_project" in code:
                return SimpleNamespace(returncode=0, stdout="single\nopenai:gpt-5.4-mini\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="0.1.0\n", stderr="")
        if str(command[0]).endswith(("crupier", "crupier.exe")):
            if "init" in command:
                project_dir = Path(command[command.index("--project") + 1])
                project_dir.mkdir(parents=True)
                (project_dir / "crupier.toml").write_text('host = "https://ollama.com/api"\n', encoding="utf-8")
                (project_dir / ".env.example").write_text(
                    "OLLAMA_HOST=https://ollama.com/api\n", encoding="utf-8"
                )
                (project_dir / ".gitignore").write_text(".env\n!.env.example\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="Created crupier.toml\n", stderr="")
            if "route" in command:
                return SimpleNamespace(
                    returncode=0,
                    stdout='{"strategy": "single", "steps": [{"role": "primary", "model": "openai:gpt-5.4-mini"}]}\n',
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="usage: crupier\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("crupier.release.subprocess.run", fake_run)

    check, smoke = _sdist_install_smoke(sdist, tmp_path)

    assert check.status == "pass"
    assert smoke["ok"] is True
    assert smoke["import_version"] == "0.1.0"
    assert [step["name"] for step in smoke["steps"]] == [
        "create_venv",
        "install_sdist",
        "import_crupier",
        "cli_help",
        "init_project",
    ]
    assert any("pip" in call and "install" in call for call in calls)


def test_default_config_check_enforces_public_onboarding_defaults():
    check = _default_config_check()

    assert check.status == "pass"
    assert check.evidence["ollama_host"] == "https://ollama.com/api"
    assert check.evidence["store_prompts"] is False
    assert check.evidence["store_responses"] is False


def test_artifact_content_check_requires_typed_marker_and_blocks_local_artifacts(tmp_path):
    clean = tmp_path / "demo-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(clean, "w") as wheel:
        wheel.writestr("crupier/__init__.py", "")
        wheel.writestr("crupier/py.typed", "")
    sdist = tmp_path / "demo-0.1.0.tar.gz"
    source = tmp_path / "srcfile.txt"
    source.write_text("demo", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="demo-0.1.0/src/crupier/py.typed")
        archive.add(source, arcname="demo-0.1.0/.env.example")
        archive.add(source, arcname="demo-0.1.0/CONTRIBUTING.md")
        archive.add(source, arcname="demo-0.1.0/CODE_OF_CONDUCT.md")
        archive.add(source, arcname="demo-0.1.0/examples/sdk_dry_run.py")

    clean_check, clean_payload = _artifact_content_check([sdist, clean])

    assert clean_check.status == "pass"
    assert clean_payload["typed_marker_present"] is True
    assert clean_payload["env_example_present"] is True
    assert clean_payload["contributing_present"] is True
    assert clean_payload["code_of_conduct_present"] is True
    assert clean_payload["example_script_present"] is True

    bad = tmp_path / "bad-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(bad, "w") as wheel:
        wheel.writestr("crupier/__init__.py", "")
        wheel.writestr("crupier/py.typed", "")
        wheel.writestr(".env.example", "OPENAI_API_KEY=")
        wheel.writestr(".env", "OPENAI_API_KEY=secret")

    bad_check, bad_payload = _artifact_content_check([bad])

    assert bad_check.status == "fail"
    assert bad_payload["forbidden_count"] == 1


def test_release_check_detects_version_mismatch(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / "src" / "crupier" / "version.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["version_sync"].status == "fail"


def test_release_check_rejects_nonfinal_versions(tmp_path):
    write_release_project(tmp_path)
    nonfinal_version = "0.1.0rc1"
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('version = "0.1.0"', f'version = "{nonfinal_version}"'),
        encoding="utf-8",
    )
    (tmp_path / "src" / "crupier" / "version.py").write_text(
        f'__version__ = "{nonfinal_version}"\n',
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["version_sync"].status == "pass"
    assert checks["public_version"].status == "fail"


def test_public_release_language_check_rejects_alpha_beta_language(tmp_path):
    write_release_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nThis is an alpha release.\n", encoding="utf-8")

    check = _public_release_language_check(tmp_path)

    assert check.status == "fail"
    assert check.evidence["match_count"] == 1
    assert check.evidence["matches"][0]["match"] == "alpha"


def test_pypi_project_name_check_passes_when_name_is_available(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("crupier.release.urlopen", fake_urlopen)

    check = check_pypi_project_name("Demo_Name")

    assert check.status == "pass"
    assert check.evidence["normalized"] == "demo-name"
    assert check.evidence["http_status"] == 404


def test_pypi_project_name_check_fails_when_name_exists_without_allow_existing(monkeypatch):
    class FakeResponse:
        status = 200

        def close(self):
            self.closed = True

    monkeypatch.setattr("crupier.release.urlopen", lambda request, timeout: FakeResponse())

    check = check_pypi_project_name("demo")

    assert check.status == "fail"
    assert check.evidence["http_status"] == 200
    assert check.evidence["allow_existing"] is False


def test_pypi_project_name_check_can_allow_existing_owned_project(monkeypatch):
    class FakeResponse:
        status = 200

        def close(self):
            self.closed = True

    monkeypatch.setattr("crupier.release.urlopen", lambda request, timeout: FakeResponse())

    check = check_pypi_project_name("demo", allow_existing=True)

    assert check.status == "pass"
    assert check.evidence["allow_existing"] is True


def test_cli_release_check_can_include_pypi_name_check(tmp_path, monkeypatch, capsys):
    write_release_project(tmp_path)
    captured = {}

    def fake_pypi_check(project, *, allow_existing):
        captured["project"] = project
        captured["allow_existing"] = allow_existing
        return ReleaseCheck(id="pypi_project_name", status="pass", summary="PyPI project name is available.")

    monkeypatch.setattr("crupier.cli.check_pypi_project_name", fake_pypi_check)

    status = main(
        [
            "--project",
            str(tmp_path),
            "release",
            "check",
            "--skip-build",
            "--check-pypi-name",
            "--allow-existing-pypi-project",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}

    assert status == 0
    assert captured == {"project": "demo", "allow_existing": True}
    assert checks["pypi_project_name"]["status"] == "pass"


def test_cli_release_check_json(tmp_path, capsys):
    write_release_project(tmp_path)

    status = main(["--project", str(tmp_path), "release", "check", "--skip-build", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 0
    assert payload["ok"] is True
    assert payload["project"] == "demo"
    assert payload["summary"]["pass"] >= 10

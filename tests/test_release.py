import json
import os
from pathlib import Path
import subprocess
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
    _artifact_metadata_check,
    _copy_release_source,
    _default_config_check,
    _runtime_safety_defaults_check,
    _public_model_examples_check,
    _public_repository_surface_check,
    _public_release_language_check,
    _sdist_install_smoke,
    _sdist_examples_smoke,
    _wheel_install_smoke,
    check_project_urls_reachable,
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
        "crupier release check --strict-public --verify-project-urls --check-pypi-name\n"
        "crupier capabilities probe --provider google --apply\n"
        "crupier release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama\n"
        "```\n\n"
        "## Python SDK Quickstart\n\n"
        "```python\n"
        "from crupier import Crupier\n"
        "client = Crupier.from_project()\n"
        "result = client.deal('Route this', dry_run=True)\n"
        "print(result.route.strategy)\n"
        "```\n\n"
        "Images can route to native vision-capable models and execute through OpenAI, Anthropic Claude, "
        "Google Gemini, and Ollama adapters when the selected model supports image input.\n\n"
        + "Install and use this package.\n" * 40,
        encoding="utf-8",
    )
    (root / "CONTRIBUTING.md").write_text(
        "# Contributing\n\n"
        "Never commit provider keys.\n\n"
        "Dependabot security updates enabled and unpaused.\n\n"
        "Protect `main` and disallow force pushes before accepting public changes.\n\n"
        "```bash\n"
        "python -m pytest\n"
        "crupier release check\n"
        "crupier release check --strict-public\n"
        "crupier release check --strict-public --verify-project-urls --check-pypi-name\n"
        "GEMINI_API_KEY\n"
        "crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama\n"
        "```\n",
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
    (root / ".gitignore").write_text(
        ".env\n"
        ".env.*\n"
        "!.env.example\n"
        ".ruff_cache/\n"
        ".crupier/registry/models.json\n"
        ".crupier/registry/capability-cards/\n"
        ".crupier/traces/\n"
        ".crupier/audits/\n"
        ".crupier/code-comments/\n"
        ".crupier/evals/history/\n"
        ".crupier/evals/results/\n"
        ".crupier/evals/runs/\n"
        ".crupier/feedback/\n"
        ".crupier/handoffs/\n"
        ".crupier/packages/\n"
        ".venv/\n"
        "venv/\n"
        "env/\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        ".pytest_cache/\n"
        "*.egg-info/\n"
        "dist/\n"
        "build/\n",
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## 0.1.0\n", encoding="utf-8")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "ISSUE_TEMPLATE").mkdir(parents=True)
    (root / ".github" / "PULL_REQUEST_TEMPLATE.md").write_text(
        "# Pull Request\n\n"
        "## Validation\n\n"
        "- Release/readiness\n"
        "- `crupier release check --strict-public --verify-project-urls --check-pypi-name`\n"
        "- No API keys\n"
        "- `.crupier/`\n",
        encoding="utf-8",
    )
    (root / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").write_text(
        "name: Bug report\nbody:\n"
        "  - attributes:\n"
        "      label: Safety confirmation\n"
        "      options:\n"
        "        - label: I have removed API keys, provider responses, .env, and .crupier/ artifacts.\n"
        "          required: true\n"
        "  - attributes:\n"
        "      label: Reproduction\n"
        "  - attributes:\n"
        "      label: Environment\n",
        encoding="utf-8",
    )
    (root / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").write_text(
        "name: Feature request\nbody:\n"
        "  - attributes:\n"
        "      label: Safety confirmation\n"
        "      options:\n"
        "        - label: I have removed API keys, provider responses, .env, and .crupier/ artifacts.\n"
        "          required: true\n"
        "  - attributes:\n"
        "      label: Use case\n"
        "  - attributes:\n"
        "      label: Constraints\n",
        encoding="utf-8",
    )
    (root / ".github" / "ISSUE_TEMPLATE" / "config.yml").write_text(
        "blank_issues_enabled: false\n",
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
        "      - uses: actions/checkout@v7\n"
        "      - uses: actions/setup-python@v6\n"
        "      - run: python -m pytest\n"
        "      - run: python -m ruff check src tests --select E9,F63,F7,F82\n"
        "      - run: python -m pip_audit --skip-editable --progress-spinner off\n"
        "      - run: crupier release check\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "publish.yml").write_text(
        "name: Publish\n"
        "concurrency:\n"
        "  group: pypi-publish-${{ github.ref }}\n"
        "  cancel-in-progress: false\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "    inputs:\n"
        "      confirm_publish:\n"
        "        type: boolean\n"
        "jobs:\n"
        "  publish:\n"
        "    environment:\n"
        "      name: pypi\n"
        "      url: https://pypi.org/p/crupier\n"
        "    permissions:\n"
        "      contents: read\n"
        "      id-token: write\n"
        "    steps:\n"
        "      - uses: actions/checkout@v7\n"
        "        with:\n"
        "          fetch-depth: 0\n"
        "      - uses: actions/setup-python@v6\n"
        "      - name: Verify publish event matches package version\n"
        "        run: echo GITHUB_EVENT_NAME GITHUB_REF_NAME REQUESTED_VERSION CONFIRM_PUBLISH 'git\", \"fetch\", \"--quiet\", \"origin\", \"main:refs/remotes/origin/main\", \"--tags' 'git\", \"rev-parse\", \"origin/main' 'Publish commit does not match origin/main.'\n"
        "        env:\n"
        "          RELEASE_IS_DRAFT: false\n"
        "          RELEASE_IS_PRERELEASE: false\n"
        "          RELEASE_TARGET_COMMITISH: main\n"
        "      - run: echo 'Publishing from draft GitHub Releases is not allowed.'\n"
        "      - run: echo 'Publishing from prerelease GitHub Releases is not allowed.'\n"
        "      - run: echo 'is not the main branch; is not main'\n"
        "      - name: Release readiness check\n"
        "        env:\n"
        "          FIRST_PUBLIC_RELEASE_VERSION: \"0.1.0\"\n"
        "        run: |\n"
        "          crupier release check --strict-public --verify-project-urls --check-pypi-name --allow-existing-pypi-project\n"
        "      - run: python -m ruff check src tests --select E9,F63,F7,F82\n"
        "      - run: python -m pip_audit --skip-editable --progress-spinner off\n"
        "      - run: python -m build --sdist --wheel --outdir dist\n"
        "      - uses: actions/upload-artifact@v7\n"
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
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Software Development :: Libraries :: Python Modules",
  "Typing :: Typed",
]

[project.optional-dependencies]
openai = ["openai>=1"]
anthropic = ["anthropic>=0.40"]
google = ["google-genai>=1"]
ollama = []
openrouter = ["openai>=1"]
pdf = ["pypdf>=5"]
all = ["openai>=1", "anthropic>=0.40", "google-genai>=1", "pypdf>=5"]
dev = ["pytest>=8", "build>=1", "twine>=5", "pip-audit>=2", "ruff>=0.14", "trove-classifiers>=2026.6.1.19", "PyYAML>=6"]

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
    assert checks["public_repository_surface"] == "pass"
    assert checks["repository_gitignore"] == "pass"
    assert checks["typing_marker"] == "pass"
    assert checks["license"] == "warn"
    assert checks["readme_pypi_links"] == "pass"
    assert checks["public_markdown"] == "pass"
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


def test_release_check_fails_when_trove_classifier_is_unknown(tmp_path, monkeypatch):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '  "Typing :: Typed",\n',
            '  "Typing :: Typed",\n  "Topic :: Definitely Not A Real Classifier",\n',
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "trove_classifiers",
        SimpleNamespace(
            classifiers=[
                "Intended Audience :: Developers",
                "Operating System :: OS Independent",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.11",
                "Programming Language :: Python :: 3.12",
                "Programming Language :: Python :: 3.13",
                "Programming Language :: Python :: 3.14",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Software Development :: Libraries :: Python Modules",
                "Typing :: Typed",
            ]
        ),
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["pyproject_metadata"].status == "fail"
    assert checks["pyproject_metadata"].evidence["unknown_classifiers"] == [
        "Topic :: Definitely Not A Real Classifier"
    ]


def test_release_check_requires_tested_python_classifiers(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '  "Programming Language :: Python :: 3.14",\n',
            "",
        ),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["pyproject_metadata"].status == "fail"
    assert (
        "classifier:Programming Language :: Python :: 3.14" in checks["pyproject_metadata"].evidence["missing"]
    )
    assert checks["pyproject_metadata"].evidence["missing_classifiers"] == ["Programming Language :: Python :: 3.14"]


def test_public_repository_surface_check_blocks_internal_docs_and_extra_policy_file(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "internal.md").write_text("internal planning\n", encoding="utf-8")
    (tmp_path / "CODE_OF_CONDUCT.md").write_text("extra policy\n", encoding="utf-8")

    check = _public_repository_surface_check(tmp_path)

    assert check.status == "fail"
    assert check.evidence["forbidden_present"] == ["CODE_OF_CONDUCT.md", "docs"]


def test_release_check_fails_when_public_file_contains_provider_secret(tmp_path):
    write_release_project(tmp_path)
    secret = "sk-proj-" + "A" * 48
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + f"\nDo not ship {secret}\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["public_secret_scan"].status == "fail"
    finding = checks["public_secret_scan"].evidence["findings"][0]
    assert finding["path"] == "README.md"
    assert finding["pattern"] == "openai_api_key"
    assert isinstance(finding["line"], int)
    assert secret not in json.dumps(checks["public_secret_scan"].to_dict())


def test_release_check_ignores_local_env_file_without_git_tracking(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-proj-" + "A" * 48 + "\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["public_secret_scan"].status == "pass"
    assert ".env" not in checks["public_secret_scan"].evidence["checked_sample"]


def test_release_check_warns_when_repository_gitignore_is_incomplete(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["repository_gitignore"].status == "warn"
    assert ".env" in checks["repository_gitignore"].evidence["missing_entries"]
    assert ".crupier/traces/" in checks["repository_gitignore"].evidence["missing_entries"]
    assert "dist/" in checks["repository_gitignore"].evidence["missing_entries"]


def test_release_check_fails_when_readme_uses_relative_pypi_links(tmp_path):
    write_release_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n[Contributing](CONTRIBUTING.md)\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["readme_pypi_links"].status == "fail"
    assert checks["readme_pypi_links"].evidence["relative_links"] == ["CONTRIBUTING.md"]


def test_release_check_fails_when_readme_provider_gate_omits_google(tmp_path):
    write_release_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "crupier release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama",
            "crupier release check --verify-providers --provider openai --provider anthropic --provider ollama",
        ),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["readme"].status == "fail"
    assert (
        "crupier release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama"
        in checks["readme"].evidence["missing_markers"]
    )


def test_release_check_fails_when_public_markdown_has_broken_relative_link(tmp_path):
    write_release_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n[Missing guide](missing-guide.md)\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["public_markdown"].status == "fail"
    assert checks["public_markdown"].evidence["broken_links"] == [
        {"source": "README.md", "target": "missing-guide.md", "reason": "missing target"}
    ]


def test_release_check_fails_when_public_markdown_has_unbalanced_code_fence(tmp_path):
    write_release_project(tmp_path)
    contributing = tmp_path / "CONTRIBUTING.md"
    contributing.write_text(contributing.read_text(encoding="utf-8") + "\n```bash\npython -m pytest\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["public_markdown"].status == "fail"
    assert checks["public_markdown"].evidence["unbalanced_fences"] == {"CONTRIBUTING.md": 3}


def test_release_check_fails_when_public_github_yaml_is_invalid(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "name: [unterminated\n",
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is False
    assert checks["public_yaml"].status == "fail"
    assert checks["public_yaml"].evidence["failures"][0]["path"] == ".github/workflows/ci.yml"


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
    assert (
        "crupier release check --strict-public --verify-project-urls --check-pypi-name"
        in checks["publish_workflow"].evidence["missing_markers"]
    )
    assert "actions/checkout@v7" in checks["publish_workflow"].evidence["missing_markers"]
    assert "concurrency:" in checks["publish_workflow"].evidence["missing_markers"]
    assert "pypi-publish-${{ github.ref }}" in checks["publish_workflow"].evidence["missing_markers"]
    assert "cancel-in-progress: false" in checks["publish_workflow"].evidence["missing_markers"]
    assert "name: pypi" in checks["publish_workflow"].evidence["missing_markers"]
    assert "url: https://pypi.org/p/crupier" in checks["publish_workflow"].evidence["missing_markers"]
    assert "    permissions:" in checks["publish_workflow"].evidence["missing_markers"]
    assert "      contents: read" in checks["publish_workflow"].evidence["missing_markers"]
    assert "      id-token: write" in checks["publish_workflow"].evidence["missing_markers"]
    assert "fetch-depth: 0" in checks["publish_workflow"].evidence["missing_markers"]
    assert "actions/upload-artifact@v7" in checks["publish_workflow"].evidence["missing_markers"]
    assert "if-no-files-found: error" in checks["publish_workflow"].evidence["missing_markers"]
    assert "Verify publish event matches package version" in checks["publish_workflow"].evidence["missing_markers"]
    assert "GITHUB_EVENT_NAME" in checks["publish_workflow"].evidence["missing_markers"]
    assert "GITHUB_REF_NAME" in checks["publish_workflow"].evidence["missing_markers"]
    assert "REQUESTED_VERSION" in checks["publish_workflow"].evidence["missing_markers"]
    assert "CONFIRM_PUBLISH" in checks["publish_workflow"].evidence["missing_markers"]
    assert "RELEASE_IS_DRAFT" in checks["publish_workflow"].evidence["missing_markers"]
    assert "RELEASE_IS_PRERELEASE" in checks["publish_workflow"].evidence["missing_markers"]
    assert "RELEASE_TARGET_COMMITISH" in checks["publish_workflow"].evidence["missing_markers"]
    assert (
        'git", "fetch", "--quiet", "origin", "main:refs/remotes/origin/main", "--tags'
        in checks["publish_workflow"].evidence["missing_markers"]
    )
    assert 'git", "rev-parse", "origin/main' in checks["publish_workflow"].evidence["missing_markers"]
    assert "Publish commit does not match origin/main." in checks["publish_workflow"].evidence["missing_markers"]
    assert (
        "Publishing from draft GitHub Releases is not allowed."
        in checks["publish_workflow"].evidence["missing_markers"]
    )
    assert (
        "Publishing from prerelease GitHub Releases is not allowed."
        in checks["publish_workflow"].evidence["missing_markers"]
    )
    assert "is not the main branch" in checks["publish_workflow"].evidence["missing_markers"]
    assert "is not main" in checks["publish_workflow"].evidence["missing_markers"]
    assert "FIRST_PUBLIC_RELEASE_VERSION" in checks["publish_workflow"].evidence["missing_markers"]
    assert "--allow-existing-pypi-project" in checks["publish_workflow"].evidence["missing_markers"]
    assert "python -m ruff check src tests --select E9,F63,F7,F82" in checks["publish_workflow"].evidence["missing_markers"]
    assert "python -m pip_audit --skip-editable --progress-spinner off" in checks["publish_workflow"].evidence["missing_markers"]


def test_release_check_warns_without_contributing_guide(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / "CONTRIBUTING.md").unlink()

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["contributing"].status == "warn"


def test_release_check_warns_when_contributing_provider_gate_omits_google(tmp_path):
    write_release_project(tmp_path)
    contributing = tmp_path / "CONTRIBUTING.md"
    contributing.write_text(
        contributing.read_text(encoding="utf-8")
        .replace("GEMINI_API_KEY\n", "")
        .replace(
            "crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama",
            "crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider ollama",
        ),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["contributing"].status == "warn"
    assert "GEMINI_API_KEY" in checks["contributing"].evidence["missing_markers"]
    assert (
        "crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama"
        in checks["contributing"].evidence["missing_markers"]
    )


def test_release_check_warns_without_public_collaboration_templates(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").unlink()
    (tmp_path / ".github" / "PULL_REQUEST_TEMPLATE.md").unlink()
    (tmp_path / ".github" / "ISSUE_TEMPLATE" / "config.yml").write_text(
        "blank_issues_enabled: true\n",
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["community_files"].status == "warn"
    assert ".github/ISSUE_TEMPLATE/feature_request.yml" in checks["community_files"].evidence["missing_files"]
    assert ".github/PULL_REQUEST_TEMPLATE.md" in checks["community_files"].evidence["missing_files"]
    assert "blank_issues_enabled: false" in checks["community_files"].evidence["missing_markers"][
        ".github/ISSUE_TEMPLATE/config.yml"
    ]


def test_release_check_warns_when_pr_template_has_stale_release_gate(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / ".github" / "PULL_REQUEST_TEMPLATE.md").write_text(
        "# Pull Request\n\n"
        "## Validation\n\n"
        "- Release/readiness\n"
        "- `crupier release check --strict-public --verify-project-urls`\n"
        "- `crupier release check --check-pypi-name`\n"
        "- No API keys\n",
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["community_files"].status == "warn"
    assert (
        "crupier release check --strict-public --verify-project-urls --check-pypi-name"
        in checks["community_files"].evidence["missing_markers"][".github/PULL_REQUEST_TEMPLATE.md"]
    )


def test_release_check_warns_when_changelog_omits_packaged_version(tmp_path):
    write_release_project(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 0.0.9\n", encoding="utf-8")

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["changelog"].status == "warn"
    assert checks["changelog"].evidence["missing_markers"] == ["## 0.1.0"]


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
    assert "actions/checkout@v7" in checks["ci"].evidence["missing_markers"]
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


def test_release_check_warns_when_dev_extra_missing_pyyaml(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(', "PyYAML>=6"', ""),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["optional_dependencies"].status == "warn"
    assert "dev:PyYAML" in checks["optional_dependencies"].evidence["missing"]
    assert checks["optional_dependencies"].evidence["dev_has_pyyaml"] is False


def test_release_check_warns_when_provider_extra_missing_expected_dependency(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('google = ["google-genai>=1"]', "google = []"),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["optional_dependencies"].status == "warn"
    assert "google:google-genai" in checks["optional_dependencies"].evidence["missing"]
    assert checks["optional_dependencies"].evidence["missing_expected_dependencies"] == {
        "google": ["google-genai"]
    }


def test_release_check_warns_when_all_extra_missing_runtime_dependency(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'all = ["openai>=1", "anthropic>=0.40", "google-genai>=1", "pypdf>=5"]',
            'all = ["openai>=1", "anthropic>=0.40", "pypdf>=5"]',
        ),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["optional_dependencies"].status == "warn"
    assert "all:google-genai" in checks["optional_dependencies"].evidence["missing"]
    assert checks["optional_dependencies"].evidence["missing_expected_dependencies"] == {
        "all": ["google-genai"]
    }


def test_release_check_warns_when_ollama_extra_adds_unused_native_sdk(tmp_path):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace("ollama = []", 'ollama = ["ollama>=0.5"]'),
        encoding="utf-8",
    )

    report = run_release_checks(tmp_path, build=False)
    checks = {check.id: check for check in report.checks}

    assert report.ok is True
    assert checks["optional_dependencies"].status == "warn"
    assert "ollama:unneeded-sdk" in checks["optional_dependencies"].evidence["missing"]
    assert checks["optional_dependencies"].evidence["unneeded_ollama_sdk_dependencies"] == {
        "ollama": ["ollama>=0.5"]
    }


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


def test_release_check_cli_loads_env_file_for_provider_readiness(tmp_path, monkeypatch, capsys):
    write_release_project(tmp_path)
    (tmp_path / "crupier.toml").write_text(
        '[project]\nname = "demo"\n\n[models]\nallow = []\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "# local provider keys\n"
        "GEMINI_API_KEY=file-secret\n"
        "OPENAI_API_KEY=file-openai\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "existing-openai")
    observed: dict[str, str | None] = {}

    def fake_verify_report(*args, **kwargs):
        observed["gemini"] = os.environ.get("GEMINI_API_KEY")
        observed["openai"] = os.environ.get("OPENAI_API_KEY")
        return {
            "ok": True,
            "openai_baseline": True,
            "providers": ["openai", "google"],
            "summary": {"ready": 2},
            "items": [
                {"provider": "openai", "status": "ready"},
                {"provider": "google", "status": "ready"},
            ],
        }

    monkeypatch.setattr("crupier.cli._build_verify_report", fake_verify_report)

    code = main(
        [
            "--project",
            str(tmp_path),
            "--env-file",
            ".env",
            "release",
            "check",
            "--skip-build",
            "--verify-providers",
            "--provider",
            "google",
            "--json",
        ]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert observed == {"gemini": "file-secret", "openai": "existing-openai"}
    assert "file-secret" not in output
    assert "file-openai" not in output


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
    assert checks["strict_public"]["evidence"]["failure_ids"] == []
    assert checks["strict_public"]["evidence"]["warning_ids"] == ["project_urls", "license"]
    assert checks["strict_public"]["evidence"]["build_skipped"] is True


def test_release_check_strict_public_blocks_existing_failures(monkeypatch, capsys):
    def fake_release_checks(*args, **kwargs):
        return ReleaseCheckReport(
            project="demo",
            version="0.1.0",
            checks=[
                ReleaseCheck(
                    id="public_repository_surface",
                    status="fail",
                    severity="high",
                    summary="Public repository surface includes files that should stay out of the public repo.",
                )
            ],
            build={"skipped": False, "ok": True},
        )

    monkeypatch.setattr("crupier.cli.run_release_checks", fake_release_checks)

    code = main(["release", "check", "--strict-public", "--json"])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}
    assert checks["strict_public"]["status"] == "fail"
    assert checks["strict_public"]["evidence"]["failure_ids"] == ["public_repository_surface"]
    assert payload["ok"] is False


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
        if len(command) >= 4 and command[1:3] == ["-m", "crupier"]:
            return SimpleNamespace(returncode=0, stdout="crupier 0.1.0\n", stderr="")
        if "-c" in command:
            code = str(command[command.index("-c") + 1])
            if "Crupier.from_project" in code:
                return SimpleNamespace(returncode=0, stdout="single\nopenai:gpt-5.4-mini\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="0.1.0\n", stderr="")
        if str(command[0]).endswith(("crupier", "crupier.exe")):
            if "--version" in command:
                return SimpleNamespace(returncode=0, stdout="crupier 0.1.0\n", stderr="")
            if "init" in command:
                project_dir = Path(command[command.index("--project") + 1])
                project_dir.mkdir(parents=True)
                (project_dir / "crupier.toml").write_text(
                    '[providers.ollama]\nhost = "https://ollama.com/api"\n\n'
                    '[providers.openrouter]\nhost = "https://openrouter.ai/api/v1"\n',
                    encoding="utf-8",
                )
                (project_dir / ".env.example").write_text(
                    "OLLAMA_HOST=https://ollama.com/api\nOPENROUTER_API_KEY=\n", encoding="utf-8"
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
        "cli_version",
        "module_version",
        "init_project",
    ]
    assert any("pip" in call and "install" in call for call in calls)
    assert any("__all__" in " ".join(call) for call in calls)


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
        if len(command) >= 4 and command[1:3] == ["-m", "crupier"]:
            return SimpleNamespace(returncode=0, stdout="crupier 0.1.0\n", stderr="")
        if "-c" in command:
            code = str(command[command.index("-c") + 1])
            if "Crupier.from_project" in code:
                return SimpleNamespace(returncode=0, stdout="single\nopenai:gpt-5.4-mini\n", stderr="")
            return SimpleNamespace(returncode=0, stdout="0.1.0\n", stderr="")
        if str(command[0]).endswith(("crupier", "crupier.exe")):
            if "--version" in command:
                return SimpleNamespace(returncode=0, stdout="crupier 0.1.0\n", stderr="")
            if "init" in command:
                project_dir = Path(command[command.index("--project") + 1])
                project_dir.mkdir(parents=True)
                (project_dir / "crupier.toml").write_text(
                    '[providers.ollama]\nhost = "https://ollama.com/api"\n\n'
                    '[providers.openrouter]\nhost = "https://openrouter.ai/api/v1"\n',
                    encoding="utf-8",
                )
                (project_dir / ".env.example").write_text(
                    "OLLAMA_HOST=https://ollama.com/api\nOPENROUTER_API_KEY=\n", encoding="utf-8"
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
        "cli_version",
        "module_version",
        "init_project",
    ]
    assert any("pip" in call and "install" in call for call in calls)
    assert any("__all__" in " ".join(call) for call in calls)


def test_default_config_check_enforces_public_onboarding_defaults():
    check = _default_config_check()

    assert check.status == "pass"
    assert check.evidence["ollama_host"] == "https://ollama.com/api"
    assert check.evidence["store_prompts"] is False
    assert check.evidence["store_responses"] is False
    assert check.evidence["max_provider_retries"] == 1
    assert check.evidence["retry_backoff_seconds"] == 0.2


def test_runtime_safety_defaults_check_enforces_server_exposure_defaults():
    check = _runtime_safety_defaults_check()

    assert check.status == "pass"
    assert check.evidence["server_host_default"] == "127.0.0.1"
    assert check.evidence["server_allow_remote_default"] is False
    assert check.evidence["server_cors_origin_default"] is None


def _write_metadata_artifacts(tmp_path, metadata_text):
    wheel = tmp_path / "demo-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo-0.1.0.dist-info/METADATA", metadata_text)

    sdist = tmp_path / "demo-0.1.0.tar.gz"
    metadata_file = tmp_path / "PKG-INFO"
    metadata_file.write_text(metadata_text, encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(metadata_file, arcname="demo-0.1.0/PKG-INFO")
        archive.add(metadata_file, arcname="demo-0.1.0/src/demo.egg-info/PKG-INFO")
    return [sdist, wheel]


def test_artifact_metadata_check_validates_built_distribution_metadata(tmp_path):
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: demo\n"
        "Version: 0.1.0\n"
        "Summary: Demo package\n"
        "Requires-Python: >=3.11\n"
        "License-Expression: MIT\n"
        "Project-URL: Repository, https://github.com/example/demo\n"
        "Classifier: Intended Audience :: Developers\n"
        "Classifier: Typing :: Typed\n"
        "Provides-Extra: dev\n"
        "Provides-Extra: openai\n"
        "\n"
    )
    project = {
        "name": "demo",
        "version": "0.1.0",
        "description": "Demo package",
        "requires-python": ">=3.11",
        "license": "MIT",
        "urls": {"Repository": "https://github.com/example/demo"},
        "classifiers": ["Intended Audience :: Developers", "Typing :: Typed"],
        "optional-dependencies": {"dev": ["pytest>=8"], "openai": ["openai>=1"]},
    }

    check, payload = _artifact_metadata_check(_write_metadata_artifacts(tmp_path, metadata), project)

    assert check.status == "pass"
    assert payload["ok"] is True
    assert payload["failure_count"] == 0
    assert payload["expected_extras"] == ["dev", "openai"]
    assert [item["requires_python"] for item in payload["inspected"]] == [">=3.11", ">=3.11"]


def test_artifact_metadata_check_fails_when_metadata_omits_expected_field(tmp_path):
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: demo\n"
        "Version: 0.1.0\n"
        "Summary: Demo package\n"
        "License-Expression: MIT\n"
        "Project-URL: Repository, https://github.com/example/demo\n"
        "Classifier: Typing :: Typed\n"
        "Provides-Extra: dev\n"
        "\n"
    )
    project = {
        "name": "demo",
        "version": "0.1.0",
        "description": "Demo package",
        "requires-python": ">=3.11",
        "license": "MIT",
        "urls": {"Repository": "https://github.com/example/demo"},
        "classifiers": ["Typing :: Typed"],
        "optional-dependencies": {"dev": ["pytest>=8"]},
    }

    check, payload = _artifact_metadata_check(_write_metadata_artifacts(tmp_path, metadata), project)

    assert check.status == "fail"
    assert payload["ok"] is False
    assert payload["failure_count"] == 2
    assert {failure["field"] for failure in payload["failures"]} == {"Requires-Python"}


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
        archive.add(source, arcname="demo-0.1.0/examples/_example_support.py")
        archive.add(source, arcname="demo-0.1.0/examples/agentic_pr_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/customer_support_triage.py")
        archive.add(source, arcname="demo-0.1.0/examples/drop_in_agent_boundary.py")
        archive.add(source, arcname="demo-0.1.0/examples/model-compare-eval.json")
        archive.add(source, arcname="demo-0.1.0/examples/multimodal_claim_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/routing-eval.json")
        archive.add(source, arcname="demo-0.1.0/examples/sdk_dry_run.py")
        archive.add(source, arcname="demo-0.1.0/examples/workflow_operations_hub.py")

    clean_check, clean_payload = _artifact_content_check([sdist, clean])

    assert clean_check.status == "pass"
    assert clean_payload["typed_marker_present"] is True
    assert clean_payload["env_example_present"] is True
    assert clean_payload["contributing_present"] is True
    assert clean_payload["missing_examples"] == []
    assert clean_payload["forbidden_count"] == 0

    bad = tmp_path / "bad-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(bad, "w") as wheel:
        wheel.writestr("crupier/__init__.py", "")
        wheel.writestr("crupier/py.typed", "")
        wheel.writestr(".env.example", "OPENAI_API_KEY=")
        wheel.writestr(".env", "OPENAI_API_KEY=secret")
        wheel.writestr("crupier.toml", "[project]\nname = 'local-dev'\n")

    bad_check, bad_payload = _artifact_content_check([bad])

    assert bad_check.status == "fail"
    assert bad_payload["forbidden_count"] == 2

    internal_docs = tmp_path / "internal-docs-0.1.0.tar.gz"
    with tarfile.open(internal_docs, "w:gz") as archive:
        archive.add(source, arcname="demo-0.1.0/src/crupier/py.typed")
        archive.add(source, arcname="demo-0.1.0/.env.example")
        archive.add(source, arcname="demo-0.1.0/CONTRIBUTING.md")
        archive.add(source, arcname="demo-0.1.0/examples/_example_support.py")
        archive.add(source, arcname="demo-0.1.0/examples/agentic_pr_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/customer_support_triage.py")
        archive.add(source, arcname="demo-0.1.0/examples/drop_in_agent_boundary.py")
        archive.add(source, arcname="demo-0.1.0/examples/model-compare-eval.json")
        archive.add(source, arcname="demo-0.1.0/examples/multimodal_claim_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/routing-eval.json")
        archive.add(source, arcname="demo-0.1.0/examples/sdk_dry_run.py")
        archive.add(source, arcname="demo-0.1.0/examples/workflow_operations_hub.py")
        archive.add(source, arcname="demo-0.1.0/docs/crupier-roadmap.md")

    docs_check, docs_payload = _artifact_content_check([internal_docs])

    assert docs_check.status == "fail"
    assert docs_payload["forbidden_count"] == 1
    assert docs_payload["forbidden"][0].endswith("demo-0.1.0/docs/crupier-roadmap.md")

    packaged_tests = tmp_path / "packaged-tests-0.1.0.tar.gz"
    with tarfile.open(packaged_tests, "w:gz") as archive:
        archive.add(source, arcname="demo-0.1.0/src/crupier/py.typed")
        archive.add(source, arcname="demo-0.1.0/.env.example")
        archive.add(source, arcname="demo-0.1.0/CONTRIBUTING.md")
        archive.add(source, arcname="demo-0.1.0/examples/_example_support.py")
        archive.add(source, arcname="demo-0.1.0/examples/agentic_pr_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/customer_support_triage.py")
        archive.add(source, arcname="demo-0.1.0/examples/drop_in_agent_boundary.py")
        archive.add(source, arcname="demo-0.1.0/examples/model-compare-eval.json")
        archive.add(source, arcname="demo-0.1.0/examples/multimodal_claim_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/routing-eval.json")
        archive.add(source, arcname="demo-0.1.0/examples/sdk_dry_run.py")
        archive.add(source, arcname="demo-0.1.0/examples/workflow_operations_hub.py")
        archive.add(source, arcname="demo-0.1.0/tests/test_release.py")

    tests_check, tests_payload = _artifact_content_check([packaged_tests])

    assert tests_check.status == "fail"
    assert tests_payload["forbidden_count"] == 1
    assert tests_payload["forbidden"][0].endswith("demo-0.1.0/tests/test_release.py")


def test_copy_release_source_uses_clean_build_tree(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (source / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    (source / ".env.local").write_text("ANTHROPIC_API_KEY=secret\n", encoding="utf-8")
    (source / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    (source / "src").mkdir()
    (source / "src" / "demo.py").write_text("print('ok')\n", encoding="utf-8")
    for directory in [".git", ".crupier", ".venv", "dist", "build", "__pycache__", "src/demo.egg-info"]:
        (source / directory).mkdir(parents=True)
        (source / directory / "artifact.txt").write_text("artifact\n", encoding="utf-8")
    (source / "src" / "__pycache__").mkdir()
    (source / "src" / "__pycache__" / "demo.pyc").write_bytes(b"cache")

    copied = _copy_release_source(source, tmp_path / "copied")

    assert (copied / "pyproject.toml").exists()
    assert (copied / ".env.example").exists()
    assert (copied / "src" / "demo.py").exists()
    assert not (copied / ".env").exists()
    assert not (copied / ".env.local").exists()
    assert not (copied / ".git").exists()
    assert not (copied / ".crupier").exists()
    assert not (copied / ".venv").exists()
    assert not (copied / "dist").exists()
    assert not (copied / "build").exists()
    assert not (copied / "src" / "demo.egg-info").exists()
    assert not (copied / "src" / "__pycache__").exists()


def test_copy_release_source_prefers_git_tracked_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tracked_files = {
        "pyproject.toml": "[project]\nname = 'demo'\n",
        ".env.example": "OPENAI_API_KEY=\n",
        "README.md": "# Demo\n",
        "src/demo.py": "print('tracked')\n",
    }
    for relative, content in tracked_files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (source / "LOCAL_NOTES.md").write_text("private notes\n", encoding="utf-8")
    (source / "src" / "untracked.py").write_text("print('local')\n", encoding="utf-8")
    (source / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "add", *tracked_files], cwd=source, check=True, capture_output=True)

    copied = _copy_release_source(source, tmp_path / "copied")

    for relative in tracked_files:
        assert (copied / relative).exists()
    assert not (copied / "LOCAL_NOTES.md").exists()
    assert not (copied / "src" / "untracked.py").exists()
    assert not (copied / ".env").exists()
    assert not (copied / ".git").exists()


def test_sdist_examples_smoke_runs_packaged_examples_without_provider_keys(tmp_path, monkeypatch):
    sdist = tmp_path / "demo-0.1.0.tar.gz"
    source = tmp_path / "source.txt"
    source.write_text("demo", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="demo-0.1.0/src/crupier/__init__.py")
        archive.add(source, arcname="demo-0.1.0/examples/_example_support.py")
        archive.add(source, arcname="demo-0.1.0/examples/agentic_pr_review.py")
        archive.add(source, arcname="demo-0.1.0/examples/drop_in_agent_boundary.py")

    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    def fake_run(command, **kwargs):
        calls.append({"command": command, "env": kwargs["env"], "cwd": Path(kwargs["cwd"])})
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert "demo-0.1.0/src" in kwargs["env"]["PYTHONPATH"]
        return SimpleNamespace(returncode=0, stdout="strategy=single\nmodels=openai:gpt-5.4-mini\n", stderr="")

    monkeypatch.setattr("crupier.release.subprocess.run", fake_run)

    check, smoke = _sdist_examples_smoke(sdist, tmp_path)

    assert check.status == "pass"
    assert smoke["ok"] is True
    assert smoke["scripts"] == ["agentic_pr_review.py", "drop_in_agent_boundary.py"]
    assert len(calls) == 2
    assert not any((call["cwd"] / ".crupier").exists() for call in calls)


def test_sdist_examples_smoke_fails_when_packaged_example_does_not_emit_route(tmp_path, monkeypatch):
    sdist = tmp_path / "demo-0.1.0.tar.gz"
    source = tmp_path / "source.txt"
    source.write_text("demo", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="demo-0.1.0/src/crupier/__init__.py")
        archive.add(source, arcname="demo-0.1.0/examples/broken.py")

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="hello\n", stderr="")

    monkeypatch.setattr("crupier.release.subprocess.run", fake_run)

    check, smoke = _sdist_examples_smoke(sdist, tmp_path)

    assert check.status == "fail"
    assert smoke["ok"] is False
    assert smoke["steps"][0]["ok"] is False


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


def test_public_model_examples_check_rejects_stale_public_model_examples(tmp_path):
    write_release_project(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\ncrupier smoke --model openai:gpt-4o-mini\n", encoding="utf-8")
    cli = tmp_path / "src" / "crupier" / "cli.py"
    cli.write_text('help="Force model, e.g. openai:gpt-4.1-mini"\n', encoding="utf-8")

    check = _public_model_examples_check(tmp_path)

    assert check.status == "fail"
    assert check.evidence["match_count"] == 2
    assert {match["model"] for match in check.evidence["matches"]} == {"gpt-4o-mini", "gpt-4.1-mini"}


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


def test_project_urls_reachable_check_passes_for_public_urls(tmp_path, monkeypatch):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[project.urls]\nRepository = "https://github.com/example/demo"\nChangelog = "https://github.com/example/demo/blob/main/CHANGELOG.md"\n',
        encoding="utf-8",
    )

    class FakeResponse:
        status = 200
        url = "https://github.com/example/demo"

        def close(self):
            self.closed = True

    monkeypatch.setattr("crupier.release.urlopen", lambda request, timeout: FakeResponse())

    check = check_project_urls_reachable(tmp_path)

    assert check.status == "pass"
    assert len(check.evidence["checked"]) == 2
    assert check.evidence["failures"] == []


def test_project_urls_reachable_check_fails_for_unreachable_url(tmp_path, monkeypatch):
    write_release_project(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8")
        + '\n[project.urls]\nRepository = "https://github.com/example/missing"\n',
        encoding="utf-8",
    )

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("crupier.release.urlopen", fake_urlopen)

    check = check_project_urls_reachable(tmp_path)

    assert check.status == "fail"
    assert check.evidence["failures"][0]["http_status"] == 404


def test_cli_release_check_can_verify_project_urls(tmp_path, monkeypatch, capsys):
    write_release_project(tmp_path)
    captured = {}

    def fake_project_urls_check(root):
        captured["root"] = str(root)
        return ReleaseCheck(
            id="project_urls_reachable",
            status="pass",
            summary="Project URLs are reachable.",
        )

    monkeypatch.setattr("crupier.cli.check_project_urls_reachable", fake_project_urls_check)

    status = main(
        [
            "--project",
            str(tmp_path),
            "release",
            "check",
            "--skip-build",
            "--verify-project-urls",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    checks = {check["id"]: check for check in payload["checks"]}

    assert status == 0
    assert captured == {"root": str(tmp_path)}
    assert checks["project_urls_reachable"]["status"] == "pass"


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

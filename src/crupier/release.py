"""Release readiness checks for the local package."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import DEFAULT_ENV_EXAMPLE, DEFAULT_TOML, OLLAMA_CLOUD_HOST, CrupierConfig

_FINAL_PUBLIC_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_NON_FINAL_RELEASE_PATTERNS = [
    re.compile(r"\b\d+\.\d+\.\d+(?:a|b|alpha|beta|rc)\d*\b", re.IGNORECASE),
    re.compile(r"\b(?:alpha|beta|pre-release|prerelease|release candidate)\b", re.IGNORECASE),
    re.compile(r"Development Status :: [34]\b"),
]
_PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/{project}/json"


@dataclass(slots=True)
class ReleaseCheck:
    id: str
    status: str
    summary: str
    severity: str = "info"
    evidence: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReleaseCheckReport:
    project: str
    version: str | None
    checks: list[ReleaseCheck]
    build: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for check in self.checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project": self.project,
            "version": self.version,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "build": self.build,
        }


def run_release_checks(root: str | Path, *, build: bool = True) -> ReleaseCheckReport:
    root_path = Path(root).resolve()
    pyproject_path = root_path / "pyproject.toml"
    data = _read_pyproject(pyproject_path)
    project = data.get("project", {}) if isinstance(data, dict) else {}
    name = str(project.get("name", "unknown"))
    version = str(project.get("version")) if project.get("version") else None
    checks = [
        _pyproject_check(pyproject_path, project),
        _project_urls_check(project),
        _version_sync_check(root_path, version),
        _public_version_check(version),
        _public_release_language_check(root_path),
        _typed_marker_check(root_path, data),
        _readme_check(root_path, project),
        _license_check(root_path, project),
        _contributing_check(root_path),
        _community_files_check(root_path),
        _security_policy_check(root_path),
        _changelog_check(root_path),
        _ci_check(root_path),
        _dependency_updates_check(root_path),
        _publish_workflow_check(root_path),
        _script_check(project),
        _optional_dependencies_check(project),
        _default_config_check(),
    ]
    build_result: dict[str, Any] = {"skipped": not build}
    if build:
        build_checks, build_result = _build_distribution_checks(root_path)
        checks.extend(build_checks)
    return ReleaseCheckReport(project=name, version=version, checks=checks, build=build_result)


def check_pypi_project_name(
    project: str,
    *,
    allow_existing: bool = False,
    timeout_seconds: float = 10.0,
) -> ReleaseCheck:
    """Check whether the configured PyPI project name is available or already claimed."""

    normalized = _normalize_pypi_project_name(project)
    if not normalized or normalized == "unknown":
        return ReleaseCheck(
            id="pypi_project_name",
            status="fail",
            severity="high",
            summary="PyPI project name could not be checked because [project].name is missing.",
            evidence={"project": project, "normalized": normalized},
            actions=["Set [project].name before checking PyPI publication readiness."],
        )

    url = _PYPI_PROJECT_JSON_URL.format(project=quote(normalized, safe=""))
    request = Request(url, headers={"User-Agent": f"crupier-release-check/{normalized}"})
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        if exc.code == 404:
            return ReleaseCheck(
                id="pypi_project_name",
                status="pass",
                severity="high",
                summary="PyPI project name is currently available for a first public upload.",
                evidence={"project": project, "normalized": normalized, "url": url, "http_status": 404},
            )
        return ReleaseCheck(
            id="pypi_project_name",
            status="warn",
            severity="medium",
            summary="PyPI project name check returned an unexpected HTTP status.",
            evidence={"project": project, "normalized": normalized, "url": url, "http_status": exc.code},
            actions=["Retry the PyPI name check before publishing."],
        )
    except (OSError, TimeoutError, URLError) as exc:
        return ReleaseCheck(
            id="pypi_project_name",
            status="warn",
            severity="medium",
            summary="PyPI project name check could not reach PyPI.",
            evidence={
                "project": project,
                "normalized": normalized,
                "url": url,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
            actions=["Retry with network access before publishing."],
        )

    try:
        status = int(getattr(response, "status", getattr(response, "code", 200)))
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    if status == 200:
        ok = allow_existing
        return ReleaseCheck(
            id="pypi_project_name",
            status="pass" if ok else "fail",
            severity="high",
            summary="PyPI project name already exists and existing projects are allowed for this check."
            if ok
            else "PyPI project name already exists.",
            evidence={
                "project": project,
                "normalized": normalized,
                "url": url,
                "http_status": status,
                "allow_existing": allow_existing,
            },
            actions=[
                "For a first public release, choose an available name or publish from the account that owns this PyPI project."
            ]
            if not ok
            else [],
        )

    return ReleaseCheck(
        id="pypi_project_name",
        status="warn",
        severity="medium",
        summary="PyPI project name check returned an unexpected response.",
        evidence={"project": project, "normalized": normalized, "url": url, "http_status": status},
        actions=["Retry the PyPI name check before publishing."],
    )


def _normalize_pypi_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip().lower())


def _read_pyproject(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _pyproject_check(path: Path, project: dict[str, Any]) -> ReleaseCheck:
    missing = [
        field
        for field in ["name", "version", "description", "readme", "requires-python", "authors", "keywords", "classifiers"]
        if not project.get(field)
    ]
    classifiers = project.get("classifiers", [])
    classifier_set = set(classifiers) if isinstance(classifiers, list) else set()
    required_classifiers = {
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    }
    missing_classifiers = sorted(required_classifiers - classifier_set)
    missing.extend(f"classifier:{classifier}" for classifier in missing_classifiers)
    return ReleaseCheck(
        id="pyproject_metadata",
        status="pass" if path.exists() and not missing else "fail",
        severity="high",
        summary="pyproject.toml has required package metadata."
        if path.exists() and not missing
        else "pyproject.toml is missing required package metadata.",
        evidence={"path": str(path), "missing": missing, "missing_classifiers": missing_classifiers},
        actions=["Fill required [project] metadata before publishing."] if missing or not path.exists() else [],
    )


def _project_urls_check(project: dict[str, Any]) -> ReleaseCheck:
    urls = project.get("urls", {}) if isinstance(project.get("urls"), dict) else {}
    placeholders: list[str] = []
    valid: dict[str, str] = {}
    for label, value in urls.items():
        text = str(value)
        lowered = text.lower()
        if not text.startswith(("https://", "http://")):
            placeholders.append(str(label))
            continue
        if any(token in lowered for token in ["example.com", "todo", "replace-me", "your-org", "yourname"]):
            placeholders.append(str(label))
            continue
        valid[str(label)] = text
    has_project_home = any(label.lower() in {"homepage", "repository", "documentation"} for label in valid)
    ok = bool(valid) and has_project_home and not placeholders
    return ReleaseCheck(
        id="project_urls",
        status="pass" if ok else "warn",
        severity="medium",
        summary="Project URLs are present for public package consumers."
        if ok
        else "Project URLs are missing or incomplete for public package consumers.",
        evidence={"urls": urls, "valid": valid, "placeholders": placeholders},
        actions=[
            "Add real [project.urls] entries such as Repository, Documentation, Changelog, and Issues once the public repository exists."
        ]
        if not ok
        else [],
    )


def _version_sync_check(root: Path, pyproject_version: str | None) -> ReleaseCheck:
    version_path = root / "src" / "crupier" / "version.py"
    module_version = None
    if version_path.exists():
        namespace: dict[str, Any] = {}
        exec(version_path.read_text(encoding="utf-8"), namespace)
        module_version = namespace.get("__version__")
    ok = bool(pyproject_version and module_version and pyproject_version == module_version)
    return ReleaseCheck(
        id="version_sync",
        status="pass" if ok else "fail",
        severity="high",
        summary="pyproject version matches crupier.__version__." if ok else "Package versions are not synchronized.",
        evidence={"pyproject": pyproject_version, "module": module_version},
        actions=["Keep [project].version and src/crupier/version.py in sync."] if not ok else [],
    )


def _public_version_check(version: str | None) -> ReleaseCheck:
    ok = bool(version and _FINAL_PUBLIC_VERSION_RE.fullmatch(version))
    return ReleaseCheck(
        id="public_version",
        status="pass" if ok else "fail",
        severity="high",
        summary="Package version is a final public release version."
        if ok
        else "Package version is not a final public release version.",
        evidence={"version": version, "expected_shape": "X.Y.Z"},
        actions=["Use a final public version such as 0.1.0; do not publish non-final, development, or local builds."]
        if not ok
        else [],
    )


def _public_release_language_check(root: Path) -> ReleaseCheck:
    public_paths = [
        root / "pyproject.toml",
        root / "README.md",
        root / "CHANGELOG.md",
        root / "docs" / "crupier-publishing.md",
    ]
    matches: list[dict[str, Any]] = []
    for path in public_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in _NON_FINAL_RELEASE_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                matches.append(
                    {
                        "path": str(path.relative_to(root)),
                        "line": line,
                        "match": match.group(0),
                    }
                )
    ok = not matches
    return ReleaseCheck(
        id="public_release_language",
        status="pass" if ok else "fail",
        severity="high",
        summary="Public package metadata and release docs describe a final 0.1.0 release."
        if ok
        else "Public package metadata or release docs still describe a non-final release.",
        evidence={"matches": matches[:50], "match_count": len(matches)},
        actions=["Remove alpha, beta, pre-release, release-candidate, or non-final version language before publishing."]
        if not ok
        else [],
    )


def _typed_marker_check(root: Path, data: dict[str, Any]) -> ReleaseCheck:
    marker = root / "src" / "crupier" / "py.typed"
    setuptools = data.get("tool", {}).get("setuptools", {}) if isinstance(data.get("tool"), dict) else {}
    package_data = setuptools.get("package-data", {}) if isinstance(setuptools, dict) else {}
    crupier_data = package_data.get("crupier", []) if isinstance(package_data, dict) else []
    declared = "py.typed" in crupier_data
    ok = marker.exists() and declared
    return ReleaseCheck(
        id="typing_marker",
        status="pass" if ok else "fail",
        severity="medium",
        summary="py.typed is packaged for type-aware work environments."
        if ok
        else "py.typed marker is missing or not declared as package data.",
        evidence={"path": "src/crupier/py.typed", "exists": marker.exists(), "declared": declared},
        actions=["Add src/crupier/py.typed and include it in [tool.setuptools.package-data]."] if not ok else [],
    )


def _readme_check(root: Path, project: dict[str, Any]) -> ReleaseCheck:
    readme = str(project.get("readme") or "README.md")
    path = root / readme
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_markers = [
        "## Installation",
        "pip install",
        "crupier init",
        "## Python SDK Quickstart",
        "from crupier import Crupier",
        "Crupier.from_project",
        "dry_run=True",
        "crupier verify",
        "crupier release check",
        "crupier release check --strict-public",
        "crupier release check --check-pypi-name",
    ]
    missing_markers = [marker for marker in required_markers if marker not in text]
    ok = path.exists() and path.stat().st_size > 500 and not missing_markers
    return ReleaseCheck(
        id="readme",
        status="pass" if ok else "fail",
        severity="high",
        summary="README exists and covers install, init, verification, and release readiness."
        if ok
        else "README is missing required public onboarding content.",
        evidence={
            "path": readme,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "missing_markers": missing_markers,
        },
        actions=["Add install, configuration, verification, and release readiness docs to README.md."] if not ok else [],
    )


def _license_check(root: Path, project: dict[str, Any]) -> ReleaseCheck:
    license_files = [path.name for path in root.iterdir() if path.is_file() and path.name.upper().startswith("LICENSE")]
    declared = bool(project.get("license") or project.get("license-files"))
    ok = bool(license_files or declared)
    return ReleaseCheck(
        id="license",
        status="pass" if ok else "warn",
        severity="medium",
        summary="License metadata/file is present." if ok else "No license file or pyproject license metadata found.",
        evidence={"license_files": license_files, "declared": declared},
        actions=["Choose a license and add LICENSE plus pyproject license metadata before public PyPI release."]
        if not ok
        else [],
    )


def _contributing_check(root: Path) -> ReleaseCheck:
    path = root / "CONTRIBUTING.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_markers = [
        "python -m pytest",
        "crupier release check",
        "crupier release check --strict-public",
        "crupier release check --check-pypi-name",
        "crupier release check --verify-providers",
        "Never commit provider keys",
    ]
    missing_markers = [marker for marker in required_markers if marker not in text]
    ok = path.exists() and not missing_markers
    return ReleaseCheck(
        id="contributing",
        status="pass" if ok else "warn",
        severity="medium",
        summary="CONTRIBUTING.md covers development, tests, release gates, and secret handling."
        if ok
        else "CONTRIBUTING.md is missing or incomplete.",
        evidence={"path": "CONTRIBUTING.md", "exists": path.exists(), "missing_markers": missing_markers},
        actions=[
            "Add CONTRIBUTING.md with local test commands, strict release gates, provider-key handling, and public example expectations."
        ]
        if not ok
        else [],
    )


def _community_files_check(root: Path) -> ReleaseCheck:
    required_files = {
        "CODE_OF_CONDUCT.md": ["Expected Behavior", "Unacceptable Behavior", "Enforcement"],
        ".github/PULL_REQUEST_TEMPLATE.md": ["Validation", "Release/readiness", "No API keys"],
        ".github/ISSUE_TEMPLATE/bug_report.yml": ["Bug report", "Reproduction", "Environment"],
        ".github/ISSUE_TEMPLATE/feature_request.yml": ["Feature request", "Use case", "Constraints"],
        ".github/ISSUE_TEMPLATE/config.yml": ["blank_issues_enabled"],
    }
    missing_files: list[str] = []
    missing_markers: dict[str, list[str]] = {}
    for relative, markers in required_files.items():
        path = root / relative
        if not path.exists():
            missing_files.append(relative)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            missing_markers[relative] = missing
    ok = not missing_files and not missing_markers
    return ReleaseCheck(
        id="community_files",
        status="pass" if ok else "warn",
        severity="medium",
        summary="Public collaboration templates are present."
        if ok
        else "Public collaboration templates are missing or incomplete.",
        evidence={"missing_files": missing_files, "missing_markers": missing_markers},
        actions=[
            "Add CODE_OF_CONDUCT.md, GitHub issue templates, and a pull request template before opening the public repository."
        ]
        if not ok
        else [],
    )


def _security_policy_check(root: Path) -> ReleaseCheck:
    path = root / "SECURITY.md"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    required_markers = [
        "## Scope",
        "## Reporting A Vulnerability",
        "GitHub private vulnerability reporting",
        "## Supported Versions",
        "## Secret Handling Expectations",
        "## Disclosure And Fix Process",
        "Do not include:",
        "API keys",
        ".env",
        ".crupier/",
    ]
    missing_markers = [marker for marker in required_markers if marker not in text]
    ok = path.exists() and not missing_markers
    return ReleaseCheck(
        id="security_policy",
        status="pass" if ok else "warn",
        severity="medium",
        summary="SECURITY.md covers vulnerability reporting, supported versions, disclosure, and secret handling."
        if ok
        else "SECURITY.md is missing or incomplete.",
        evidence={"path": "SECURITY.md", "exists": path.exists(), "missing_markers": missing_markers},
        actions=["Add a vulnerability reporting policy with private reporting, supported versions, disclosure, and secret-handling guidance before public release."]
        if not ok
        else [],
    )


def _changelog_check(root: Path) -> ReleaseCheck:
    path = root / "CHANGELOG.md"
    return ReleaseCheck(
        id="changelog",
        status="pass" if path.exists() else "warn",
        severity="medium",
        summary="CHANGELOG.md is present." if path.exists() else "CHANGELOG.md is missing.",
        evidence={"path": "CHANGELOG.md", "exists": path.exists()},
        actions=["Maintain a changelog for package consumers."] if not path.exists() else [],
    )


def _ci_check(root: Path) -> ReleaseCheck:
    workflows = sorted((root / ".github" / "workflows").glob("*.yml")) + sorted(
        (root / ".github" / "workflows").glob("*.yaml")
    )
    ci_path = root / ".github" / "workflows" / "ci.yml"
    ci_text = ci_path.read_text(encoding="utf-8", errors="replace") if ci_path.exists() else ""
    required_markers = [
        "permissions:",
        "contents: read",
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "python -m pytest",
        "python -m ruff check src tests --select E9,F63,F7,F82",
        "python -m pip_audit --skip-editable --progress-spinner off",
        "crupier release check",
    ]
    missing_markers = [marker for marker in required_markers if marker not in ci_text]
    ok = bool(workflows) and not missing_markers
    return ReleaseCheck(
        id="ci",
        status="pass" if ok else "warn",
        severity="medium",
        summary="GitHub Actions CI runs tests and release readiness with minimal permissions."
        if ok
        else "GitHub Actions CI is missing or incomplete.",
        evidence={
            "workflows": [str(path.relative_to(root)) for path in workflows],
            "ci_path": ".github/workflows/ci.yml",
            "missing_markers": missing_markers,
        },
        actions=["Add CI that uses read-only contents permission, runs tests, and runs `crupier release check`."]
        if not ok
        else [],
    )


def _dependency_updates_check(root: Path) -> ReleaseCheck:
    path = root / ".github" / "dependabot.yml"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    required_markers = [
        "version: 2",
        "package-ecosystem: pip",
        "package-ecosystem: github-actions",
        "interval: weekly",
    ]
    missing_markers = [marker for marker in required_markers if marker not in text]
    ok = path.exists() and not missing_markers
    return ReleaseCheck(
        id="dependency_updates",
        status="pass" if ok else "warn",
        severity="medium",
        summary="Dependabot is configured for Python tooling and GitHub Actions."
        if ok
        else "Dependabot is missing or incomplete for public maintenance.",
        evidence={"path": ".github/dependabot.yml", "exists": path.exists(), "missing_markers": missing_markers},
        actions=["Add .github/dependabot.yml entries for pip and github-actions update checks."] if not ok else [],
    )


def _publish_workflow_check(root: Path) -> ReleaseCheck:
    path = root / ".github" / "workflows" / "publish.yml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_markers = [
        "id-token: write",
        "environment: pypi",
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "crupier release check --strict-public",
        "python -m ruff check src tests --select E9,F63,F7,F82",
        "python -m pip_audit --skip-editable --progress-spinner off",
        "python -m build --sdist --wheel --outdir dist",
        "actions/upload-artifact@v6",
        "if-no-files-found: error",
        "pypa/gh-action-pypi-publish",
    ]
    missing_markers = [marker for marker in required_markers if marker not in text]
    ok = path.exists() and not missing_markers
    return ReleaseCheck(
        id="publish_workflow",
        status="pass" if ok else "fail",
        severity="high",
        summary="PyPI publish workflow uses trusted publishing and strict public release checks."
        if ok
        else "PyPI publish workflow is missing trusted publishing or strict public release checks.",
        evidence={"path": ".github/workflows/publish.yml", "exists": path.exists(), "missing_markers": missing_markers},
        actions=[
            "Configure .github/workflows/publish.yml with current checkout/setup-python actions, OIDC trusted publishing, `crupier release check --strict-public`, artifact upload, and PyPI publishing."
        ]
        if not ok
        else [],
    )


def _script_check(project: dict[str, Any]) -> ReleaseCheck:
    scripts = project.get("scripts", {}) if isinstance(project.get("scripts"), dict) else {}
    ok = scripts.get("crupier") == "crupier.cli:main"
    return ReleaseCheck(
        id="console_script",
        status="pass" if ok else "fail",
        severity="high",
        summary="crupier console script is configured." if ok else "crupier console script is missing.",
        evidence={"scripts": scripts},
        actions=["Add [project.scripts] crupier = 'crupier.cli:main'."] if not ok else [],
    )


def _optional_dependencies_check(project: dict[str, Any]) -> ReleaseCheck:
    optional = project.get("optional-dependencies", {})
    expected = {"openai", "anthropic", "google", "ollama", "openrouter", "pdf", "all", "dev"}
    present = set(optional) if isinstance(optional, dict) else set()
    missing = sorted(expected - present)
    dev_dependencies = optional.get("dev", []) if isinstance(optional, dict) else []
    has_pip_audit = any(str(dependency).startswith("pip-audit") for dependency in dev_dependencies)
    has_ruff = any(str(dependency).startswith("ruff") for dependency in dev_dependencies)
    if not has_pip_audit:
        missing.append("dev:pip-audit")
    if not has_ruff:
        missing.append("dev:ruff")
    return ReleaseCheck(
        id="optional_dependencies",
        status="pass" if not missing else "warn",
        severity="medium",
        summary="Provider extras are declared." if not missing else "Some expected optional dependencies are missing.",
        evidence={
            "present": sorted(present),
            "missing": missing,
            "dev_has_pip_audit": has_pip_audit,
            "dev_has_ruff": has_ruff,
        },
        actions=[
            "Declare extras for every supported provider and dev workflow, including pip-audit for dependency vulnerability checks and Ruff for critical lint."
        ]
        if missing
        else [],
    )


def _default_config_check() -> ReleaseCheck:
    data = tomllib.loads(DEFAULT_TOML)
    config = CrupierConfig.from_dict(data)
    ollama = config.providers.get("ollama")
    failures: list[str] = []
    if not ollama:
        failures.append("providers.ollama missing from default config")
    else:
        if ollama.host != OLLAMA_CLOUD_HOST:
            failures.append("providers.ollama.host must default to Ollama Cloud")
        if ollama.env_key != "OLLAMA_API_KEY":
            failures.append("providers.ollama.env_key must be OLLAMA_API_KEY")
    if config.logging.persist_traces:
        failures.append("logging.persist_traces must default to false")
    if config.logging.store_prompts:
        failures.append("logging.store_prompts must default to false")
    if config.logging.store_responses:
        failures.append("logging.store_responses must default to false")
    if not config.logging.redact_secrets:
        failures.append("logging.redact_secrets must default to true")
    if config.routing.allow_latest_aliases:
        failures.append("routing.allow_latest_aliases must default to false")
    if config.routing.allow_preview_models:
        failures.append("routing.allow_preview_models must default to false")
    if f"OLLAMA_HOST={OLLAMA_CLOUD_HOST}" not in DEFAULT_ENV_EXAMPLE:
        failures.append(".env.example must advertise Ollama Cloud host")
    ok = not failures
    return ReleaseCheck(
        id="default_config",
        status="pass" if ok else "fail",
        severity="high",
        summary="Default project config is safe for public onboarding."
        if ok
        else "Default project config has unsafe or stale public onboarding defaults.",
        evidence={
            "failures": failures,
            "ollama_host": ollama.host if ollama else None,
            "persist_traces": config.logging.persist_traces,
            "store_prompts": config.logging.store_prompts,
            "store_responses": config.logging.store_responses,
            "redact_secrets": config.logging.redact_secrets,
            "allow_latest_aliases": config.routing.allow_latest_aliases,
            "allow_preview_models": config.routing.allow_preview_models,
        },
        actions=[
            "Keep `crupier init` defaults pointed at Ollama Cloud and keep prompt/response storage opt-in."
        ]
        if not ok
        else [],
    )


def _build_distribution_checks(root: Path) -> tuple[list[ReleaseCheck], dict[str, Any]]:
    if shutil.which(sys.executable) is None:
        return (
            [
                ReleaseCheck(
                    id="distribution_build",
                    status="fail",
                    severity="high",
                    summary="Python executable is unavailable for distribution build.",
                )
            ],
            {"skipped": False, "ok": False, "error": "python executable unavailable"},
        )
    with tempfile.TemporaryDirectory(prefix="crupier-build-") as tmp:
        tmp_path = Path(tmp)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        command = [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        artifacts = [*sdists, *wheels]
        ok = result.returncode == 0 and bool(wheels) and bool(sdists)
        build = {
            "skipped": False,
            "ok": ok,
            "command": command,
            "artifact_count": len(artifacts),
            "wheel_count": len(wheels),
            "sdist_count": len(sdists),
            "artifacts": [path.name for path in artifacts],
            "wheels": [path.name for path in wheels],
            "sdists": [path.name for path in sdists],
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-2000:],
        }
        checks = [
            ReleaseCheck(
                id="distribution_build",
                status="pass" if ok else "fail",
                severity="high",
                summary="Wheel and sdist build successfully." if ok else "Wheel/sdist build failed.",
                evidence={"artifacts": build["artifacts"], "returncode": result.returncode},
                actions=[
                    "Install the dev extra (`pip install -e .[dev]`) and fix package build errors before release."
                ]
                if not ok
                else [],
            )
        ]
        content_check, content = _artifact_content_check(artifacts)
        checks.append(content_check)
        build["artifact_content"] = content
        twine_check, twine = _twine_check(artifacts)
        checks.append(twine_check)
        build["twine_check"] = twine
        wheel_smoke_check, wheel_smoke = _wheel_install_smoke(wheels[0] if wheels else None, tmp_path)
        checks.append(wheel_smoke_check)
        build["install_smoke"] = wheel_smoke
        build["wheel_install_smoke"] = wheel_smoke
        sdist_smoke_check, sdist_smoke = _sdist_install_smoke(sdists[0] if sdists else None, tmp_path)
        checks.append(sdist_smoke_check)
        build["sdist_install_smoke"] = sdist_smoke
        build["ok"] = bool(ok and twine.get("ok") and wheel_smoke.get("ok") and sdist_smoke.get("ok"))
        return checks, build


def _artifact_content_check(artifacts: list[Path]) -> tuple[ReleaseCheck, dict[str, Any]]:
    inspected: list[dict[str, Any]] = []
    forbidden: list[str] = []
    typed_marker_present = False
    env_example_present = False
    contributing_present = False
    code_of_conduct_present = False
    example_script_present = False
    for artifact in artifacts:
        names: list[str] = []
        try:
            if artifact.suffix == ".whl":
                with zipfile.ZipFile(artifact) as wheel:
                    names = wheel.namelist()
            elif artifact.name.endswith(".tar.gz"):
                with tarfile.open(artifact, "r:gz") as sdist:
                    names = sdist.getnames()
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            forbidden.append(f"{artifact.name}: unreadable artifact: {exc}")
        inspected.append({"artifact": artifact.name, "file_count": len(names)})
        for name in names:
            if _artifact_entry_is_forbidden(name):
                forbidden.append(f"{artifact.name}:{name}")
            if name.endswith("crupier/py.typed"):
                typed_marker_present = True
            if name == ".env.example" or name.endswith("/.env.example"):
                env_example_present = True
            if name == "CONTRIBUTING.md" or name.endswith("/CONTRIBUTING.md"):
                contributing_present = True
            if name == "CODE_OF_CONDUCT.md" or name.endswith("/CODE_OF_CONDUCT.md"):
                code_of_conduct_present = True
            if name == "examples/sdk_dry_run.py" or name.endswith("/examples/sdk_dry_run.py"):
                example_script_present = True
    ok = (
        bool(artifacts)
        and not forbidden
        and typed_marker_present
        and env_example_present
        and contributing_present
        and code_of_conduct_present
        and example_script_present
    )
    payload = {
        "ok": ok,
        "artifacts": [artifact.name for artifact in artifacts],
        "inspected": inspected,
        "forbidden": forbidden[:50],
        "forbidden_count": len(forbidden),
        "typed_marker_present": typed_marker_present,
        "env_example_present": env_example_present,
        "contributing_present": contributing_present,
        "code_of_conduct_present": code_of_conduct_present,
        "example_script_present": example_script_present,
    }
    return (
        ReleaseCheck(
            id="artifact_content",
            status="pass" if ok else "fail",
            severity="high",
            summary="Built distributions contain expected package files and no local secret/cache artifacts."
            if ok
            else "Built distributions contain unexpected content or are missing py.typed/.env.example/CONTRIBUTING/CODE_OF_CONDUCT/examples.",
            evidence=payload,
            actions=["Inspect built distributions, include .env.example, and exclude local secret/cache artifacts before publishing."]
            if not ok
            else [],
        ),
        payload,
    )


def _artifact_entry_is_forbidden(name: str) -> bool:
    parts = PurePosixPath(name).parts
    for part in parts:
        if part == ".env":
            return True
        if part.startswith(".env.") and part != ".env.example":
            return True
        if part in {".crupier", "__pycache__", ".pytest_cache"}:
            return True
    return False


def _twine_check(artifacts: list[Path]) -> tuple[ReleaseCheck, dict[str, Any]]:
    if not artifacts:
        return (
            ReleaseCheck(
                id="twine_check",
                status="fail",
                severity="high",
                summary="Twine metadata check could not run because no distributions were built.",
                actions=["Fix distribution build errors before publishing."],
            ),
            {"skipped": True, "ok": False, "reason": "missing_artifacts"},
        )
    command = [sys.executable, "-m", "twine", "check", *[str(path) for path in artifacts]]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    ok = result.returncode == 0
    payload = {
        "skipped": False,
        "ok": ok,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "artifacts": [path.name for path in artifacts],
    }
    return (
        ReleaseCheck(
            id="twine_check",
            status="pass" if ok else "fail",
            severity="high",
            summary="Twine metadata/render check passed for built distributions."
            if ok
            else "Twine metadata/render check failed for built distributions.",
            evidence={"artifacts": payload["artifacts"], "returncode": result.returncode},
            actions=[
                "Install the dev extra (`pip install -e .[dev]`) and fix package metadata/README rendering before upload."
            ]
            if not ok
            else [],
        ),
        payload,
    )


def _wheel_install_smoke(wheel: Path | None, tmp_path: Path) -> tuple[ReleaseCheck, dict[str, Any]]:
    if wheel is None:
        return (
            ReleaseCheck(
                id="wheel_install_smoke",
                status="fail",
                severity="high",
                summary="Wheel install smoke could not run because no wheel was built.",
                actions=["Fix package build errors before release."],
            ),
            {"skipped": True, "ok": False, "reason": "missing_wheel"},
        )
    venv_dir = tmp_path / "install-smoke-venv"
    create = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = venv_dir / ("Scripts/crupier.exe" if sys.platform == "win32" else "bin/crupier")
    steps: list[dict[str, Any]] = [
        {"name": "create_venv", "returncode": create.returncode, "stderr_tail": create.stderr[-1000:]}
    ]
    ok = create.returncode == 0 and python.exists()
    if ok:
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            text=True,
            capture_output=True,
            check=False,
        )
        steps.append({"name": "install_wheel", "returncode": install.returncode, "stderr_tail": install.stderr[-1000:]})
        ok = install.returncode == 0
    else:
        install = None
    if ok:
        imported = subprocess.run(
            [str(python), "-c", "import crupier; print(crupier.__version__)"],
            text=True,
            capture_output=True,
            check=False,
        )
        steps.append({"name": "import_crupier", "returncode": imported.returncode, "stdout": imported.stdout.strip()})
        ok = imported.returncode == 0
    else:
        imported = None
    if ok:
        cli = subprocess.run([str(script), "--help"], text=True, capture_output=True, check=False)
        steps.append({"name": "cli_help", "returncode": cli.returncode, "stdout_head": cli.stdout[:200]})
        ok = cli.returncode == 0
    else:
        cli = None
    if ok:
        init_ok, init_step = _installed_init_smoke(
            script,
            python,
            tmp_path / "wheel-init-smoke-project",
        )
        steps.append(init_step)
        ok = init_ok
    smoke = {
        "skipped": False,
        "ok": ok,
        "wheel": wheel.name,
        "steps": steps,
        "import_version": imported.stdout.strip() if imported and imported.returncode == 0 else None,
    }
    return (
        ReleaseCheck(
            id="wheel_install_smoke",
            status="pass" if ok else "fail",
            severity="high",
            summary=(
                "Built wheel installs, imports, exposes the crupier CLI, runs crupier init, "
                "routes dry-run, and executes the Python SDK quickstart."
            )
            if ok
            else "Built wheel failed install/import/CLI/init smoke.",
            evidence={
                "wheel": wheel.name,
                "steps": [{k: v for k, v in step.items() if k in {"name", "returncode"}} for step in steps],
            },
            actions=["Fix wheel installation, import, console-script packaging, or init defaults before release."]
            if not ok
            else [],
        ),
        smoke,
    )


def _sdist_install_smoke(sdist: Path | None, tmp_path: Path) -> tuple[ReleaseCheck, dict[str, Any]]:
    if sdist is None:
        return (
            ReleaseCheck(
                id="sdist_install_smoke",
                status="fail",
                severity="high",
                summary="Sdist install smoke could not run because no sdist was built.",
                actions=["Fix package build errors before release."],
            ),
            {"skipped": True, "ok": False, "reason": "missing_sdist"},
        )
    venv_dir = tmp_path / "sdist-install-smoke-venv"
    create = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = venv_dir / ("Scripts/crupier.exe" if sys.platform == "win32" else "bin/crupier")
    steps: list[dict[str, Any]] = [
        {"name": "create_venv", "returncode": create.returncode, "stderr_tail": create.stderr[-1000:]}
    ]
    ok = create.returncode == 0 and python.exists()
    if ok:
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(sdist)],
            text=True,
            capture_output=True,
            check=False,
        )
        steps.append({"name": "install_sdist", "returncode": install.returncode, "stderr_tail": install.stderr[-1000:]})
        ok = install.returncode == 0
    else:
        install = None
    if ok:
        imported = subprocess.run(
            [str(python), "-c", "import crupier; print(crupier.__version__)"],
            text=True,
            capture_output=True,
            check=False,
        )
        steps.append({"name": "import_crupier", "returncode": imported.returncode, "stdout": imported.stdout.strip()})
        ok = imported.returncode == 0
    else:
        imported = None
    if ok:
        cli = subprocess.run([str(script), "--help"], text=True, capture_output=True, check=False)
        steps.append({"name": "cli_help", "returncode": cli.returncode, "stdout_head": cli.stdout[:200]})
        ok = cli.returncode == 0
    else:
        cli = None
    if ok:
        init_ok, init_step = _installed_init_smoke(
            script,
            python,
            tmp_path / "sdist-init-smoke-project",
        )
        steps.append(init_step)
        ok = init_ok
    smoke = {
        "skipped": False,
        "ok": ok,
        "sdist": sdist.name,
        "steps": steps,
        "import_version": imported.stdout.strip() if imported and imported.returncode == 0 else None,
    }
    return (
        ReleaseCheck(
            id="sdist_install_smoke",
            status="pass" if ok else "fail",
            severity="high",
            summary=(
                "Built sdist installs, imports, exposes the crupier CLI, runs crupier init, "
                "routes dry-run, and executes the Python SDK quickstart."
            )
            if ok
            else "Built sdist failed install/import/CLI/init smoke.",
            evidence={
                "sdist": sdist.name,
                "steps": [{k: v for k, v in step.items() if k in {"name", "returncode"}} for step in steps],
            },
            actions=["Fix sdist installation, import, console-script packaging, or init defaults before release."]
            if not ok
            else [],
        ),
        smoke,
    )


def _installed_init_smoke(script: Path, python: Path, project_dir: Path) -> tuple[bool, dict[str, Any]]:
    init = subprocess.run(
        [str(script), "--project", str(project_dir), "init"],
        text=True,
        capture_output=True,
        check=False,
    )
    toml_path = project_dir / "crupier.toml"
    env_example_path = project_dir / ".env.example"
    gitignore_path = project_dir / ".gitignore"
    toml_text = toml_path.read_text(encoding="utf-8") if toml_path.exists() else ""
    env_text = env_example_path.read_text(encoding="utf-8") if env_example_path.exists() else ""
    gitignore_text = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    init_ok = (
        init.returncode == 0
        and toml_path.exists()
        and env_example_path.exists()
        and gitignore_path.exists()
        and 'host = "https://ollama.com/api"' in toml_text
        and "OLLAMA_HOST=https://ollama.com/api" in env_text
        and ".env" in gitignore_text
        and "!.env.example" in gitignore_text
    )
    route = subprocess.run(
        [str(script), "--project", str(project_dir), "route", "Say hello in one short sentence", "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    route_payload: dict[str, Any] = {}
    if route.stdout.strip():
        try:
            parsed = json.loads(route.stdout)
            route_payload = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            route_payload = {}
    route_ok = route.returncode == 0 and bool(route_payload.get("strategy")) and bool(route_payload.get("steps"))
    sdk = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from crupier import Crupier\n"
                f"client = Crupier.from_project({str(project_dir)!r})\n"
                "result = client.deal('Plan a short support reply', input={'priority': 'normal'}, dry_run=True)\n"
                "print(result.route.strategy)\n"
                "print(result.route.model_summary)\n"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    sdk_lines = [line.strip() for line in sdk.stdout.splitlines() if line.strip()]
    sdk_ok = sdk.returncode == 0 and len(sdk_lines) >= 2 and bool(sdk_lines[0]) and bool(sdk_lines[1])
    generated = init_ok and route_ok and sdk_ok
    return generated, {
        "name": "init_project",
        "returncode": init.returncode,
        "stdout_head": init.stdout[:200],
        "stderr_tail": init.stderr[-1000:],
        "init_ok": init_ok,
        "route_returncode": route.returncode,
        "route_stdout_head": route.stdout[:200],
        "route_stderr_tail": route.stderr[-1000:],
        "route_ok": route_ok,
        "sdk_returncode": sdk.returncode,
        "sdk_stdout_head": sdk.stdout[:200],
        "sdk_stderr_tail": sdk.stderr[-1000:],
        "sdk_ok": sdk_ok,
        "generated": generated,
        "project": str(project_dir),
    }


def format_release_check_report(report: ReleaseCheckReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)

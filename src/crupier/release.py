"""Release readiness checks for the local package."""

from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import asdict, dataclass, field
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import (
    DEFAULT_ENV_EXAMPLE,
    DEFAULT_GITIGNORE_ENTRIES,
    DEFAULT_TOML,
    OLLAMA_CLOUD_HOST,
    OPENROUTER_DEFAULT_HOST,
    CrupierConfig,
)
from .default_cards import BUILTIN_CAPABILITY_CARDS

_FINAL_PUBLIC_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_NON_FINAL_RELEASE_PATTERNS = [
    re.compile(r"\b\d+\.\d+\.\d+(?:a|b|alpha|beta|rc)\d*\b", re.IGNORECASE),
    re.compile(r"\b(?:alpha|beta|pre-release|prerelease|release candidate)\b", re.IGNORECASE),
    re.compile(r"Development Status :: [34]\b"),
]
_PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/{project}/json"
_EXPECTED_EXAMPLE_FILES = {
    "examples/_example_support.py",
    "examples/agentic_pr_review.py",
    "examples/customer_support_triage.py",
    "examples/drop_in_agent_boundary.py",
    "examples/model-compare-eval.json",
    "examples/multimodal_claim_review.py",
    "examples/routing-eval.json",
    "examples/sdk_dry_run.py",
    "examples/workflow_operations_hub.py",
}
_RELEASE_SOURCE_IGNORED_NAMES = {
    ".crupier",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "env",
    "venv",
    "__pycache__",
}
_RELEASE_SOURCE_IGNORED_PATTERNS = {
    "*.egg-info",
    "*.py[cod]",
    ".DS_Store",
}
_SECRET_SCAN_PATTERNS = (
    ("anthropic_api_key", re.compile(r"\bsk-ant-api\d{2}-[A-Za-z0-9_\-]{20,}\b")),
    ("openrouter_api_key", re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{20,}\b")),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-[A-Za-z0-9_\-]{20,}|[A-Za-z0-9_\-]{32,})\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("ollama_cloud_key", re.compile(r"\b[a-f0-9]{40}\.[A-Za-z0-9_\-]{20,}\b", re.IGNORECASE)),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{32,}\b", re.IGNORECASE)),
)
_SECRET_SCAN_SKIP_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "build",
    "dist",
    "env",
    "venv",
    "__pycache__",
}
_SECRET_SCAN_SKIP_SUFFIXES = {
    ".7z",
    ".bin",
    ".bmp",
    ".bz2",
    ".class",
    ".db",
    ".dll",
    ".dylib",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".tar",
    ".tgz",
    ".webp",
    ".whl",
    ".zip",
}
_SECRET_SCAN_MAX_BYTES = 2_000_000


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
        _public_model_examples_check(root_path),
        _public_repository_surface_check(root_path),
        _public_secret_scan_check(root_path),
        _repository_gitignore_check(root_path),
        _typed_marker_check(root_path, data),
        _readme_check(root_path, project),
        _readme_pypi_links_check(root_path, project),
        _public_markdown_check(root_path),
        _public_yaml_check(root_path),
        _license_check(root_path, project),
        _contributing_check(root_path),
        _community_files_check(root_path),
        _security_policy_check(root_path),
        _changelog_check(root_path, version),
        _ci_check(root_path),
        _dependency_updates_check(root_path),
        _publish_workflow_check(root_path),
        _script_check(project),
        _optional_dependencies_check(project),
        _default_config_check(),
        _runtime_safety_defaults_check(),
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


def check_project_urls_reachable(root: str | Path, *, timeout_seconds: float = 10.0) -> ReleaseCheck:
    """Verify that public project URLs in pyproject.toml resolve over HTTP."""

    root_path = Path(root).resolve()
    data = _read_pyproject(root_path / "pyproject.toml")
    project = data.get("project", {}) if isinstance(data, dict) else {}
    syntax_check = _project_urls_check(project if isinstance(project, dict) else {})
    urls = syntax_check.evidence.get("valid", {}) if isinstance(syntax_check.evidence, dict) else {}
    if syntax_check.status != "pass" or not isinstance(urls, dict) or not urls:
        return ReleaseCheck(
            id="project_urls_reachable",
            status="fail",
            severity="high",
            summary="Project URLs could not be verified because public metadata URLs are missing or incomplete.",
            evidence={"project_urls_check": syntax_check.to_dict()},
            actions=["Add real reachable [project.urls] entries before publishing publicly."],
        )

    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label, url in sorted(urls.items()):
        request = Request(str(url), headers={"User-Agent": "crupier-release-check/project-urls"})
        item: dict[str, Any] = {"label": str(label), "url": str(url)}
        try:
            response = urlopen(request, timeout=timeout_seconds)
        except HTTPError as exc:
            item.update({"ok": False, "http_status": exc.code, "error_type": exc.__class__.__name__})
            failures.append(item)
            checked.append(item)
            continue
        except (OSError, TimeoutError, URLError) as exc:
            item.update({"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)})
            failures.append(item)
            checked.append(item)
            continue

        try:
            status = int(getattr(response, "status", getattr(response, "code", 200)))
            final_url = getattr(response, "url", str(url))
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        item.update({"ok": 200 <= status < 400, "http_status": status, "final_url": final_url})
        if not item["ok"]:
            failures.append(item)
        checked.append(item)

    ok = not failures
    return ReleaseCheck(
        id="project_urls_reachable",
        status="pass" if ok else "fail",
        severity="high",
        summary="Project URLs are reachable for public package consumers."
        if ok
        else "One or more public project URLs could not be reached.",
        evidence={"checked": checked, "failures": failures},
        actions=[
            "Make the GitHub repository public, verify public access to project metadata links, or replace [project.urls] with reachable public URLs before publishing."
        ]
        if not ok
        else [],
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    }
    missing_classifiers = sorted(required_classifiers - classifier_set)
    missing.extend(f"classifier:{classifier}" for classifier in missing_classifiers)
    trove_validation_available = False
    unknown_classifiers: list[str] = []
    try:
        import trove_classifiers  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        pass
    else:
        trove_validation_available = True
        known_classifiers = set(trove_classifiers.classifiers)
        unknown_classifiers = sorted(classifier for classifier in classifier_set if classifier not in known_classifiers)
        missing.extend(f"unknown-classifier:{classifier}" for classifier in unknown_classifiers)
    return ReleaseCheck(
        id="pyproject_metadata",
        status="pass" if path.exists() and not missing else "fail",
        severity="high",
        summary="pyproject.toml has required package metadata."
        if path.exists() and not missing
        else "pyproject.toml is missing required package metadata.",
        evidence={
            "path": str(path),
            "missing": missing,
            "missing_classifiers": missing_classifiers,
            "trove_validation_available": trove_validation_available,
            "unknown_classifiers": unknown_classifiers,
        },
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
    has_project_home = any(label.lower() in {"homepage", "repository"} for label in valid)
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
            "Add real [project.urls] entries such as Repository, Changelog, and Issues once the public repository exists."
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
        summary="Public package metadata and onboarding files describe a final 0.1.0 release."
        if ok
        else "Public package metadata or onboarding files still describe a non-final release.",
        evidence={"matches": matches[:50], "match_count": len(matches)},
        actions=["Remove alpha, beta, pre-release, release-candidate, or non-final version language before publishing."]
        if not ok
        else [],
    )


def _public_model_examples_check(root: Path) -> ReleaseCheck:
    stale_examples = ["gpt-4o-mini", "gpt-4.1-mini"]
    public_paths = [
        root / "README.md",
        root / "src" / "crupier" / "cli.py",
    ]
    matches: list[dict[str, Any]] = []
    for path in public_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for model in stale_examples:
            start = 0
            while True:
                index = text.find(model, start)
                if index == -1:
                    break
                matches.append(
                    {
                        "path": str(path.relative_to(root)),
                        "line": text.count("\n", 0, index) + 1,
                        "model": model,
                    }
                )
                start = index + len(model)
    ok = not matches
    return ReleaseCheck(
        id="public_model_examples",
        status="pass" if ok else "fail",
        severity="medium",
        summary="Public README and CLI examples use current default model references."
        if ok
        else "Public README or CLI examples still show stale model references.",
        evidence={"matches": matches[:50], "match_count": len(matches), "stale_examples": stale_examples},
        actions=["Update public README/CLI examples to current default allowlist models before publishing."]
        if not ok
        else [],
    )


def _public_repository_surface_check(root: Path) -> ReleaseCheck:
    forbidden_paths = [
        "CODE_OF_CONDUCT.md",
        "docs",
    ]
    present = [relative for relative in forbidden_paths if (root / relative).exists()]
    ok = not present
    return ReleaseCheck(
        id="public_repository_surface",
        status="pass" if ok else "fail",
        severity="high",
        summary="Public repository surface excludes internal docs and extra community policy files."
        if ok
        else "Public repository surface includes files that should stay out of the public repo.",
        evidence={"forbidden_present": present},
        actions=["Remove internal docs and extra community policy files before opening the repository publicly."]
        if not ok
        else [],
    )


def _public_secret_scan_check(root: Path) -> ReleaseCheck:
    tracked = _git_tracked_files(root)
    include_env_files = bool(tracked)
    candidates = tracked if tracked else _public_secret_scan_fallback_files(root)
    checked: list[str] = []
    skipped_large: list[str] = []
    findings: list[dict[str, Any]] = []

    for relative in sorted(set(candidates), key=lambda path: path.as_posix()):
        if not _should_secret_scan_path(relative, include_env_files=include_env_files):
            continue
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _SECRET_SCAN_MAX_BYTES:
            skipped_large.append(relative.as_posix())
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        checked.append(relative.as_posix())
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern_name, pattern in _SECRET_SCAN_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "pattern": pattern_name,
                        }
                    )
                    break
            if len(findings) >= 50:
                break
        if len(findings) >= 50:
            break

    ok = not findings
    return ReleaseCheck(
        id="public_secret_scan",
        status="pass" if ok else "fail",
        severity="high",
        summary="Public tracked files do not contain provider-key shaped secrets."
        if ok
        else "Public tracked files contain provider-key shaped secrets.",
        evidence={
            "checked_count": len(checked),
            "checked_sample": checked[:25],
            "skipped_large": skipped_large,
            "finding_count": len(findings),
            "findings": findings,
            "scanned_git_tracked_files": bool(tracked),
        },
        actions=[
            "Remove committed provider keys/tokens, rotate any exposed credentials, and rerun the release check before opening the repository publicly."
        ]
        if not ok
        else [],
    )


def _public_secret_scan_fallback_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        files.append(relative)
    return files


def _should_secret_scan_path(relative: Path, *, include_env_files: bool) -> bool:
    parts = relative.parts
    if any(part in _SECRET_SCAN_SKIP_NAMES for part in parts):
        return False
    name = relative.name
    if not include_env_files and (name == ".env" or (name.startswith(".env.") and name != ".env.example")):
        return False
    if name.endswith((".pyc", ".pyo")):
        return False
    suffixes = {suffix.lower() for suffix in relative.suffixes}
    return not suffixes.intersection(_SECRET_SCAN_SKIP_SUFFIXES)


def _repository_gitignore_check(root: Path) -> ReleaseCheck:
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    normalized = {line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")}
    required_entries = [
        *DEFAULT_GITIGNORE_ENTRIES,
        ".venv/",
        "venv/",
        "env/",
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        "*.egg-info/",
        "dist/",
        "build/",
    ]
    missing = [entry for entry in required_entries if entry not in normalized]
    ok = path.exists() and not missing
    return ReleaseCheck(
        id="repository_gitignore",
        status="pass" if ok else "warn",
        severity="medium",
        summary="Repository .gitignore protects local keys, caches, builds, and generated Crupier artifacts."
        if ok
        else "Repository .gitignore is missing entries for local keys, caches, builds, or generated Crupier artifacts.",
        evidence={
            "path": ".gitignore",
            "exists": path.exists(),
            "missing_entries": missing,
        },
        actions=[
            "Add .gitignore entries for local .env files, virtualenvs, caches, dist/build outputs, and generated .crupier artifacts before public release."
        ]
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
        "crupier release check --strict-public --verify-project-urls --check-pypi-name",
        "crupier capabilities probe --provider google --apply",
        "crupier release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama",
        "OpenAI, Anthropic Claude, Google Gemini, and Ollama adapters",
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
        actions=["Add install, configuration, verification, and release readiness content to README.md."]
        if not ok
        else [],
    )


def _readme_pypi_links_check(root: Path, project: dict[str, Any]) -> ReleaseCheck:
    readme = str(project.get("readme") or "README.md")
    path = root / readme
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    relative_links = _relative_markdown_links(text)
    ok = path.exists() and not relative_links
    return ReleaseCheck(
        id="readme_pypi_links",
        status="pass" if ok else "fail",
        severity="medium",
        summary="README links are absolute for PyPI rendering."
        if ok
        else "README contains relative links that may not render correctly on PyPI.",
        evidence={"path": readme, "exists": path.exists(), "relative_links": relative_links},
        actions=["Use absolute HTTPS links for README links that will render on PyPI."] if not ok else [],
    )


def _public_markdown_check(root: Path) -> ReleaseCheck:
    public_docs = [
        Path("README.md"),
        Path("CONTRIBUTING.md"),
        Path("SECURITY.md"),
        Path("CHANGELOG.md"),
        Path(".github/PULL_REQUEST_TEMPLATE.md"),
    ]
    checked: list[str] = []
    unbalanced_fences: dict[str, int] = {}
    broken_links: list[dict[str, str]] = []
    root = root.resolve()

    for relative in public_docs:
        path = root / relative
        if not path.exists():
            continue
        checked.append(relative.as_posix())
        text = path.read_text(encoding="utf-8", errors="replace")
        fence_count = sum(1 for line in text.splitlines() if line.strip().startswith("```"))
        if fence_count % 2:
            unbalanced_fences[relative.as_posix()] = fence_count
        broken_links.extend(_broken_relative_markdown_links(root, relative, text))

    ok = not unbalanced_fences and not broken_links
    return ReleaseCheck(
        id="public_markdown",
        status="pass" if ok else "fail",
        severity="high",
        summary="Public Markdown files have balanced code fences and valid relative links."
        if ok
        else "Public Markdown files contain broken fences or relative links.",
        evidence={
            "checked": checked,
            "unbalanced_fences": unbalanced_fences,
            "broken_links": broken_links,
        },
        actions=["Fix broken Markdown fences and relative links before publishing public docs."] if not ok else [],
    )


def _public_yaml_check(root: Path) -> ReleaseCheck:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return ReleaseCheck(
            id="public_yaml",
            status="warn",
            severity="medium",
            summary="PyYAML is not installed, so public GitHub YAML syntax was not checked.",
            evidence={"parser_available": False, "checked": [], "failures": []},
            actions=["Install the development extra with `pip install -e '.[dev]'` before strict public release checks."],
        )

    github_root = root / ".github"
    yaml_paths: list[Path] = []
    if github_root.exists():
        yaml_paths = sorted({*github_root.rglob("*.yml"), *github_root.rglob("*.yaml")})

    checked: list[str] = []
    failures: list[dict[str, str]] = []
    for path in yaml_paths:
        relative = path.relative_to(root).as_posix()
        checked.append(relative)
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # pragma: no cover - exact parser exception classes vary by PyYAML version
            failures.append(
                {
                    "path": relative,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
            continue
        if parsed is None:
            failures.append(
                {
                    "path": relative,
                    "error_type": "EmptyDocument",
                    "error": "YAML document is empty.",
                }
            )
            continue
        if not isinstance(parsed, dict):
            failures.append(
                {
                    "path": relative,
                    "error_type": "InvalidTopLevel",
                    "error": "YAML top-level document must be a mapping.",
                }
            )

    ok = not failures
    return ReleaseCheck(
        id="public_yaml",
        status="pass" if ok else "fail",
        severity="high",
        summary="Public GitHub YAML files parse successfully."
        if ok
        else "Public GitHub YAML files contain syntax or shape errors.",
        evidence={"parser_available": True, "checked": checked, "failures": failures},
        actions=["Fix invalid public GitHub workflow, Dependabot, or issue-template YAML before publishing."]
        if not ok
        else [],
    )


def _broken_relative_markdown_links(root: Path, source_relative: Path, text: str) -> list[dict[str, str]]:
    broken: list[dict[str, str]] = []
    source_parent = (root / source_relative).parent
    for match in re.finditer(r"!?\[[^\]]+\]\(([^)]+)\)", text):
        raw_target = match.group(1).strip().strip("<>")
        target = raw_target.split("#", 1)[0].split("?", 1)[0]
        if not target or target.startswith("#") or target.startswith("//"):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
            continue
        resolved = (source_parent / target).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            broken.append(
                {
                    "source": source_relative.as_posix(),
                    "target": raw_target,
                    "reason": "points outside repository",
                }
            )
            continue
        if not resolved.exists():
            broken.append(
                {
                    "source": source_relative.as_posix(),
                    "target": raw_target,
                    "reason": "missing target",
                }
            )
    return broken


def _relative_markdown_links(text: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r"!?\[[^\]]+\]\(([^)]+)\)", text):
        raw_target = match.group(1).strip().strip("<>")
        target = raw_target.split("#", 1)[0].split("?", 1)[0]
        if not target or target.startswith("#") or target.startswith("//"):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
            continue
        links.append(raw_target)
    return links


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
        "crupier release check --strict-public --verify-project-urls --check-pypi-name",
        "GEMINI_API_KEY",
        "crupier --env-file .env release check --verify-providers --provider openai --provider anthropic --provider google --provider ollama",
        "Never commit provider keys",
        "Dependabot security updates",
        "Protect `main`",
        "disallow force pushes",
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
            "Add CONTRIBUTING.md with local test commands, strict release gates, provider-key handling, public example expectations, and public branch-protection expectations."
        ]
        if not ok
        else [],
    )


def _community_files_check(root: Path) -> ReleaseCheck:
    required_files = {
        ".github/PULL_REQUEST_TEMPLATE.md": [
            "Validation",
            "Release/readiness",
            "No API keys",
            ".crupier/",
            "crupier release check --strict-public --verify-project-urls --check-pypi-name",
        ],
        ".github/ISSUE_TEMPLATE/bug_report.yml": [
            "Bug report",
            "Reproduction",
            "Environment",
            "Safety confirmation",
            "I have removed API keys",
            "provider responses",
            ".env",
            ".crupier/",
            "required: true",
        ],
        ".github/ISSUE_TEMPLATE/feature_request.yml": [
            "Feature request",
            "Use case",
            "Constraints",
            "Safety confirmation",
            "I have removed API keys",
            "provider responses",
            ".env",
            ".crupier/",
            "required: true",
        ],
        ".github/ISSUE_TEMPLATE/config.yml": ["blank_issues_enabled: false"],
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
        actions=["Add GitHub issue templates and a pull request template before opening the public repository."]
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


def _changelog_check(root: Path, version: str | None) -> ReleaseCheck:
    path = root / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    version_marker = f"## {version}" if version else ""
    missing_markers = [version_marker] if version_marker and version_marker not in text else []
    ok = path.exists() and not missing_markers
    return ReleaseCheck(
        id="changelog",
        status="pass" if ok else "warn",
        severity="medium",
        summary="CHANGELOG.md includes the packaged version."
        if ok
        else "CHANGELOG.md is missing or does not mention the packaged version.",
        evidence={
            "path": "CHANGELOG.md",
            "exists": path.exists(),
            "version": version,
            "missing_markers": missing_markers,
        },
        actions=["Maintain a changelog entry for the exact version being packaged."] if not ok else [],
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
        "actions/checkout@v7",
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
        "concurrency:",
        "pypi-publish-${{ github.ref }}",
        "cancel-in-progress: false",
        "environment:",
        "name: pypi",
        "url: https://pypi.org/p/crupier",
        "    permissions:",
        "      contents: read",
        "      id-token: write",
        "workflow_dispatch:",
        "confirm_publish",
        "actions/checkout@v7",
        "fetch-depth: 0",
        "actions/setup-python@v6",
        "Verify publish event matches package version",
        "GITHUB_EVENT_NAME",
        "GITHUB_REF_NAME",
        "REQUESTED_VERSION",
        "CONFIRM_PUBLISH",
        "RELEASE_IS_DRAFT",
        "RELEASE_IS_PRERELEASE",
        "RELEASE_TARGET_COMMITISH",
        "git\", \"fetch\", \"--quiet\", \"origin\", \"main:refs/remotes/origin/main\", \"--tags",
        "git\", \"rev-parse\", \"origin/main",
        "Publish commit does not match origin/main.",
        "Publishing from draft GitHub Releases is not allowed.",
        "Publishing from prerelease GitHub Releases is not allowed.",
        "is not the main branch",
        "is not main",
        "FIRST_PUBLIC_RELEASE_VERSION",
        "crupier release check --strict-public --verify-project-urls --check-pypi-name",
        "--allow-existing-pypi-project",
        "python -m ruff check src tests --select E9,F63,F7,F82",
        "python -m pip_audit --skip-editable --progress-spinner off",
        "python -m build --sdist --wheel --outdir dist",
        "actions/upload-artifact@v7",
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
            "Configure .github/workflows/publish.yml with current checkout/setup-python actions, release-tag/manual-version matching, draft/prerelease blocking, main-branch-only publishing, serialized publish concurrency, job-scoped OIDC trusted publishing permissions, strict release checks, first-upload PyPI name blocking, maintenance-release existing-project allowance, artifact upload, and PyPI publishing."
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
    expected_dependency_prefixes = {
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "google": ["google-genai"],
        "openrouter": ["openai"],
        "pdf": ["pypdf"],
        "all": ["openai", "anthropic", "google-genai", "pypdf"],
    }
    present = set(optional) if isinstance(optional, dict) else set()
    missing = sorted(expected - present)
    dev_dependencies = optional.get("dev", []) if isinstance(optional, dict) else []
    has_pip_audit = any(str(dependency).startswith("pip-audit") for dependency in dev_dependencies)
    has_ruff = any(str(dependency).startswith("ruff") for dependency in dev_dependencies)
    has_trove_classifiers = any(str(dependency).startswith("trove-classifiers") for dependency in dev_dependencies)
    has_pyyaml = any(str(dependency).lower().startswith("pyyaml") for dependency in dev_dependencies)
    missing_expected_dependencies: dict[str, list[str]] = {}
    unneeded_ollama_sdk_dependencies: dict[str, list[str]] = {}
    if isinstance(optional, dict):
        for extra, dependency_prefixes in expected_dependency_prefixes.items():
            dependencies = optional.get(extra, [])
            if not isinstance(dependencies, list):
                dependencies = []
            normalized = [str(dependency).lower().strip() for dependency in dependencies]
            for prefix in dependency_prefixes:
                if not any(dependency.startswith(prefix) for dependency in normalized):
                    missing_expected_dependencies.setdefault(extra, []).append(prefix)
        for extra, dependencies in optional.items():
            if not isinstance(dependencies, list):
                continue
            matches = [
                str(dependency)
                for dependency in dependencies
                if str(dependency).lower().strip().startswith("ollama")
            ]
            if matches:
                unneeded_ollama_sdk_dependencies[str(extra)] = matches
    if not has_pip_audit:
        missing.append("dev:pip-audit")
    if not has_ruff:
        missing.append("dev:ruff")
    if not has_trove_classifiers:
        missing.append("dev:trove-classifiers")
    if not has_pyyaml:
        missing.append("dev:PyYAML")
    for extra, dependency_prefixes in missing_expected_dependencies.items():
        for prefix in dependency_prefixes:
            missing.append(f"{extra}:{prefix}")
    if unneeded_ollama_sdk_dependencies:
        missing.append("ollama:unneeded-sdk")
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
            "dev_has_pyyaml": has_pyyaml,
            "missing_expected_dependencies": missing_expected_dependencies,
            "unneeded_ollama_sdk_dependencies": unneeded_ollama_sdk_dependencies,
        },
        actions=[
            "Declare extras for every supported provider and dev workflow, make `crupier[all]` include all runtime SDK/file extras except Ollama's built-in REST adapter, keep Ollama without an extra SDK dependency, include pip-audit for dependency vulnerability checks, Ruff for critical lint, trove-classifiers for PyPI classifier validation, and PyYAML for public GitHub YAML release checks."
        ]
        if missing
        else [],
    )


def _default_config_check() -> ReleaseCheck:
    data = tomllib.loads(DEFAULT_TOML)
    config = CrupierConfig.from_dict(data)
    ollama = config.providers.get("ollama")
    openrouter = config.providers.get("openrouter")
    failures: list[str] = []
    builtin_model_keys = {
        f"{card['model_ref']['provider']}:{card['model_ref']['model']}" for card in BUILTIN_CAPABILITY_CARDS
    }
    missing_builtin_cards = [model for model in config.models.allow if model not in builtin_model_keys]
    if missing_builtin_cards:
        failures.append("default allowed models must have built-in capability cards: " + ", ".join(missing_builtin_cards))
    if config.orchestrator.model and config.orchestrator.model not in config.models.allow:
        failures.append("orchestrator.model must be included in the default allowlist")
    if config.orchestrator.fallback_model and config.orchestrator.fallback_model not in config.models.allow:
        failures.append("orchestrator.fallback_model must be included in the default allowlist")
    if not ollama:
        failures.append("providers.ollama missing from default config")
    else:
        if ollama.host != OLLAMA_CLOUD_HOST:
            failures.append("providers.ollama.host must default to Ollama Cloud")
        if ollama.env_key != "OLLAMA_API_KEY":
            failures.append("providers.ollama.env_key must be OLLAMA_API_KEY")
    if not openrouter:
        failures.append("providers.openrouter missing from default config")
    else:
        if openrouter.enabled:
            failures.append("providers.openrouter must default to disabled BYOK")
        if openrouter.mode != "byok":
            failures.append("providers.openrouter.mode must be byok")
        if openrouter.host != OPENROUTER_DEFAULT_HOST:
            failures.append("providers.openrouter.host must default to OpenRouter OpenAI-compatible API")
        if openrouter.env_key != "OPENROUTER_API_KEY":
            failures.append("providers.openrouter.env_key must be OPENROUTER_API_KEY")
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
    if config.routing.max_provider_retries != 1:
        failures.append("routing.max_provider_retries must default to 1")
    if config.routing.retry_backoff_seconds != 0.2:
        failures.append("routing.retry_backoff_seconds must default to 0.2")
    if f"OLLAMA_HOST={OLLAMA_CLOUD_HOST}" not in DEFAULT_ENV_EXAMPLE:
        failures.append(".env.example must advertise Ollama Cloud host")
    if "OPENROUTER_API_KEY=" not in DEFAULT_ENV_EXAMPLE:
        failures.append(".env.example must advertise optional OpenRouter BYOK env key")
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
            "max_provider_retries": config.routing.max_provider_retries,
            "retry_backoff_seconds": config.routing.retry_backoff_seconds,
            "openrouter_host": openrouter.host if openrouter else None,
            "openrouter_mode": openrouter.mode if openrouter else None,
            "missing_builtin_cards": missing_builtin_cards,
            "default_allowlist": config.models.allow,
            "orchestrator_model": config.orchestrator.model,
            "orchestrator_fallback_model": config.orchestrator.fallback_model,
        },
        actions=[
            "Keep `crupier init` defaults pointed at Ollama Cloud, keep OpenRouter disabled BYOK with an explicit OpenAI-compatible host, and keep prompt/response storage opt-in."
        ]
        if not ok
        else [],
    )


def _runtime_safety_defaults_check() -> ReleaseCheck:
    from .server import build_openai_compatible_server

    signature = inspect.signature(build_openai_compatible_server)
    parameters = signature.parameters
    failures: list[str] = []
    host_default = _signature_default(parameters, "host")
    allow_remote_default = _signature_default(parameters, "allow_remote")
    cors_origin_default = _signature_default(parameters, "cors_origin")
    if host_default != "127.0.0.1":
        failures.append("OpenAI-compatible server must default to loopback host 127.0.0.1")
    if allow_remote_default is not False:
        failures.append("OpenAI-compatible server must default allow_remote to false")
    if cors_origin_default is not None:
        failures.append("OpenAI-compatible server must default browser CORS to disabled")

    ok = not failures
    return ReleaseCheck(
        id="runtime_safety_defaults",
        status="pass" if ok else "fail",
        severity="high",
        summary="Runtime safety defaults keep local server and provider retries bounded."
        if ok
        else "Runtime safety defaults are unsafe for public onboarding.",
        evidence={
            "failures": failures,
            "server_host_default": host_default,
            "server_allow_remote_default": allow_remote_default,
            "server_cors_origin_default": cors_origin_default,
        },
        actions=[
            "Keep `crupier serve` bound to loopback by default, require explicit remote bind opt-in, and keep browser CORS disabled unless an origin is configured."
        ]
        if not ok
        else [],
    )


def _signature_default(parameters: dict[str, inspect.Parameter], name: str) -> Any:
    parameter = parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


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
    project_data = _read_pyproject(root / "pyproject.toml")
    project = project_data.get("project", {}) if isinstance(project_data, dict) else {}
    with tempfile.TemporaryDirectory(prefix="crupier-build-") as tmp:
        tmp_path = Path(tmp)
        build_root = _copy_release_source(root, tmp_path / "source")
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        command = [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)]
        result = subprocess.run(command, cwd=build_root, text=True, capture_output=True, check=False)
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        artifacts = [*sdists, *wheels]
        ok = result.returncode == 0 and bool(wheels) and bool(sdists)
        build = {
            "skipped": False,
            "ok": ok,
            "command": command,
            "source_root": str(build_root),
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
        metadata_check, metadata = _artifact_metadata_check(artifacts, project)
        checks.append(metadata_check)
        build["artifact_metadata"] = metadata
        examples_check, examples_smoke = _sdist_examples_smoke(sdists[0] if sdists else None, tmp_path)
        checks.append(examples_check)
        build["sdist_examples_smoke"] = examples_smoke
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
        build["ok"] = bool(
            ok
            and content.get("ok")
            and metadata.get("ok")
            and examples_smoke.get("ok")
            and twine.get("ok")
            and wheel_smoke.get("ok")
            and sdist_smoke.get("ok")
        )
        return checks, build


def _copy_release_source(root: Path, destination: Path) -> Path:
    """Copy release source into a clean build tree so local artifacts cannot leak."""

    tracked = _git_tracked_files(root)
    if tracked:
        destination.mkdir(parents=True)
        for relative in tracked:
            source = root / relative
            target = destination / relative
            if source.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.readlink(source))
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        return destination
    shutil.copytree(root, destination, ignore=_release_source_ignore)
    return destination


def _git_tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            text=False,
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0 or not result.stdout:
        return []
    files = []
    for raw in result.stdout.split(b"\0"):
        if raw:
            files.append(Path(raw.decode("utf-8", errors="surrogateescape")))
    return files


def _release_source_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in _RELEASE_SOURCE_IGNORED_NAMES:
            ignored.add(name)
            continue
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            ignored.add(name)
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in _RELEASE_SOURCE_IGNORED_PATTERNS):
            ignored.add(name)
    return ignored


def _artifact_content_check(artifacts: list[Path]) -> tuple[ReleaseCheck, dict[str, Any]]:
    inspected: list[dict[str, Any]] = []
    forbidden: list[str] = []
    typed_marker_present = False
    env_example_present = False
    contributing_present = False
    found_examples: set[str] = set()
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
            normalized = _artifact_relative_name(name)
            if normalized in _EXPECTED_EXAMPLE_FILES:
                found_examples.add(normalized)
    missing_examples = sorted(_EXPECTED_EXAMPLE_FILES - found_examples)
    ok = (
        bool(artifacts)
        and not forbidden
        and typed_marker_present
        and env_example_present
        and contributing_present
        and not missing_examples
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
        "expected_examples": sorted(_EXPECTED_EXAMPLE_FILES),
        "found_examples": sorted(found_examples),
        "missing_examples": missing_examples,
    }
    return (
        ReleaseCheck(
            id="artifact_content",
            status="pass" if ok else "fail",
            severity="high",
            summary="Built distributions contain expected package files and no local secret/cache/internal-doc artifacts."
            if ok
            else "Built distributions contain unexpected content or are missing py.typed/.env.example/CONTRIBUTING/public examples.",
            evidence=payload,
            actions=[
                "Inspect built distributions, include public onboarding files, and exclude local secret/cache/project config artifacts and internal planning docs before publishing."
            ]
            if not ok
            else [],
        ),
        payload,
    )


def _artifact_metadata_check(artifacts: list[Path], project: dict[str, Any]) -> tuple[ReleaseCheck, dict[str, Any]]:
    expected_name = str(project.get("name") or "")
    expected_version = str(project.get("version") or "")
    expected_summary = str(project.get("description") or "")
    expected_requires_python = str(project.get("requires-python") or "")
    expected_license = str(project.get("license") or "")
    expected_urls = project.get("urls", {}) if isinstance(project.get("urls"), dict) else {}
    expected_classifiers = project.get("classifiers", []) if isinstance(project.get("classifiers"), list) else []
    expected_extras = sorted(project.get("optional-dependencies", {})) if isinstance(project.get("optional-dependencies"), dict) else []

    inspected: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for artifact in artifacts:
        try:
            metadata_text = _read_artifact_metadata(artifact)
        except (OSError, tarfile.TarError, zipfile.BadZipFile, KeyError) as exc:
            failures.append(
                {
                    "artifact": artifact.name,
                    "field": "metadata",
                    "expected": "readable METADATA/PKG-INFO",
                    "actual": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue

        metadata = Parser().parsestr(metadata_text)
        values = {
            "Name": metadata.get("Name", ""),
            "Version": metadata.get("Version", ""),
            "Summary": metadata.get("Summary", ""),
            "Requires-Python": metadata.get("Requires-Python", ""),
            "License-Expression": metadata.get("License-Expression", ""),
            "License": metadata.get("License", ""),
            "Project-URL": metadata.get_all("Project-URL", []),
            "Classifier": metadata.get_all("Classifier", []),
            "Provides-Extra": metadata.get_all("Provides-Extra", []),
        }
        inspected.append(
            {
                "artifact": artifact.name,
                "name": values["Name"],
                "version": values["Version"],
                "requires_python": values["Requires-Python"],
                "license_expression": values["License-Expression"],
                "project_url_count": len(values["Project-URL"]),
                "classifier_count": len(values["Classifier"]),
                "provides_extra": sorted(values["Provides-Extra"]),
            }
        )
        for field, expected, actual in [
            ("Name", expected_name, values["Name"]),
            ("Version", expected_version, values["Version"]),
            ("Summary", expected_summary, values["Summary"]),
            ("Requires-Python", expected_requires_python, values["Requires-Python"]),
        ]:
            if expected and actual != expected:
                failures.append(
                    {"artifact": artifact.name, "field": field, "expected": expected, "actual": str(actual)}
                )
        if expected_license and expected_license not in {values["License-Expression"], values["License"]}:
            failures.append(
                {
                    "artifact": artifact.name,
                    "field": "License-Expression",
                    "expected": expected_license,
                    "actual": values["License-Expression"] or values["License"],
                }
            )
        project_urls = set(values["Project-URL"])
        for label, url in sorted(expected_urls.items()):
            expected_url = f"{label}, {url}"
            if expected_url not in project_urls:
                failures.append(
                    {
                        "artifact": artifact.name,
                        "field": "Project-URL",
                        "expected": expected_url,
                        "actual": "; ".join(values["Project-URL"]),
                    }
                )
        classifiers = set(values["Classifier"])
        for classifier in sorted(str(item) for item in expected_classifiers):
            if classifier not in classifiers:
                failures.append(
                    {
                        "artifact": artifact.name,
                        "field": "Classifier",
                        "expected": classifier,
                        "actual": "; ".join(values["Classifier"]),
                    }
                )
        extras = set(values["Provides-Extra"])
        for extra in expected_extras:
            if extra not in extras:
                failures.append(
                    {
                        "artifact": artifact.name,
                        "field": "Provides-Extra",
                        "expected": extra,
                        "actual": "; ".join(values["Provides-Extra"]),
                    }
                )

    ok = bool(artifacts) and not failures
    payload = {
        "ok": ok,
        "artifacts": [artifact.name for artifact in artifacts],
        "inspected": inspected,
        "failures": failures[:50],
        "failure_count": len(failures),
        "expected_extras": expected_extras,
    }
    return (
        ReleaseCheck(
            id="artifact_metadata",
            status="pass" if ok else "fail",
            severity="high",
            summary="Built distributions expose expected PyPI metadata."
            if ok
            else "Built distributions expose incomplete or incorrect PyPI metadata.",
            evidence=payload,
            actions=[
                "Fix package metadata so built wheel and sdist expose the expected name, version, Python requirement, license, project URLs, classifiers, and extras."
            ]
            if not ok
            else [],
        ),
        payload,
    )


def _read_artifact_metadata(artifact: Path) -> str:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as wheel:
            names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise KeyError(f"expected exactly one wheel METADATA file, found {len(names)}")
            return wheel.read(names[0]).decode("utf-8", errors="replace")
    if artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, "r:gz") as sdist:
            names = []
            for name in sdist.getnames():
                parts = PurePosixPath(name).parts
                if not parts or parts[-1] != "PKG-INFO":
                    continue
                if any(part.endswith(".egg-info") for part in parts):
                    continue
                names.append(name)
            if len(names) != 1:
                raise KeyError(f"expected exactly one sdist PKG-INFO file, found {len(names)}")
            handle = sdist.extractfile(names[0])
            if handle is None:
                raise KeyError("sdist PKG-INFO could not be read")
            return handle.read().decode("utf-8", errors="replace")
    raise KeyError(f"unsupported artifact type: {artifact.name}")


def _artifact_entry_is_forbidden(name: str) -> bool:
    parts = PurePosixPath(name).parts
    if parts and parts[-1] == "crupier.toml":
        return True
    for part in parts:
        if part == "docs":
            return True
        if part == "tests":
            return True
        if part == ".env":
            return True
        if part.startswith(".env.") and part != ".env.example":
            return True
        if part in {".crupier", "__pycache__", ".pytest_cache"}:
            return True
    return False


def _artifact_relative_name(name: str) -> str:
    parts = PurePosixPath(name).parts
    if not parts:
        return name
    for marker in ("examples", "src", "tests", "crupier"):
        if marker in parts:
            return str(PurePosixPath(*parts[parts.index(marker) :]))
    return str(PurePosixPath(*parts))


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


def _sdist_examples_smoke(sdist: Path | None, tmp_path: Path) -> tuple[ReleaseCheck, dict[str, Any]]:
    if sdist is None:
        return (
            ReleaseCheck(
                id="sdist_examples_smoke",
                status="fail",
                severity="high",
                summary="Sdist example smoke could not run because no sdist was built.",
                actions=["Fix package build errors before release."],
            ),
            {"skipped": True, "ok": False, "reason": "missing_sdist"},
        )
    extract_dir = tmp_path / "sdist-examples"
    try:
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(extract_dir, filter="data")
    except TypeError:
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(extract_dir)
    except (OSError, tarfile.TarError) as exc:
        return (
            ReleaseCheck(
                id="sdist_examples_smoke",
                status="fail",
                severity="high",
                summary="Sdist example smoke could not extract the built sdist.",
                evidence={"sdist": sdist.name, "error_type": exc.__class__.__name__, "error": str(exc)},
                actions=["Fix sdist archive generation before release."],
            ),
            {"skipped": False, "ok": False, "sdist": sdist.name, "error": str(exc), "steps": []},
        )

    source_roots = sorted(path for path in extract_dir.iterdir() if path.is_dir())
    source_root = source_roots[0] if source_roots else extract_dir
    examples_dir = source_root / "examples"
    src_dir = source_root / "src"
    steps: list[dict[str, Any]] = []
    ok = examples_dir.exists() and src_dir.exists()
    if ok:
        scripts = sorted(path for path in examples_dir.glob("*.py") if not path.name.startswith("_"))
        ok = bool(scripts)
    else:
        scripts = []

    for script in scripts:
        env = _example_smoke_env(src_dir)
        work_dir = tmp_path / f"example-smoke-{script.stem}"
        work_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=work_dir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        step = {
            "name": script.name,
            "returncode": result.returncode,
            "stdout_head": result.stdout[:500],
            "stderr_tail": result.stderr[-1000:],
            "created_crupier_dir": (work_dir / ".crupier").exists(),
        }
        step["ok"] = (
            result.returncode == 0
            and "strategy=" in result.stdout
            and "models=" in result.stdout
            and not step["created_crupier_dir"]
        )
        steps.append(step)
        ok = ok and bool(step["ok"])

    payload = {
        "skipped": False,
        "ok": ok,
        "sdist": sdist.name,
        "source_root": str(source_root),
        "scripts": [script.name for script in scripts],
        "steps": steps,
    }
    return (
        ReleaseCheck(
            id="sdist_examples_smoke",
            status="pass" if ok else "fail",
            severity="high",
            summary="Public examples from the built sdist run without provider keys or local artifact writes."
            if ok
            else "One or more public examples from the built sdist failed offline smoke validation.",
            evidence={
                "sdist": sdist.name,
                "scripts": payload["scripts"],
                "steps": [
                    {key: value for key, value in step.items() if key in {"name", "returncode", "ok"}}
                    for step in steps
                ],
            },
            actions=[
                "Keep packaged examples executable without provider keys, ensure they print route decisions, and avoid writing .crupier artifacts."
            ]
            if not ok
            else [],
        ),
        payload,
    )


def _example_smoke_env(src_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    env.pop("GEMINI_API_KEY", None)
    env.pop("OLLAMA_API_KEY", None)
    env.pop("OPENROUTER_API_KEY", None)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_dir) if not existing else f"{src_dir}{os.pathsep}{existing}"
    return env


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
    import_lines: list[str] = []
    if ok:
        imported = subprocess.run(
            [str(python), "-c", _public_api_import_code()],
            text=True,
            capture_output=True,
            check=False,
        )
        import_lines = [line.strip() for line in imported.stdout.splitlines() if line.strip()]
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
        cli_version = subprocess.run([str(script), "--version"], text=True, capture_output=True, check=False)
        steps.append(
            {"name": "cli_version", "returncode": cli_version.returncode, "stdout": cli_version.stdout.strip()}
        )
        ok = cli_version.returncode == 0 and bool(import_lines and import_lines[0] in cli_version.stdout)
    else:
        cli_version = None
    if ok:
        module_version = subprocess.run(
            [str(python), "-m", "crupier", "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        steps.append(
            {"name": "module_version", "returncode": module_version.returncode, "stdout": module_version.stdout.strip()}
        )
        ok = module_version.returncode == 0 and bool(import_lines and import_lines[0] in module_version.stdout)
    else:
        module_version = None
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
        "import_version": import_lines[0] if imported and imported.returncode == 0 and import_lines else None,
        "public_export_count": int(import_lines[1]) if len(import_lines) > 1 and import_lines[1].isdigit() else None,
    }
    return (
        ReleaseCheck(
            id="wheel_install_smoke",
            status="pass" if ok else "fail",
            severity="high",
            summary=(
                "Built wheel installs, imports, validates public exports, exposes crupier CLI/module help/version, "
                "runs crupier init, routes dry-run, and executes the Python SDK quickstart."
            )
            if ok
            else "Built wheel failed install/import/CLI/module/help/version/init smoke.",
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
    import_lines: list[str] = []
    if ok:
        imported = subprocess.run(
            [str(python), "-c", _public_api_import_code()],
            text=True,
            capture_output=True,
            check=False,
        )
        import_lines = [line.strip() for line in imported.stdout.splitlines() if line.strip()]
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
        cli_version = subprocess.run([str(script), "--version"], text=True, capture_output=True, check=False)
        steps.append(
            {"name": "cli_version", "returncode": cli_version.returncode, "stdout": cli_version.stdout.strip()}
        )
        ok = cli_version.returncode == 0 and bool(import_lines and import_lines[0] in cli_version.stdout)
    else:
        cli_version = None
    if ok:
        module_version = subprocess.run(
            [str(python), "-m", "crupier", "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        steps.append(
            {"name": "module_version", "returncode": module_version.returncode, "stdout": module_version.stdout.strip()}
        )
        ok = module_version.returncode == 0 and bool(import_lines and import_lines[0] in module_version.stdout)
    else:
        module_version = None
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
        "import_version": import_lines[0] if imported and imported.returncode == 0 and import_lines else None,
        "public_export_count": int(import_lines[1]) if len(import_lines) > 1 and import_lines[1].isdigit() else None,
    }
    return (
        ReleaseCheck(
            id="sdist_install_smoke",
            status="pass" if ok else "fail",
            severity="high",
            summary=(
                "Built sdist installs, imports, validates public exports, exposes crupier CLI/module help/version, "
                "runs crupier init, routes dry-run, and executes the Python SDK quickstart."
            )
            if ok
            else "Built sdist failed install/import/CLI/module/help/version/init smoke.",
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
        and 'host = "https://openrouter.ai/api/v1"' in toml_text
        and "OLLAMA_HOST=https://ollama.com/api" in env_text
        and "OPENROUTER_API_KEY=" in env_text
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


def _public_api_import_code() -> str:
    return (
        "import crupier\n"
        "missing = [name for name in crupier.__all__ if not hasattr(crupier, name)]\n"
        "if missing:\n"
        "    raise SystemExit('missing public exports: ' + ','.join(missing))\n"
        "print(crupier.__version__)\n"
        "print(len(crupier.__all__))\n"
    )


def format_release_check_report(report: ReleaseCheckReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

import crupier.release as release


class _Response:
    def __init__(self, status: int, *, url: str = "https://example.test/final") -> None:
        self.status = status
        self.url = url
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _write_project_urls(root: Path, urls: dict[str, str] | None = None) -> None:
    values = urls or {}
    lines = ["[project]", 'name = "demo"', "", "[project.urls]"]
    lines.extend(f'{label} = "{url}"' for label, url in values.items())
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_pypi_name_rejects_missing_name_without_network() -> None:
    check = release.check_pypi_project_name("unknown")

    assert check.status == "fail"
    assert check.evidence["normalized"] == "unknown"


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (HTTPError("https://pypi.test", 503, "Unavailable", {}, None), "warn"),
        (URLError("offline"), "warn"),
    ],
)
def test_pypi_name_reports_network_failures(monkeypatch, failure: Exception, expected_status: str) -> None:
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(release, "urlopen", fail)

    check = release.check_pypi_project_name("demo")

    assert check.status == expected_status


def test_pypi_name_closes_and_reports_unexpected_response(monkeypatch) -> None:
    response = _Response(503)
    monkeypatch.setattr(release, "urlopen", lambda *_args, **_kwargs: response)

    check = release.check_pypi_project_name("demo")

    assert check.status == "warn"
    assert check.evidence["http_status"] == 503
    assert response.closed is True


def test_project_url_reachability_requires_valid_metadata(tmp_path: Path) -> None:
    _write_project_urls(tmp_path, {"Docs": "ftp://example.test/docs"})

    check = release.check_project_urls_reachable(tmp_path)

    assert check.status == "fail"
    assert check.evidence["project_urls_check"]["id"] == "project_urls"


def test_project_url_reachability_collects_every_failure(tmp_path: Path, monkeypatch) -> None:
    _write_project_urls(
        tmp_path,
        {
            "Repository": "https://one.test/repo",
            "Issues": "https://two.test/issues",
            "Changelog": "https://three.test/changelog",
        },
    )
    responses = iter(
        [
            HTTPError("https://one.test/repo", 404, "missing", {}, None),
            URLError("offline"),
            _Response(503),
        ]
    )

    def fake_urlopen(*_args, **_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(release, "urlopen", fake_urlopen)

    check = release.check_project_urls_reachable(tmp_path)

    assert check.status == "fail"
    assert len(check.evidence["checked"]) == 3
    assert len(check.evidence["failures"]) == 3


def test_read_pyproject_and_project_url_edge_cases(tmp_path: Path) -> None:
    assert release._read_pyproject(tmp_path / "missing.toml") == {}

    check = release._project_urls_check(
        {
            "urls": {
                "Repository": "ssh://git@example.test/demo",
                "Docs": "https://example.com/todo",
            }
        }
    )
    assert check.status == "warn"
    assert check.evidence["placeholders"] == ["Repository", "Docs"]


def test_public_release_language_ignores_missing_files(tmp_path: Path) -> None:
    check = release._public_release_language_check(tmp_path)

    assert check.status == "pass"


def test_secret_scan_handles_untracked_candidates_large_files_and_filters(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "ok.py").write_text("print('safe')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=local-only\n", encoding="utf-8")
    (tmp_path / "large.txt").write_bytes(b"x" * (release._SECRET_SCAN_MAX_BYTES + 1))
    (tmp_path / "image.png").write_bytes(b"not-an-image")
    (tmp_path / "cache.pyc").write_bytes(b"cache")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "link.py").symlink_to(tmp_path / "ok.py")
    monkeypatch.setattr(release, "_git_tracked_files", lambda *_args, **_kwargs: [])

    check = release._public_secret_scan_check(tmp_path)

    assert check.status == "pass"
    assert "large.txt" in check.evidence["skipped_large"]
    assert ".env" not in check.evidence["checked_sample"]
    assert release._should_secret_scan_path(Path(".venv/key.txt"), include_env_files=True) is False
    assert release._should_secret_scan_path(Path(".env.local"), include_env_files=False) is False
    assert release._should_secret_scan_path(Path("module.pyo"), include_env_files=True) is False
    assert release._should_secret_scan_path(Path("archive.tar.gz"), include_env_files=True) is False
    assert release._should_secret_scan_path(Path("src/module.py"), include_env_files=True) is True


def test_secret_scan_caps_findings(tmp_path: Path, monkeypatch) -> None:
    secret = "sk-proj-" + "a" * 40
    (tmp_path / "many.txt").write_text("\n".join([secret] * 60), encoding="utf-8")
    monkeypatch.setattr(release, "_git_tracked_files", lambda *_args, **_kwargs: [Path("many.txt")])

    check = release._public_secret_scan_check(tmp_path)

    assert check.status == "fail"
    assert check.evidence["finding_count"] == 50


def test_public_yaml_rejects_empty_and_non_mapping_documents(tmp_path: Path) -> None:
    github = tmp_path / ".github"
    github.mkdir()
    (github / "empty.yml").write_text("", encoding="utf-8")
    (github / "list.yaml").write_text("- one\n- two\n", encoding="utf-8")

    check = release._public_yaml_check(tmp_path)

    assert check.status == "fail"
    assert {item["error_type"] for item in check.evidence["failures"]} == {
        "EmptyDocument",
        "InvalidTopLevel",
    }


def test_markdown_link_helpers_cover_external_anchor_and_escape(tmp_path: Path) -> None:
    source = Path("README.md")
    (tmp_path / source).write_text("# docs\n", encoding="utf-8")
    (tmp_path / "exists.md").write_text("ok\n", encoding="utf-8")
    text = (
        "[anchor](#title) [cdn](//cdn.example.test/x) [web](https://example.test) "
        "[exists](exists.md?raw=1) [missing](missing.md) [escape](../outside.md)"
    )

    broken = release._broken_relative_markdown_links(tmp_path.resolve(), source, text)

    assert {item["reason"] for item in broken} == {"missing target", "points outside repository"}
    assert release._relative_markdown_links(text) == ["exists.md?raw=1", "missing.md", "../outside.md"]


def test_optional_dependencies_handles_invalid_shapes_and_unneeded_sdk() -> None:
    check = release._optional_dependencies_check(
        {
            "optional-dependencies": {
                "openai": "not-a-list",
                "ollama": ["ollama>=1"],
                "dev": ["pytest>=9"],
            }
        }
    )

    assert check.status == "warn"
    assert "ollama:unneeded-sdk" in check.evidence["missing"]
    assert check.evidence["missing_expected_dependencies"]["openai"] == ["openai"]


def _unsafe_default_config(*, include_providers: bool) -> SimpleNamespace:
    providers = {}
    if include_providers:
        providers = {
            "ollama": SimpleNamespace(host="http://localhost", env_key="WRONG"),
            "openrouter": SimpleNamespace(enabled=True, mode="managed", host="http://wrong", env_key="WRONG"),
            "inference": SimpleNamespace(
                enabled=True,
                mode="managed",
                host="http://wrong",
                options={"auth": "required"},
            ),
        }
    return SimpleNamespace(
        providers=providers,
        models=SimpleNamespace(allow=["missing:model"]),
        orchestrator=SimpleNamespace(mode="deterministic", model="other:model", fallback_model="fallback:model"),
        logging=SimpleNamespace(
            persist_traces=True,
            store_prompts=True,
            store_responses=True,
            redact_secrets=False,
        ),
        routing=SimpleNamespace(
            allow_latest_aliases=True,
            allow_preview_models=True,
            max_provider_retries=9,
            max_latency_ms=30000,
            retry_backoff_seconds=3.0,
            require_operational_providers=False,
        ),
    )


@pytest.mark.parametrize("include_providers", [False, True])
def test_default_config_check_reports_all_unsafe_defaults(monkeypatch, include_providers: bool) -> None:
    config = _unsafe_default_config(include_providers=include_providers)
    monkeypatch.setattr(release.CrupierConfig, "from_dict", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(release, "BUILTIN_CAPABILITY_CARDS", [])
    monkeypatch.setattr(release, "DEFAULT_ENV_EXAMPLE", "")

    check = release._default_config_check()

    assert check.status == "fail"
    assert len(check.evidence["failures"]) >= 15
    assert check.evidence["missing_builtin_cards"] == ["missing:model"]


def test_runtime_safety_check_rejects_remote_defaults(monkeypatch) -> None:
    parameters = {
        "host": inspect.Parameter("host", inspect.Parameter.KEYWORD_ONLY, default="0.0.0.0"),
        "allow_remote": inspect.Parameter("allow_remote", inspect.Parameter.KEYWORD_ONLY, default=True),
        "cors_origin": inspect.Parameter("cors_origin", inspect.Parameter.KEYWORD_ONLY, default="*"),
    }
    monkeypatch.setattr(release.inspect, "signature", lambda _callable: SimpleNamespace(parameters=parameters))

    check = release._runtime_safety_defaults_check()

    assert check.status == "fail"
    assert len(check.evidence["failures"]) == 3
    assert release._signature_default({}, "missing") is None


def test_build_distribution_check_handles_missing_python(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(release.shutil, "which", lambda _executable: None)

    checks, payload = release._build_distribution_checks(tmp_path)

    assert checks[0].status == "fail"
    assert payload["error"] == "python executable unavailable"


@pytest.mark.parametrize("successful", [False, True])
def test_build_distribution_check_orchestrates_all_artifact_checks(tmp_path: Path, monkeypatch, successful: bool) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    monkeypatch.setattr(release.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(release, "_copy_release_source", lambda root, _destination: root)

    def fake_build(command, **_kwargs):
        dist_dir = Path(command[-1])
        if successful:
            (dist_dir / "demo-0.1.0.tar.gz").write_bytes(b"sdist")
            (dist_dir / "demo-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        return SimpleNamespace(returncode=0 if successful else 1, stdout="", stderr="build output")

    monkeypatch.setattr(release.subprocess, "run", fake_build)

    def check_result(identifier: str):
        payload = {"ok": successful}
        return release.ReleaseCheck(id=identifier, status="pass" if successful else "fail", summary=identifier), payload

    monkeypatch.setattr(release, "_artifact_content_check", lambda _artifacts: check_result("artifact_content"))
    monkeypatch.setattr(release, "_artifact_metadata_check", lambda _artifacts, _project: check_result("artifact_metadata"))
    monkeypatch.setattr(release, "_sdist_examples_smoke", lambda _sdist, _tmp: check_result("sdist_examples_smoke"))
    monkeypatch.setattr(release, "_twine_check", lambda _artifacts: check_result("twine_check"))
    monkeypatch.setattr(release, "_wheel_install_smoke", lambda _wheel, _tmp: check_result("wheel_install_smoke"))
    monkeypatch.setattr(release, "_sdist_install_smoke", lambda _sdist, _tmp: check_result("sdist_install_smoke"))

    checks, payload = release._build_distribution_checks(tmp_path)

    assert len(checks) == 7
    assert payload["ok"] is successful
    assert payload["artifact_count"] == (2 if successful else 0)


def test_release_source_helpers_cover_symlink_git_failure_and_patterns(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "target.txt").write_text("target\n", encoding="utf-8")
    (source / "link.txt").symlink_to(source / "target.txt")
    original_git_tracked_files = release._git_tracked_files
    monkeypatch.setattr(release, "_git_tracked_files", lambda *_args, **_kwargs: [Path("link.txt")])

    copied = release._copy_release_source(source, tmp_path / "copied")

    assert (copied / "link.txt").is_symlink()
    assert release._release_source_path_ignored(Path("src/cache.pyc")) is True
    assert release._release_source_path_ignored(Path("src/code.py")) is False

    def fail_run(*_args, **_kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(release.subprocess, "run", fail_run)
    assert original_git_tracked_files(source) == []


def test_artifact_metadata_reports_unreadable_and_every_missing_field(tmp_path: Path) -> None:
    unsupported = tmp_path / "demo.bin"
    unsupported.write_bytes(b"not metadata")
    project = {
        "name": "demo",
        "version": "0.1.0",
        "description": "Demo",
        "requires-python": ">=3.11",
        "license": "MIT",
        "urls": {"Repository": "https://example.test/demo"},
        "classifiers": ["Typing :: Typed"],
        "optional-dependencies": {"dev": []},
    }

    check, payload = release._artifact_metadata_check([unsupported], project)

    assert check.status == "fail"
    assert payload["failures"][0]["field"] == "metadata"

    wheel = tmp_path / "demo.whl"
    import zipfile

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo.dist-info/METADATA", "Metadata-Version: 2.4\nName: wrong\n\n")

    _, payload = release._artifact_metadata_check([wheel], project)
    fields = {item["field"] for item in payload["failures"]}
    assert {"Name", "Version", "Summary", "Requires-Python", "License-Expression", "Project-URL", "Classifier", "Provides-Extra"} <= fields


def test_artifact_helpers_reject_invalid_metadata_and_private_entries(tmp_path: Path) -> None:
    import tarfile
    import zipfile

    wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("one.dist-info/METADATA", "one")
        archive.writestr("two.dist-info/METADATA", "two")
    with pytest.raises(KeyError, match="exactly one"):
        release._read_artifact_metadata(wheel)

    sdist = tmp_path / "bad.tar.gz"
    source = tmp_path / "source.txt"
    source.write_text("data", encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="demo/src/demo.egg-info/PKG-INFO")
    with pytest.raises(KeyError, match="exactly one"):
        release._read_artifact_metadata(sdist)

    with pytest.raises(KeyError, match="unsupported"):
        release._read_artifact_metadata(tmp_path / "artifact.zip")

    for name in ["pkg/.env.local", "pkg/.crupier/x", "pkg/__pycache__/x.pyc", "pkg/.pytest_cache/x"]:
        assert release._artifact_entry_is_forbidden(name) is True
    assert release._artifact_entry_is_forbidden("pkg/.env.example") is False
    assert release._artifact_relative_name("README.md") == "README.md"


@pytest.mark.parametrize("returncode", [0, 1])
def test_twine_check_reports_process_result(tmp_path: Path, monkeypatch, returncode: int) -> None:
    artifact = tmp_path / "demo.whl"
    artifact.write_bytes(b"wheel")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode, stdout="stdout", stderr="stderr"),
    )

    check, payload = release._twine_check([artifact])

    assert check.status == ("pass" if returncode == 0 else "fail")
    assert payload["ok"] is (returncode == 0)


def test_missing_artifacts_are_explicit_failures(tmp_path: Path) -> None:
    checks = [
        release._twine_check([]),
        release._sdist_examples_smoke(None, tmp_path),
        release._wheel_install_smoke(None, tmp_path),
        release._sdist_install_smoke(None, tmp_path),
    ]

    assert all(check.status == "fail" and payload["skipped"] for check, payload in checks)


def test_sdist_examples_reports_invalid_archive(tmp_path: Path) -> None:
    sdist = tmp_path / "broken.tar.gz"
    sdist.write_bytes(b"broken")

    check, payload = release._sdist_examples_smoke(sdist, tmp_path)

    assert check.status == "fail"
    assert payload["steps"] == []


@pytest.mark.parametrize("smoke", [release._wheel_install_smoke, release._sdist_install_smoke])
def test_install_smoke_stops_cleanly_when_venv_creation_fails(tmp_path: Path, monkeypatch, smoke) -> None:
    artifact = tmp_path / ("demo.whl" if smoke is release._wheel_install_smoke else "demo.tar.gz")
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(
        release.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="venv failed"),
    )

    check, payload = smoke(artifact, tmp_path)

    assert check.status == "fail"
    assert [step["name"] for step in payload["steps"]] == ["create_venv"]


def test_installed_init_smoke_handles_non_json_route(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    responses = iter(
        [
            SimpleNamespace(returncode=1, stdout="", stderr="init failed"),
            SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="sdk failed"),
        ]
    )
    monkeypatch.setattr(release.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    ok, payload = release._installed_init_smoke(Path("crupier"), Path("python"), project)

    assert ok is False
    assert payload["route_ok"] is False


def test_release_report_formatter_is_stable_json() -> None:
    report = release.ReleaseCheckReport(
        project="demo",
        version="0.4.0",
        checks=[release.ReleaseCheck(id="ready", status="pass", summary="Ready")],
    )

    payload = json.loads(release.format_release_check_report(report))

    assert payload["ok"] is True
    assert payload["summary"] == {"pass": 1}

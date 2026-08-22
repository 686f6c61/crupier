import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crupier.errors import CrupierConfigError, CrupierModelUnsupportedError
from crupier.models import FileAsset, FileRepresentation, FileRoutingPlan
from crupier.multimodal import (
    _extract_pdf_text,
    _guess_mime,
    _kind_for,
    _looks_local,
    _name_from_uri,
    can_execute_native_images,
    native_file_payloads,
    native_image_payloads,
    normalize_file,
    normalize_files,
    plan_file_representations,
    prepare_extracted_file_context,
    split_file_execution_inputs,
)


def test_normalize_files_accepts_supported_reference_forms(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")
    existing = FileAsset(kind="image", name="ready.png")

    assets = normalize_files(
        (
            existing,
            {"path": str(text), "metadata": {"source": "test"}},
            b"binary",
            "https://example.com/report.pdf",
        )
    )

    assert normalize_files(None) == []
    assert assets[0] is existing
    assert assets[1].kind == "text"
    assert assets[1].exists is True
    assert assets[1].size_bytes == 5
    assert assets[1].metadata == {"source": "test"}
    assert assets[2] == FileAsset(kind="binary", name="bytes", size_bytes=6, exists=True)
    assert assets[3].kind == "pdf"
    assert assets[3].name == "report.pdf"
    assert assets[3].exists is None


def test_normalize_file_rejects_unsupported_reference():
    with pytest.raises(TypeError, match="Unsupported file reference int"):
        normalize_file(42)


@pytest.mark.parametrize(
    ("reference", "kind"),
    [
        ("photo.heic", "image"),
        ("call.opus", "audio"),
        ("clip.m4v", "video"),
        ("ledger.parquet", "spreadsheet"),
        ("brief.odt", "document"),
        ("main.rs", "code"),
        ("config.yaml", "text"),
        ("archive.bin", "unknown"),
    ],
)
def test_normalize_file_classifies_extension_families(reference, kind):
    assert normalize_file(reference).kind == kind


@pytest.mark.parametrize(
    ("mime_type", "kind"),
    [
        ("image/avif", "image"),
        ("application/pdf", "pdf"),
        ("audio/x-custom", "audio"),
        ("video/x-custom", "video"),
        ("text/x-custom", "text"),
    ],
)
def test_normalize_file_classifies_mime_families(mime_type, kind):
    assert normalize_file({"name": "payload", "mime_type": mime_type}).kind == kind


def test_normalize_file_reads_data_uri_metadata_without_decoding_payload():
    image = normalize_file("data:image/png;base64,YWJj")
    generic = normalize_file("data:application/octet-stream;base64,YWJj")

    assert image == FileAsset(
        kind="image",
        name="image",
        uri="data:image/png;base64,YWJj",
        mime_type="image/png",
        exists=True,
    )
    assert generic.name == "data"
    assert generic.kind == "unknown"


def test_plan_file_representations_covers_every_file_family():
    assets = [
        FileAsset(kind="image", name="scan.png"),
        FileAsset(kind="pdf", name="invoice.pdf"),
        FileAsset(kind="audio", name="call.mp3"),
        FileAsset(kind="video", name="demo.mp4"),
        FileAsset(kind="spreadsheet", name="data.xlsx"),
        FileAsset(kind="document", name="brief.docx"),
        FileAsset(kind="code", name="app.py"),
        FileAsset(kind="text", name="notes.txt"),
        FileAsset(kind="unknown", name="archive.bin", exists=False),
    ]

    plan = plan_file_representations(assets, task="Review invoice table")

    assert plan is not None
    assert [item.representation for item in plan.representations] == [
        "native_vision",
        "table_rows_and_text",
        "native_audio",
        "transcript_and_frames",
        "table_rows",
        "extracted_text",
        "code_chunks",
        "inline_text",
        "metadata_only",
    ]
    assert plan.required_model_modalities == ["audio", "image", "text"]
    assert plan.required_model_capabilities == ["audio_input", "vision_input"]
    assert "file_path_not_found" in plan.warnings
    assert "unknown_file_kind" in plan.warnings
    assert plan.extraction_required is True


def test_plan_file_representations_honors_native_and_extract_strategies():
    assets = [
        FileAsset(kind="image", name="scan.png"),
        FileAsset(kind="pdf", name="report.pdf"),
        FileAsset(kind="audio", name="call.mp3"),
        FileAsset(kind="video", name="clip.mp4"),
    ]

    native = plan_file_representations(assets, constraints={"file_strategy": "native"})
    extracted_image = plan_file_representations([assets[0]], constraints={"file_strategy": "text"})
    extracted_audio = plan_file_representations([assets[2]], constraints={"file_strategy": "extract"})

    assert native is not None
    assert [item.representation for item in native.representations] == [
        "native_vision",
        "native_pdf",
        "native_audio",
        "native_video",
    ]
    assert set(native.required_model_capabilities) == {
        "audio_input",
        "file_input",
        "pdf_native_input",
        "video_input",
        "vision_input",
    }
    assert extracted_image.representations[0].pipeline == ["ocr"]
    assert extracted_audio.representations[0].pipeline == ["audio_transcription"]


def test_plan_and_split_native_and_extracted_inputs():
    image = FileAsset(kind="image", name="image.png")
    text = FileAsset(kind="text", name="notes.txt")
    plan = plan_file_representations([image, text])

    assert plan_file_representations([]) is None
    assert can_execute_native_images(None) is False
    assert can_execute_native_images(FileRoutingPlan()) is False
    assert can_execute_native_images(plan) is False
    forced_extraction = FileRoutingPlan(
        assets=[image],
        representations=[FileRepresentation("image.png", "image", "native_vision")],
        extraction_required=True,
    )
    assert can_execute_native_images(forced_extraction) is False

    native, extraction = split_file_execution_inputs(plan)
    assert native == [image]
    assert extraction is not None
    assert extraction.assets == [text]
    assert extraction.required_model_modalities == ["text"]
    assert extraction.extraction_required is False

    image_plan = plan_file_representations([image])
    assert can_execute_native_images(image_plan) is True
    native_only, no_extraction = split_file_execution_inputs(image_plan)
    assert native_only == [image]
    assert no_extraction is None


def test_file_reference_helpers_cover_suffix_url_and_platform_forms():
    assert _guess_mime(None) is None
    assert _name_from_uri(None) is None
    assert _name_from_uri("https://example.com/files/report.pdf?download=1") == "report.pdf"
    assert _looks_local(r"C:\files\report.pdf") is True
    for suffix, expected in [
        (".jpg", "image"),
        (".wav", "audio"),
        (".mov", "video"),
        (".xlsx", "spreadsheet"),
        (".docx", "document"),
    ]:
        assert _kind_for(None, suffix) == expected
    assert _kind_for(None, ".pdf") == "pdf"


def test_host_file_policy_covers_data_remote_rooted_and_denied_references(tmp_path):
    from crupier import multimodal

    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    rooted_asset = FileAsset(kind="text", name="inside.txt", uri="inside.txt")
    rooted_dict = {"path": "inside.txt"}
    files = [
        {},
        {"uri": "data:text/plain,ok"},
        {"url": "https://example.com/remote.txt"},
        rooted_dict,
        rooted_asset,
    ]

    multimodal.enforce_file_access_policy(files, allowed_root=root)

    assert rooted_dict["uri"] == str(inside)
    assert rooted_asset.uri == str(inside)
    assert rooted_asset.metadata["file_root"] == str(root.resolve())
    assert multimodal.is_host_filesystem_uri(None) is False
    assert multimodal.is_host_filesystem_uri("data:text/plain,ok") is False
    assert multimodal.is_host_filesystem_uri("file:///tmp/example") is True

    with pytest.raises(CrupierConfigError, match="cannot reference local paths"):
        multimodal.enforce_file_access_policy([{"uri": str(inside)}], allow_host_paths=False)


def test_file_root_resolution_rejects_escape_and_missing_and_decodes_file_uris(tmp_path):
    from crupier import multimodal

    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(CrupierConfigError, match="escapes"):
        multimodal.resolve_path_inside_root(str(root), root)
    with pytest.raises(CrupierConfigError, match="does not exist"):
        multimodal.resolve_path_inside_root("missing.txt", root)

    local = multimodal._path_from_filesystem_uri("file://localhost/tmp/a%20b.txt")
    remote = multimodal._path_from_filesystem_uri("file://server/share/a.txt")
    assert local == Path("/tmp/a b.txt")
    assert remote == Path("/server/share/a.txt")


def test_bounded_data_url_rejects_shape_encoding_and_size():
    from crupier import multimodal

    for uri, message in [
        ("data:text/plain", "Invalid data URL"),
        ("data:text/plain;base64,***", "Invalid data URL encoding"),
        ("data:text/plain,abcd", "above max 3 bytes"),
    ]:
        with pytest.raises(CrupierConfigError, match=message):
            multimodal._require_bounded_data_url(uri, 3)


def test_native_payload_and_local_extraction_honor_file_root(tmp_path):
    from crupier import multimodal

    root = tmp_path / "root"
    root.mkdir()
    image = root / "image.png"
    image.write_bytes(b"image")
    asset = FileAsset(
        kind="image",
        name="image.png",
        uri="image.png",
        metadata={"file_root": str(root)},
    )

    payload = native_image_payloads([asset])[0]
    assert base64.b64decode(payload["base64"]) == b"image"
    assert multimodal._local_path(asset) == image


def test_extracted_pdf_and_ocr_context_propagate_truncation_and_adapter(tmp_path, monkeypatch):
    from crupier import multimodal

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")
    pdf_asset = normalize_file(pdf)
    pdf_plan = plan_file_representations([pdf_asset])
    monkeypatch.setattr(
        multimodal,
        "_extract_pdf_text",
        lambda *args, **kwargs: ("clipped", True),
    )
    context = prepare_extracted_file_context([pdf_asset], pdf_plan)
    assert "pdf_text_truncated" in context["warnings"]

    image = tmp_path / "scan.png"
    image.write_bytes(b"image")
    image_asset = normalize_file(image)
    image_plan = plan_file_representations([image_asset], constraints={"file_strategy": "extract"})

    class OCR:
        name = "test"

        def extract(self, path, *, max_chars, timeout_seconds=None):
            assert path == image
            assert timeout_seconds == 7
            return SimpleNamespace(
                text="recognized",
                warnings=[],
                extractor="test-ocr",
                truncated=False,
                tables=[],
                to_prompt_text=lambda: "recognized",
                to_dict=lambda include_content=False: {"extractor": "test-ocr"},
            )

    ocr_context = prepare_extracted_file_context(
        [image_asset],
        image_plan,
        ocr_adapter=OCR(),
        ocr_timeout_seconds=7,
    )
    assert "recognized" in ocr_context["body"]


def test_pypdf_marks_character_truncation_and_wraps_parse_errors(tmp_path, monkeypatch):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")

    class TruncatingReader:
        def __init__(self, path):
            self.pages = [SimpleNamespace(extract_text=lambda: "long text")]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=TruncatingReader))
    assert _extract_pdf_text(
        normalize_file(pdf),
        max_file_bytes=100,
        max_chars=4,
        max_pages=10,
    ) == ("long", True)

    class BrokenReader:
        def __init__(self, path):
            raise ValueError("corrupt PDF")

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=BrokenReader))
    with pytest.raises(CrupierModelUnsupportedError, match="failed: corrupt PDF"):
        _extract_pdf_text(
            normalize_file(pdf),
            max_file_bytes=100,
            max_chars=100,
            max_pages=10,
        )


def test_native_file_payloads_encode_local_and_data_url_assets(tmp_path):
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"local-image")
    local = normalize_file(image_path)
    encoded = base64.b64encode(b"data-image").decode("ascii")
    data_asset = normalize_file(f"data:image/png;base64,{encoded}")
    percent_asset = normalize_file("data:image/png,hello%20image")

    payloads = native_image_payloads([local, data_asset, percent_asset])

    assert base64.b64decode(payloads[0]["base64"]) == b"local-image"
    assert payloads[0]["name"] == "photo.png"
    assert base64.b64decode(payloads[1]["base64"]) == b"data-image"
    assert base64.b64decode(payloads[2]["base64"]) == b"hello image"
    assert all(item["data_url"].startswith("data:image/png;base64,") for item in payloads)


@pytest.mark.parametrize(
    ("asset", "allowed", "max_bytes", "message"),
    [
        (FileAsset(kind="audio", name="call.wav"), {"image"}, 100, "expected one of: image"),
        (FileAsset(kind="image", name="remote.png", uri="https://example.com/remote.png"), {"image"}, 100, "local file path or data URL"),
        (FileAsset(kind="image", name="missing.png", uri="/definitely/missing.png"), {"image"}, 100, "does not exist"),
        (FileAsset(kind="image", name="bad", uri="data:image/png;base64"), {"image"}, 100, "invalid data URL"),
        (FileAsset(kind="image", name="audio", uri="data:audio/wav;base64,YQ=="), {"image"}, 100, "expected one of: image"),
        (FileAsset(kind="image", name="bad", uri="data:image/png;base64,***"), {"image"}, 100, "invalid data URL encoding"),
        (FileAsset(kind="image", name="large", uri="data:image/png;base64,YWJj"), {"image"}, 2, "above max 2 bytes"),
    ],
)
def test_native_file_payloads_reject_invalid_or_unsupported_assets(asset, allowed, max_bytes, message):
    with pytest.raises(CrupierModelUnsupportedError, match=message):
        native_file_payloads([asset], allowed_kinds=allowed, max_bytes=max_bytes)


def test_native_file_payloads_enforce_local_size_limit(tmp_path):
    path = tmp_path / "large.png"
    path.write_bytes(b"1234")

    with pytest.raises(CrupierModelUnsupportedError, match="above max 3 bytes"):
        native_image_payloads([normalize_file(path)], max_bytes=3)


def test_prepare_extracted_context_preserves_duplicate_name_order_and_budget(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "same.txt"
    second_path = second_dir / "same.txt"
    first_path.write_text("abc", encoding="utf-8")
    second_path.write_text("defgh", encoding="utf-8")
    first = FileAsset(kind="text", name="same.txt", uri=str(first_path))
    second = FileAsset(kind="code", name="same.txt", uri=str(second_path))
    plan = FileRoutingPlan(
        assets=[first, second],
        representations=[
            FileRepresentation("same.txt", "text", "inline_text"),
            FileRepresentation("same.txt", "code", "code_chunks", pipeline=["code_chunking"]),
        ],
        extraction_required=True,
    )

    context = prepare_extracted_file_context([first, second], plan, max_chars=5)

    assert [item["representation"] for item in context["files"]] == ["inline_text", "code_chunks"]
    assert context["files"][0]["chars"] == 3
    assert context["files"][1]["chars"] == 2
    assert context["files"][1]["truncated"] is True
    assert "abc" in context["body"] and "de" in context["body"]
    assert any("truncated" in warning for warning in context["warnings"])


def test_prepare_extracted_context_reports_empty_budget_and_missing_representation(tmp_path):
    first = tmp_path / "empty.txt"
    second = tmp_path / "later.txt"
    first.write_text("", encoding="utf-8")
    second.write_text("later", encoding="utf-8")
    assets = [normalize_file(first), normalize_file(second)]
    plan = plan_file_representations(assets)

    context = prepare_extracted_file_context(assets, plan, max_chars=0)

    assert any("extracted text was empty" in warning for warning in context["warnings"])
    assert any("file context budget exhausted" in warning for warning in context["warnings"])
    with pytest.raises(CrupierModelUnsupportedError, match="No file routing plan"):
        prepare_extracted_file_context(assets, None)
    with pytest.raises(CrupierModelUnsupportedError, match="No file representation"):
        prepare_extracted_file_context([assets[0]], FileRoutingPlan())


def test_prepare_extracted_context_rejects_remote_missing_large_and_unsupported_files(tmp_path):
    large = tmp_path / "large.txt"
    large.write_text("1234", encoding="utf-8")
    cases = [
        FileAsset(kind="text", name="remote.txt", uri="https://example.com/remote.txt"),
        FileAsset(kind="text", name="missing.txt", uri=str(tmp_path / "missing.txt")),
        normalize_file(large),
        FileAsset(kind="audio", name="call.mp3", uri=str(large)),
    ]
    limits = [100, 100, 3, 100]
    messages = ["local path", "does not exist", "above max 3 bytes", "is not supported"]

    for asset, limit, message in zip(cases, limits, messages, strict=True):
        plan = plan_file_representations([asset])
        with pytest.raises(CrupierModelUnsupportedError, match=message):
            prepare_extracted_file_context([asset], plan, max_file_bytes=limit)


def test_prepare_pdf_table_context_marks_text_only_warning(tmp_path, monkeypatch):
    from crupier import multimodal

    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"pdf")
    asset = normalize_file(pdf)
    plan = plan_file_representations([asset], task="extract invoice table")
    monkeypatch.setattr(
        multimodal,
        "_extract_pdf_text",
        lambda asset, max_file_bytes, max_chars, max_pages: ("row 1", False),
    )

    context = prepare_extracted_file_context([asset], plan)

    assert context["files"][0]["representation"] == "table_rows_and_text"
    assert "pdf_table_extraction_not_implemented_text_only" in context["warnings"]


def test_extract_pdf_text_uses_pypdf_when_available(tmp_path, monkeypatch):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")

    class FakeReader:
        def __init__(self, path):
            assert path == str(pdf)
            self.pages = [SimpleNamespace(extract_text=lambda: "first"), SimpleNamespace(extract_text=lambda: None)]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakeReader))

    assert _extract_pdf_text(
        normalize_file(pdf),
        max_file_bytes=100,
        max_chars=100,
        max_pages=10,
    ) == ("first\n\n", False)


def test_extract_pdf_text_falls_back_to_pdftotext(tmp_path, monkeypatch):
    from crupier import multimodal

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setattr(multimodal.shutil, "which", lambda name: "/usr/bin/pdftotext")

    def fake_run(command, **kwargs):
        assert command[:2] == ["/usr/bin/pdftotext", "-f"]
        assert "-layout" in command
        assert kwargs["timeout"] == 30
        output = command[-1]
        with open(output, "w", encoding="utf-8") as handle:
            handle.write("fallback text")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(multimodal.subprocess, "run", fake_run)

    assert _extract_pdf_text(
        normalize_file(pdf),
        max_file_bytes=100,
        max_chars=100,
        max_pages=10,
    ) == ("fallback text", False)


def test_pdftotext_command_uses_absolute_path_and_option_terminator(tmp_path, monkeypatch):
    from crupier import multimodal

    monkeypatch.chdir(tmp_path)
    pdf = Path("-f.pdf")
    pdf.write_bytes(b"pdf")
    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setattr(multimodal.shutil, "which", lambda name: "/usr/bin/pdftotext")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        with open(command[-1], "w", encoding="utf-8") as handle:
            handle.write("fallback text")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(multimodal.subprocess, "run", fake_run)

    _extract_pdf_text(
        normalize_file(pdf),
        max_file_bytes=100,
        max_chars=100,
        max_pages=10,
    )

    separator = observed["command"].index("--")
    assert observed["command"][separator + 1] == str(pdf.resolve())
    assert Path(observed["command"][separator + 1]).is_absolute()


def test_extract_pdf_text_reports_dependency_process_and_size_errors(tmp_path, monkeypatch):
    from crupier import multimodal

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"1234")
    asset = normalize_file(pdf)

    with pytest.raises(CrupierModelUnsupportedError, match="above max 3 bytes"):
        _extract_pdf_text(asset, max_file_bytes=3, max_chars=100, max_pages=10)

    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setattr(multimodal.shutil, "which", lambda name: None)
    with pytest.raises(CrupierModelUnsupportedError, match="requires `pypdf`"):
        _extract_pdf_text(asset, max_file_bytes=100, max_chars=100, max_pages=10)

    monkeypatch.setattr(multimodal.shutil, "which", lambda name: "/usr/bin/pdftotext")
    monkeypatch.setattr(
        multimodal.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="invalid PDF"),
    )
    with pytest.raises(CrupierModelUnsupportedError, match="failed: invalid PDF"):
        _extract_pdf_text(asset, max_file_bytes=100, max_chars=100, max_pages=10)

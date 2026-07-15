"""File input normalization and multimodal representation planning."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes, urlparse

from .errors import CrupierModelUnsupportedError
from .models import FileAsset, FileRepresentation, FileRoutingPlan


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SPREADSHEET_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".ods", ".parquet"}
DOCUMENT_EXTENSIONS = {".doc", ".docx", ".odt", ".rtf"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".html", ".css"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".sql",
    ".sh",
    ".zsh",
}

MODEL_CAPABILITIES = {"vision_input", "audio_input", "video_input", "file_input", "pdf_native_input"}


def normalize_files(files: list[Any] | tuple[Any, ...] | None) -> list[FileAsset]:
    """Normalize supported file references without reading file contents."""

    return [normalize_file(item) for item in (files or [])]


def normalize_file(value: Any) -> FileAsset:
    if isinstance(value, FileAsset):
        return value
    if isinstance(value, dict):
        return _asset_from_dict(value)
    if isinstance(value, (str, Path)):
        return _asset_from_reference(str(value))
    if isinstance(value, bytes):
        return FileAsset(kind="binary", name="bytes", size_bytes=len(value), exists=True)
    raise TypeError(f"Unsupported file reference {type(value).__name__}; pass a path, URL, dict, bytes, or FileAsset.")


def plan_file_representations(
    files: list[FileAsset],
    *,
    task: str = "",
    constraints: dict[str, Any] | None = None,
) -> FileRoutingPlan | None:
    """Build a compact representation plan for file inputs."""

    if not files:
        return None
    constraints = constraints or {}
    representations = [_plan_asset(asset, task=task, constraints=constraints) for asset in files]
    required_modalities = _sorted_unique(
        modality for item in representations for modality in item.required_model_modalities
    )
    required_capabilities = _sorted_unique(
        capability
        for item in representations
        for capability in item.required_model_capabilities
        if capability in MODEL_CAPABILITIES
    )
    warnings = _sorted_unique(warning for item in representations for warning in item.warnings)
    extraction_required = any(item.pipeline for item in representations)
    return FileRoutingPlan(
        assets=files,
        representations=representations,
        required_model_modalities=required_modalities,
        required_model_capabilities=required_capabilities,
        extraction_required=extraction_required,
        warnings=warnings,
    )


def can_execute_native_images(file_plan: FileRoutingPlan | None) -> bool:
    """Return true when every file can be sent as native image input now."""

    if file_plan is None or not file_plan.assets:
        return False
    if file_plan.extraction_required:
        return False
    return all(item.representation == "native_vision" for item in file_plan.representations)


def split_file_execution_inputs(
    file_plan: FileRoutingPlan,
) -> tuple[list[FileAsset], FileRoutingPlan | None]:
    """Separate provider-native assets from assets that need local extraction."""

    native_images: list[FileAsset] = []
    extraction_assets: list[FileAsset] = []
    extraction_representations: list[FileRepresentation] = []
    for asset, representation in zip(file_plan.assets, file_plan.representations, strict=True):
        if representation.representation.startswith("native_") and not representation.pipeline:
            native_images.append(asset)
            continue
        extraction_assets.append(asset)
        extraction_representations.append(representation)
    if not extraction_assets:
        return native_images, None
    extraction_plan = FileRoutingPlan(
        assets=extraction_assets,
        representations=extraction_representations,
        required_model_modalities=_sorted_unique(
            modality for item in extraction_representations for modality in item.required_model_modalities
        ),
        required_model_capabilities=_sorted_unique(
            capability
            for item in extraction_representations
            for capability in item.required_model_capabilities
        ),
        extraction_required=any(item.pipeline for item in extraction_representations),
        warnings=_sorted_unique(warning for item in extraction_representations for warning in item.warnings),
    )
    return native_images, extraction_plan


def native_image_payloads(files: list[FileAsset], *, max_bytes: int = 20_000_000) -> list[dict[str, str]]:
    """Read local image files for provider-native multimodal requests."""

    return native_file_payloads(files, allowed_kinds={"image"}, max_bytes=max_bytes)


def native_file_payloads(
    files: list[FileAsset],
    *,
    allowed_kinds: set[str],
    max_bytes: int = 20_000_000,
) -> list[dict[str, str]]:
    """Read bounded local/data-URL assets for a provider-native request."""

    payloads: list[dict[str, str]] = []
    for asset in files:
        if asset.kind not in allowed_kinds:
            expected = ", ".join(sorted(allowed_kinds))
            raise CrupierModelUnsupportedError(
                f"File {asset.name or '<unnamed>'!r} has kind {asset.kind!r}; expected one of: {expected}."
            )
        if asset.uri and asset.uri.startswith("data:"):
            payloads.append(_data_file_payload(asset, allowed_kinds=allowed_kinds, max_bytes=max_bytes))
            continue
        if not asset.uri or not _looks_local(asset.uri):
            raise CrupierModelUnsupportedError(
                f"File {asset.name or '<unnamed>'!r} must be a local file path or data URL for real execution."
            )
        path = Path(asset.uri).expanduser()
        if not path.exists() or not path.is_file():
            raise CrupierModelUnsupportedError(f"File {asset.name or str(path)!r} does not exist.")
        size = path.stat().st_size
        if size > max_bytes:
            raise CrupierModelUnsupportedError(
                f"File {asset.name or str(path)!r} is {size} bytes, above max {max_bytes} bytes."
            )
        mime_type = asset.mime_type or _guess_mime(asset.name or str(path)) or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payloads.append(
            {
                "name": asset.name or path.name,
                "kind": asset.kind,
                "mime_type": mime_type,
                "base64": encoded,
                "data_url": f"data:{mime_type};base64,{encoded}",
            }
        )
    return payloads


def _data_file_payload(asset: FileAsset, *, allowed_kinds: set[str], max_bytes: int) -> dict[str, str]:
    assert asset.uri is not None
    header, separator, payload = asset.uri.partition(",")
    if not separator or not header.startswith("data:"):
        raise CrupierModelUnsupportedError(f"Image {asset.name or '<unnamed>'!r} has an invalid data URL.")
    media_parts = header[5:].split(";")
    mime_type = media_parts[0] or asset.mime_type or "application/octet-stream"
    inferred_kind = _kind_for(mime_type, "")
    if inferred_kind not in allowed_kinds:
        expected = ", ".join(sorted(allowed_kinds))
        raise CrupierModelUnsupportedError(
            f"Data URL uses MIME type {mime_type!r} ({inferred_kind}); expected one of: {expected}."
        )
    try:
        raw = base64.b64decode(payload, validate=True) if "base64" in media_parts[1:] else unquote_to_bytes(payload)
    except (ValueError, binascii.Error) as exc:
        raise CrupierModelUnsupportedError(
            f"File {asset.name or '<unnamed>'!r} has invalid data URL encoding."
        ) from exc
    if len(raw) > max_bytes:
        raise CrupierModelUnsupportedError(
            f"File data URL is {len(raw)} bytes, above max {max_bytes} bytes."
        )
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "name": asset.name or inferred_kind,
        "kind": inferred_kind,
        "mime_type": mime_type,
        "base64": encoded,
        "data_url": f"data:{mime_type};base64,{encoded}",
    }


def prepare_extracted_file_context(
    files: list[FileAsset],
    file_plan: FileRoutingPlan | None,
    *,
    max_file_bytes: int = 2_000_000,
    max_chars: int = 80_000,
) -> dict[str, Any]:
    """Read supported local files into a bounded text context for real execution."""

    if file_plan is None:
        raise CrupierModelUnsupportedError("No file routing plan is available for real file extraction.")
    sections: list[str] = []
    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    representations = {item.asset_name: item for item in file_plan.representations}
    remaining_chars = max_chars

    for index, asset in enumerate(files, start=1):
        representation = (
            file_plan.representations[index - 1]
            if index - 1 < len(file_plan.representations)
            else representations.get(asset.name)
        )
        if representation is None:
            raise CrupierModelUnsupportedError(f"No file representation is available for {asset.name or index!r}.")
        extracted, item_warnings = _extract_asset_text(asset, representation.representation, max_file_bytes=max_file_bytes)
        warnings.extend(item_warnings)
        if not extracted.strip():
            warnings.append(f"{asset.name or index}: extracted text was empty")
        truncated = extracted[:remaining_chars]
        if len(extracted) > len(truncated):
            warnings.append(f"{asset.name or index}: truncated to fit max_file_context_chars")
        remaining_chars -= len(truncated)
        sections.append(
            "\n".join(
                [
                    f"### File {index}: {asset.name or '<unnamed>'}",
                    f"kind: {asset.kind}",
                    f"representation: {representation.representation}",
                    "content:",
                    truncated,
                ]
            )
        )
        items.append(
            {
                "name": asset.name,
                "kind": asset.kind,
                "representation": representation.representation,
                "chars": len(truncated),
                "truncated": len(extracted) > len(truncated),
            }
        )
        if remaining_chars <= 0:
            if index < len(files):
                warnings.append("file context budget exhausted before all files were included")
            break

    return {
        "body": "\n\n".join(sections),
        "files": items,
        "warnings": warnings,
        "max_chars": max_chars,
    }


def _asset_from_dict(data: dict[str, Any]) -> FileAsset:
    uri = data.get("uri") or data.get("path") or data.get("url")
    name = data.get("name") or _name_from_uri(uri)
    mime_type = data.get("mime_type") or _data_uri_mime(uri) or _guess_mime(name or uri)
    kind = data.get("kind") or _kind_for(mime_type, _suffix(name or uri))
    size_bytes = data.get("size_bytes")
    exists = data.get("exists")
    if size_bytes is None and exists is None and uri and _looks_local(uri):
        path = Path(uri).expanduser()
        exists = path.exists()
        if exists and path.is_file():
            size_bytes = path.stat().st_size
    return FileAsset(
        kind=str(kind),
        name=name,
        uri=str(uri) if uri is not None else None,
        mime_type=mime_type,
        size_bytes=size_bytes,
        page_count=data.get("page_count"),
        duration_seconds=data.get("duration_seconds"),
        exists=exists,
        metadata=dict(data.get("metadata", {})),
    )


def _asset_from_reference(reference: str) -> FileAsset:
    data_mime = _data_uri_mime(reference)
    if data_mime:
        return FileAsset(
            kind=_kind_for(data_mime, ""),
            name="image" if data_mime.startswith("image/") else "data",
            uri=reference,
            mime_type=data_mime,
            exists=True,
        )
    name = _name_from_uri(reference)
    mime_type = _guess_mime(name or reference)
    path = Path(reference).expanduser() if _looks_local(reference) else None
    exists = path.exists() if path else None
    size_bytes = path.stat().st_size if path and exists and path.is_file() else None
    return FileAsset(
        kind=_kind_for(mime_type, _suffix(name or reference)),
        name=name,
        uri=reference,
        mime_type=mime_type,
        size_bytes=size_bytes,
        exists=exists,
    )


def _plan_asset(asset: FileAsset, *, task: str, constraints: dict[str, Any]) -> FileRepresentation:
    strategy = str(constraints.get("file_strategy", "auto"))
    require_native = bool(constraints.get("require_native_file_input", False)) or strategy == "native"
    force_extract = strategy in {"extract", "extracted", "text"}
    task_text = task.lower()

    if asset.kind == "image":
        if force_extract:
            return _representation(
                asset,
                "ocr_text",
                pipeline=["ocr"],
                reason="Image was forced through text extraction/OCR before model routing.",
                warnings=["ocr_extraction_not_implemented"],
            )
        return _representation(
            asset,
            "native_vision",
            modalities=["image"],
            model_capabilities=["vision_input"],
            reason="Image input needs a model with verified or declared vision input.",
        )

    if asset.kind == "pdf":
        if require_native:
            return _representation(
                asset,
                "native_pdf",
                modalities=["file"],
                model_capabilities=["file_input", "pdf_native_input"],
                reason="PDF native input was requested by constraints.",
            )
        pipeline = ["pdf_text_extraction", "chunking"]
        representation = "extracted_text_chunks"
        if any(word in task_text for word in ["table", "tabla", "spreadsheet", "invoice", "factura"]):
            pipeline.insert(1, "table_extraction")
            representation = "table_rows_and_text"
        return _representation(
            asset,
            representation,
            pipeline=pipeline,
            reason="PDF is planned as extracted text/tables so text models can compete on cost and latency.",
            warnings=["pdf_table_extraction_not_implemented"] if "table_extraction" in pipeline else [],
        )

    if asset.kind == "audio":
        if not force_extract:
            return _representation(
                asset,
                "native_audio",
                modalities=["audio"],
                model_capabilities=["audio_input"],
                reason=(
                    "Audio native input was requested by constraints."
                    if require_native
                    else "Audio defaults to native input while transcript preprocessing is unavailable."
                ),
            )
        return _representation(
            asset,
            "transcript",
            pipeline=["audio_transcription"],
            reason="Audio transcript preprocessing was explicitly requested.",
            warnings=["audio_transcription_not_implemented"],
        )

    if asset.kind == "video":
        if require_native:
            return _representation(
                asset,
                "native_video",
                modalities=["video"],
                model_capabilities=["video_input"],
                reason="Video native input was requested by constraints.",
            )
        return _representation(
            asset,
            "transcript_and_frames",
            modalities=["image"],
            model_capabilities=["vision_input"],
            pipeline=["audio_transcription", "frame_sampling"],
            reason="Video is planned as transcript plus sampled frames for quality/cost control.",
            warnings=["video_preprocessing_not_implemented"],
        )

    if asset.kind == "spreadsheet":
        return _representation(
            asset,
            "table_rows",
            pipeline=["table_extraction"],
            reason="Spreadsheet input should be parsed structurally before LLM routing.",
            warnings=["spreadsheet_parsing_not_implemented"],
        )

    if asset.kind == "document":
        return _representation(
            asset,
            "extracted_text",
            pipeline=["document_text_extraction"],
            reason="Document input should be converted to text and tables before model routing.",
            warnings=["document_extraction_not_implemented"],
        )

    if asset.kind == "code":
        return _representation(
            asset,
            "code_chunks",
            pipeline=["code_chunking"],
            reason="Code files should be chunked with path/language metadata before LLM routing.",
        )

    if asset.kind == "text":
        return _representation(asset, "inline_text", reason="Text-like file can be treated as text context.")

    return _representation(
        asset,
        "metadata_only",
        reason="Unknown file type; Crupier can route metadata but cannot choose a richer representation yet.",
        warnings=["unknown_file_kind"],
    )


def _representation(
    asset: FileAsset,
    representation: str,
    *,
    modalities: list[str] | None = None,
    model_capabilities: list[str] | None = None,
    pipeline: list[str] | None = None,
    reason: str,
    warnings: list[str] | None = None,
) -> FileRepresentation:
    asset_warnings = list(warnings or [])
    if asset.exists is False:
        asset_warnings.append("file_path_not_found")
    return FileRepresentation(
        asset_name=asset.name,
        kind=asset.kind,
        representation=representation,
        required_model_modalities=_sorted_unique(modalities or ["text"]),
        required_model_capabilities=_sorted_unique(model_capabilities or []),
        pipeline=list(pipeline or []),
        reason=reason,
        warnings=_sorted_unique(asset_warnings),
    )


def _guess_mime(name: str | None) -> str | None:
    if not name:
        return None
    return mimetypes.guess_type(name)[0]


def _kind_for(mime_type: str | None, suffix: str) -> str:
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type == "application/pdf":
            return "pdf"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("text/"):
            return "text"
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def _name_from_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    if "://" in uri:
        path = urlparse(uri).path
        return Path(path).name or None
    return Path(uri).name or None


def _data_uri_mime(uri: Any) -> str | None:
    if not isinstance(uri, str) or not uri.startswith("data:"):
        return None
    header = uri.partition(",")[0]
    mime_type = header[5:].split(";", 1)[0]
    return mime_type or None


def _suffix(name: str | None) -> str:
    return Path(name or "").suffix.lower()


def _looks_local(reference: str) -> bool:
    if len(reference) >= 2 and reference[0].isalpha() and reference[1] == ":":
        return True
    return not urlparse(reference).scheme


def _sorted_unique(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value})


def _extract_asset_text(asset: FileAsset, representation: str, *, max_file_bytes: int) -> tuple[str, list[str]]:
    if asset.kind in {"text", "code"} or representation in {"inline_text", "code_chunks"}:
        return _read_local_text(asset, max_file_bytes=max_file_bytes), []
    if asset.kind == "pdf" and representation in {"extracted_text_chunks", "table_rows_and_text"}:
        text = _extract_pdf_text(asset, max_file_bytes=max_file_bytes)
        warnings = ["pdf_table_extraction_not_implemented_text_only"] if representation == "table_rows_and_text" else []
        return text, warnings
    raise CrupierModelUnsupportedError(
        f"Real extraction for {asset.kind!r} as {representation!r} is not supported by the local extractor."
    )


def _local_path(asset: FileAsset) -> Path:
    if not asset.uri or not _looks_local(asset.uri):
        raise CrupierModelUnsupportedError(f"File {asset.name or '<unnamed>'!r} must be a local path for extraction.")
    path = Path(asset.uri).expanduser()
    if not path.exists() or not path.is_file():
        raise CrupierModelUnsupportedError(f"File {asset.name or str(path)!r} does not exist.")
    return path


def _read_local_text(asset: FileAsset, *, max_file_bytes: int) -> str:
    path = _local_path(asset)
    size = path.stat().st_size
    if size > max_file_bytes:
        raise CrupierModelUnsupportedError(
            f"File {asset.name or str(path)!r} is {size} bytes, above max {max_file_bytes} bytes."
        )
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf_text(asset: FileAsset, *, max_file_bytes: int) -> str:
    path = _local_path(asset)
    size = path.stat().st_size
    if size > max_file_bytes:
        raise CrupierModelUnsupportedError(
            f"PDF {asset.name or str(path)!r} is {size} bytes, above max {max_file_bytes} bytes."
        )
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        reader = None
    else:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        raise CrupierModelUnsupportedError(
            "PDF text extraction requires `pypdf` (`pip install crupier[pdf]`) or a `pdftotext` binary."
        )
    with tempfile.TemporaryDirectory(prefix="crupier-pdf-") as tmp_dir:
        out_path = Path(tmp_dir) / "out.txt"
        completed = subprocess.run(
            [pdftotext, "-layout", str(path), str(out_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise CrupierModelUnsupportedError(f"PDF text extraction failed: {completed.stderr.strip()}")
        return out_path.read_text(encoding="utf-8", errors="replace")

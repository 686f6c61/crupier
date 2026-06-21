import json

from crupier import Crupier, FileAsset
from crupier.adapters import AdapterResponse
from crupier.config import CrupierConfig
from crupier.errors import CrupierModelUnsupportedError
from crupier.multimodal import normalize_file, plan_file_representations


def make_config(tmp_path, *, allow, strategy="single"):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "test", "default_profile": "agentic"},
            "providers": {
                "openai": {"enabled": True, "env_key": "OPENAI_API_KEY"},
                "ollama": {"enabled": True, "host": "http://localhost:11434"},
            },
            "models": {"allow": allow},
            "routing": {"default_strategy": strategy, "allow_fusion": True, "allow_parallel": True},
            "profiles": {"agentic": {"prefer": ["tool_use"], "strategy": strategy}},
        }
    )
    config.root = tmp_path
    return config


class FakeVisionAdapter:
    provider = "openai"

    def __init__(self):
        self.calls = []

    def generate(self, *, model, prompt, request):
        self.calls.append({"model": model, "prompt": prompt, "files": [file.to_dict() for file in request.files]})
        return AdapterResponse(text="image says total 12.50", metadata={"provider": "openai", "model": model})


def test_normalize_file_infers_image_kind_without_reading_contents(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"not really an image")

    asset = normalize_file(image)

    assert asset.kind == "image"
    assert asset.name == "receipt.png"
    assert asset.mime_type == "image/png"
    assert asset.size_bytes == len(b"not really an image")
    assert "uri" not in asset.to_dict()
    assert asset.to_dict(include_uri=True)["uri"].endswith("receipt.png")


def test_image_file_filters_to_vision_capable_model(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["ollama:qwen3.5:122b", "openai:gpt-5.4-mini"]))

    result = client.deal(
        "Extract the total from this receipt image",
        files=[tmp_path / "receipt.png"],
        trace="summary",
    )

    assert result.route is not None
    assert result.route.models == ["openai:gpt-5.4-mini"]
    file_plan = result.route.input_plan["files"]
    assert file_plan["representations"][0]["representation"] == "native_vision"
    assert file_plan["required_model_modalities"] == ["image"]
    assert result.trace is not None
    assert any(item["model"] == "ollama:qwen3.5:122b" for item in result.trace.excluded_models)


def test_pdf_defaults_to_extracted_text_route_and_keeps_text_models(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["ollama:qwen3.5:122b"]))

    result = client.deal(
        "Extract clauses from this PDF",
        files=[{"name": "contract.pdf", "mime_type": "application/pdf"}],
        trace="summary",
    )

    assert result.route is not None
    assert result.route.models == ["ollama:qwen3.5:122b"]
    file_plan = result.route.input_plan["files"]
    assert file_plan["representations"][0]["representation"] == "extracted_text_chunks"
    assert file_plan["required_model_modalities"] == ["text"]
    assert file_plan["extraction_required"] is True


def test_native_pdf_constraint_requires_file_input(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["ollama:qwen3.5:122b", "openai:gpt-5.4-mini"]))

    result = client.deal(
        "Read this PDF natively",
        files=[FileAsset(kind="pdf", name="contract.pdf")],
        constraints={"require_native_file_input": True},
        trace="summary",
    )

    assert result.route is not None
    assert result.route.models == ["openai:gpt-5.4-mini"]
    file_plan = result.route.input_plan["files"]
    assert file_plan["representations"][0]["representation"] == "native_pdf"
    assert set(file_plan["required_model_capabilities"]) == {"file_input", "pdf_native_input"}


def test_real_execution_allows_native_image_inputs(tmp_path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake image bytes")
    adapter = FakeVisionAdapter()
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]), adapters={"openai": adapter})

    result = client.deal("Read image", files=[image], dry_run=False)

    assert result.output_text == "image says total 12.50"
    assert adapter.calls[0]["files"][0]["kind"] == "image"


def test_real_execution_extracts_text_file_context(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("Project codename: ALPHA-17", encoding="utf-8")
    adapter = FakeVisionAdapter()
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]), adapters={"openai": adapter})

    result = client.deal("What is the project codename?", files=[notes], dry_run=False, trace="summary")

    assert result.output_text == "image says total 12.50"
    assert adapter.calls[0]["files"] == []
    assert "File context:" in adapter.calls[0]["prompt"]
    assert "ALPHA-17" in adapter.calls[0]["prompt"]
    assert result.trace is not None
    assert result.trace.final_quality_signals["file_context"]["files"][0]["name"] == "notes.txt"


def test_real_execution_extracts_pdf_text_context(tmp_path, monkeypatch):
    import crupier.multimodal as multimodal

    pdf = tmp_path / "contract.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake enough for the monkeypatched extractor")
    monkeypatch.setattr(multimodal, "_extract_pdf_text", lambda asset, max_file_bytes: "PDF passphrase: BERYL-42")
    adapter = FakeVisionAdapter()
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]), adapters={"openai": adapter})

    result = client.deal("What is the PDF passphrase?", files=[pdf], dry_run=False, trace="summary")

    assert result.output_text == "image says total 12.50"
    assert adapter.calls[0]["files"] == []
    assert "PDF passphrase: BERYL-42" in adapter.calls[0]["prompt"]
    assert result.trace is not None
    assert result.trace.final_quality_signals["file_context"]["files"][0]["representation"] == "extracted_text_chunks"


def test_markdown_file_is_text_context(tmp_path):
    markdown = tmp_path / "README.md"
    markdown.write_text("# Hello", encoding="utf-8")

    asset = normalize_file(markdown)

    assert asset.kind == "text"


def test_real_execution_still_blocks_native_pdf_without_mapping(tmp_path):
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]))

    try:
        client.deal(
            "Read PDF",
            files=[{"kind": "pdf", "name": "contract.pdf"}],
            constraints={"require_native_file_input": True},
            dry_run=False,
        )
    except CrupierModelUnsupportedError as exc:
        assert "native_pdf" in str(exc)
    else:
        raise AssertionError("native PDF execution should be blocked until provider mappings exist")


def test_real_execution_blocks_unimplemented_audio_transcription(tmp_path):
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"fake audio")
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]))

    try:
        client.deal("Summarize the call", files=[audio], dry_run=False)
    except CrupierModelUnsupportedError as exc:
        assert "audio" in str(exc)
        assert "transcript" in str(exc)
    else:
        raise AssertionError("audio transcription should be explicit unsupported until implemented")


def test_real_execution_blocks_unimplemented_image_ocr(tmp_path):
    image = tmp_path / "scan.png"
    image.write_bytes(b"fake image")
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]))

    try:
        client.deal("OCR this scan", files=[image], constraints={"file_strategy": "extract"}, dry_run=False)
    except CrupierModelUnsupportedError as exc:
        assert "image" in str(exc)
        assert "ocr_text" in str(exc)
    else:
        raise AssertionError("image OCR should be explicit unsupported until implemented")


def test_real_execution_blocks_unimplemented_spreadsheet_parsing(tmp_path):
    sheet = tmp_path / "ledger.xlsx"
    sheet.write_bytes(b"fake spreadsheet")
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]))

    try:
        client.deal("Summarize this ledger", files=[sheet], dry_run=False)
    except CrupierModelUnsupportedError as exc:
        assert "spreadsheet" in str(exc)
        assert "table_rows" in str(exc)
    else:
        raise AssertionError("spreadsheet parsing should be explicit unsupported until implemented")


def test_real_execution_blocks_unimplemented_document_extraction(tmp_path):
    document = tmp_path / "brief.docx"
    document.write_bytes(b"fake docx")
    client = Crupier(make_config(tmp_path, allow=["openai:gpt-5.4-mini"]))

    try:
        client.deal("Summarize this brief", files=[document], dry_run=False)
    except CrupierModelUnsupportedError as exc:
        assert "document" in str(exc)
        assert "extracted_text" in str(exc)
    else:
        raise AssertionError("document extraction should be explicit unsupported until implemented")


def test_file_routing_plan_serializes_without_uri_by_default():
    plan = plan_file_representations([FileAsset(kind="image", name="secret.png", uri="/tmp/secret.png")], task="Read")

    assert plan is not None
    data = plan.to_dict()
    assert "uri" not in json.dumps(data)
    assert data["representations"][0]["representation"] == "native_vision"

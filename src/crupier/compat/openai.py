"""OpenAI-like compatibility client backed by Crupier."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from crupier.client import Crupier
from crupier.config import CrupierConfig
from crupier.errors import CrupierProviderUnavailableError
from crupier.models import CrupierResult, OperationResult


class CompatObject(dict):
    """Small dict/attribute hybrid for SDK-like responses."""

    def __init__(self, **items: Any):
        super().__init__(items)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.to_dict()


class CompatBinaryResponse:
    """Small OpenAI-like binary response used for speech output."""

    def __init__(self, content: bytes, *, crupier: CompatObject):
        self.content = content
        self.crupier = crupier

    def read(self) -> bytes:
        return self.content

    def iter_bytes(self, *, chunk_size: int = 65_536) -> Iterator[bytes]:
        for start in range(0, len(self.content), max(1, chunk_size)):
            yield self.content[start : start + max(1, chunk_size)]

    def stream_to_file(self, path: str | Path) -> None:
        Path(path).write_bytes(self.content)


class OpenAI:
    """OpenAI-like client that routes requests through Crupier.

    This is not a full clone of the OpenAI SDK. It implements the first useful
    drop-in surface: ``responses.create`` and ``chat.completions.create``.
    """

    def __init__(
        self,
        *,
        crupier: Crupier | None = None,
        config: CrupierConfig | dict[str, Any] | None = None,
        project: str = ".",
        dry_run: bool | None = None,
        compat_mode: str = "balanced",
        **_: Any,
    ):
        if crupier is not None:
            self._crupier = crupier
        elif config is not None:
            self._crupier = Crupier.from_config(config)
        else:
            self._crupier = Crupier.from_project(project)
        self._dry_run = dry_run
        self._compat_mode = compat_mode
        self.responses = _Responses(self)
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
        self.images = _Images(self)
        self.audio = _Audio(self)
        self.rerank = _Rerank(self)

    def _deal(
        self,
        *,
        task: str,
        input: Any = None,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
        mode: str | None = None,
        tools: list[Any] | None = None,
        response_format: Any = None,
        stream: bool = False,
        dry_run: bool | None = None,
        trace: bool | str = False,
        **kwargs: Any,
    ) -> CrupierResult:
        constraints = _compat_constraints(
            model=model,
            stream=stream,
            compat_mode=str(kwargs.pop("compat_mode", self._compat_mode)),
            kwargs=kwargs,
        )
        files, normalized_input, normalized_messages = _extract_file_inputs(input=input, messages=messages)
        response_schema = _response_schema_from_format(response_format)
        if dry_run is None:
            dry_run = self._dry_run
        return self._crupier.deal(
            task=task,
            input=normalized_input,
            messages=normalized_messages,
            files=files,
            tools=tools,
            response_schema=response_schema,
            mode=mode,
            constraints=constraints,
            dry_run=dry_run,
            trace=trace,
        )

    def _operation_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        values = dict(kwargs)
        constraints = dict(values.pop("constraints", {}) or {})
        for key in ("max_calls", "max_cost_usd", "max_latency_ms", "timeout_seconds"):
            if values.get(key) is not None:
                constraints[key] = values.pop(key)
        for ignored in (
            "background",
            "moderation",
            "output_compression",
            "output_format",
            "quality",
            "style",
            "user",
        ):
            values.pop(ignored, None)
        dry_run = values.pop("dry_run", self._dry_run)
        if dry_run is not None:
            values["dry_run"] = bool(dry_run)
        if constraints:
            values["constraints"] = constraints
        return values


class _Responses:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(
        self,
        *,
        input: Any = None,
        model: str | None = None,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        response_format: Any = None,
        stream: bool = False,
        mode: str | None = None,
        dry_run: bool | None = None,
        trace: bool | str = False,
        **kwargs: Any,
    ) -> CompatObject | Iterator[CompatObject]:
        include_obfuscation = kwargs.get("include_obfuscation", True)
        result = self._owner._deal(
            task=instructions or "Respond to the provided input.",
            input=input,
            model=model,
            mode=mode,
            tools=tools,
            response_format=response_format,
            stream=stream,
            dry_run=dry_run,
            trace=trace,
            **kwargs,
        )
        response = _responses_object(result, requested_model=model)
        if stream:
            return _responses_stream(response, include_obfuscation=include_obfuscation)
        return response


class _Chat:
    def __init__(self, owner: OpenAI):
        self.completions = _ChatCompletions(owner)


class _ChatCompletions:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[Any] | None = None,
        functions: list[Any] | None = None,
        response_format: Any = None,
        stream: bool = False,
        mode: str | None = None,
        dry_run: bool | None = None,
        trace: bool | str = False,
        **kwargs: Any,
    ) -> CompatObject | Iterator[CompatObject]:
        stream_options = kwargs.get("stream_options")
        include_usage = bool(stream_options.get("include_usage")) if isinstance(stream_options, dict) else False
        task = _task_from_messages(messages)
        result = self._owner._deal(
            task=task,
            messages=messages,
            model=model,
            mode=mode,
            tools=tools or functions,
            response_format=response_format,
            stream=stream,
            dry_run=dry_run,
            trace=trace,
            **kwargs,
        )
        response = _chat_completion_object(result, requested_model=model)
        if stream:
            return _chat_completion_stream(response, include_usage=include_usage)
        return response


class _Embeddings:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(
        self,
        *,
        input: Any,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
        **kwargs: Any,
    ) -> CompatObject:
        del encoding_format, user
        requested_dimensions = None if dimensions is None else int(dimensions)
        result = self._owner._crupier.embed(
            input=input,
            model=model,
            dimensions=requested_dimensions,
            **self._owner._operation_kwargs(kwargs),
        )
        embeddings = result.data
        if not isinstance(embeddings, list):
            raise CrupierProviderUnavailableError("Embedding provider returned an invalid vector list.", retryable=False)
        if not embeddings:
            raise CrupierProviderUnavailableError("Embedding provider returned no vectors.", retryable=False)
        if requested_dimensions is not None and any(
            not isinstance(embedding, list) or len(embedding) != requested_dimensions for embedding in embeddings
        ):
            raise CrupierProviderUnavailableError(
                f"Embedding provider did not honor dimensions={requested_dimensions}.",
                retryable=False,
            )
        data = [
            CompatObject(object="embedding", index=index, embedding=embedding)
            for index, embedding in enumerate(embeddings)
        ]
        return CompatObject(
            object="list",
            data=data,
            model=result.model,
            usage=_embedding_usage_object(result.usage),
            crupier=_operation_metadata(result),
        )


class _Images:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs: Any,
    ) -> CompatObject:
        result = self._owner._crupier.generate_image(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            response_format=response_format,
            **self._owner._operation_kwargs(kwargs),
        )
        return _image_object(result)

    def edit(
        self,
        *,
        image: Any,
        prompt: str,
        model: str | None = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs: Any,
    ) -> CompatObject:
        result = self._owner._crupier.edit_image(
            prompt=prompt,
            images=image,
            model=model,
            n=n,
            size=size,
            response_format=response_format,
            **self._owner._operation_kwargs(kwargs),
        )
        return _image_object(result)


class _Audio:
    def __init__(self, owner: OpenAI):
        self.speech = _Speech(owner)
        self.transcriptions = _Transcriptions(owner)


class _Speech:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(
        self,
        *,
        input: str,
        voice: str,
        model: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> CompatBinaryResponse:
        result = self._owner._crupier.synthesize(
            input=input,
            voice=voice,
            model=model,
            response_format=response_format,
            speed=speed,
            **self._owner._operation_kwargs(kwargs),
        )
        if not isinstance(result.data, bytes):
            raise CrupierProviderUnavailableError("Speech provider returned a non-binary response.", retryable=False)
        return CompatBinaryResponse(result.data, crupier=_operation_metadata(result))


class _Transcriptions:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(
        self,
        *,
        file: Any,
        model: str | None = None,
        language: str | None = None,
        response_format: str = "json",
        timestamp_granularities: list[str] | None = None,
        **kwargs: Any,
    ) -> CompatObject:
        result = self._owner._crupier.transcribe(
            file=file,
            model=model,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            **self._owner._operation_kwargs(kwargs),
        )
        data = dict(result.data) if isinstance(result.data, dict) else {"text": str(result.data or "")}
        data.update({"model": result.model, "crupier": _operation_metadata(result)})
        return CompatObject(**data)


class _Rerank:
    def __init__(self, owner: OpenAI):
        self._owner = owner

    def create(
        self,
        *,
        query: str,
        documents: list[str],
        model: str | None = None,
        top_n: int | None = None,
        **kwargs: Any,
    ) -> CompatObject:
        result = self._owner._crupier.rerank(
            query=query,
            documents=documents,
            model=model,
            top_n=top_n,
            **self._owner._operation_kwargs(kwargs),
        )
        return CompatObject(
            id=f"rerank_{uuid4().hex[:16]}",
            model=result.model,
            results=[CompatObject(**item) for item in result.data or []],
            usage=CompatObject(**result.usage),
            crupier=_operation_metadata(result),
        )


def _compat_constraints(
    *,
    model: str | None,
    stream: bool,
    compat_mode: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    constraints: dict[str, Any] = {"compat": "openai", "compat_mode": compat_mode}
    if model:
        constraints["requested_model"] = model
        if compat_mode == "strict":
            constraints["force_model"] = _model_ref_for_openai(model)
    if stream:
        constraints["stream"] = True
    for source, target in [
        ("max_tokens", "max_output_tokens"),
        ("max_completion_tokens", "max_output_tokens"),
        ("max_output_tokens", "max_output_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ]:
        if source in kwargs and kwargs[source] is not None:
            constraints[target] = kwargs[source]
    return constraints


def _response_schema_from_format(response_format: Any) -> Any:
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        if response_format.get("type") in {"json_schema", "json_object"}:
            return response_format
    return response_format


def _extract_file_inputs(
    *,
    input: Any = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[list[Any], Any, list[dict[str, Any]] | None]:
    files: list[Any] = []
    normalized_messages = [_normalize_message(message, files) for message in messages] if messages else None
    normalized_input = _normalize_input(input, files)
    return files, normalized_input, normalized_messages


def _normalize_message(message: dict[str, Any], files: list[Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        message = dict(message)
        message["content"] = [_normalize_content_part(part, files) for part in content]
    return message


def _normalize_input(value: Any, files: list[Any]) -> Any:
    if isinstance(value, list):
        return [_normalize_input(item, files) for item in value]
    if isinstance(value, dict):
        if "content" in value and isinstance(value["content"], list):
            item = dict(value)
            item["content"] = [_normalize_content_part(part, files) for part in item["content"]]
            return item
        return {key: _normalize_input(item, files) for key, item in value.items()}
    return value


def _normalize_content_part(part: Any, files: list[Any]) -> Any:
    if not isinstance(part, dict):
        return part
    part_type = part.get("type")
    if part_type in {"input_image", "image_url"}:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            uri = image_url.get("url")
        else:
            uri = part.get("image_url") or part.get("url")
        files.append({"uri": uri, "kind": "image", "mime_type": part.get("mime_type")})
        return {"type": "text", "text": "[image input planned by Crupier]"}
    if part_type in {"input_file", "file"}:
        files.append(
            {
                "uri": part.get("file_id") or part.get("filename") or part.get("url"),
                "name": part.get("filename") or part.get("name"),
                "mime_type": part.get("mime_type"),
            }
        )
        return {"type": "text", "text": "[file input planned by Crupier]"}
    return part


def _task_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _text_from_content(message.get("content"))
            if text:
                return text if len(text) <= 300 else text[:297] + "..."
    return "Respond to the chat messages."


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("input_text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks)
    return ""


def _responses_object(result: CrupierResult, *, requested_model: str | None) -> CompatObject:
    model = _response_model(result, requested_model)
    content = CompatObject(type="output_text", text=result.output_text)
    message = CompatObject(
        id=f"msg_{uuid4().hex[:16]}",
        type="message",
        role="assistant",
        content=[content],
    )
    return CompatObject(
        id=f"resp_{uuid4().hex[:16]}",
        object="response",
        created_at=int(time.time()),
        status="completed",
        model=model,
        output_text=result.output_text,
        output=[message],
        usage=_usage_object(result),
        crupier=_crupier_metadata(result),
    )


def _chat_completion_object(result: CrupierResult, *, requested_model: str | None) -> CompatObject:
    model = _response_model(result, requested_model)
    message = CompatObject(role="assistant", content=result.output_text)
    choice = CompatObject(index=0, message=message, finish_reason="stop")
    return CompatObject(
        id=f"chatcmpl_{uuid4().hex[:16]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=_usage_object(result),
        crupier=_crupier_metadata(result),
    )


def _responses_stream(
    response: CompatObject,
    *,
    include_obfuscation: bool | None = None,
) -> Iterator[CompatObject]:
    item_id = response.output[0].id if response.output else f"msg_{uuid4().hex[:16]}"
    created = CompatObject(**response.to_dict())
    created.status = "in_progress"
    created.output_text = ""
    created.output = []
    yield CompatObject(type="response.created", response=created)
    for delta in _text_chunks(response.output_text):
        event = CompatObject(
            type="response.output_text.delta",
            response_id=response.id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            delta=delta,
        )
        if include_obfuscation:
            event.obfuscation = ""
        yield event
    yield CompatObject(
        type="response.output_text.done",
        response_id=response.id,
        item_id=item_id,
        output_index=0,
        content_index=0,
        text=response.output_text,
    )
    yield CompatObject(type="response.completed", response=response)


def _chat_completion_stream(response: CompatObject, *, include_usage: bool = False) -> Iterator[CompatObject]:
    chunk_base = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
    }
    yield CompatObject(
        **chunk_base,
        choices=[CompatObject(index=0, delta=CompatObject(role="assistant"), finish_reason=None, logprobs=None)],
        usage=None,
    )
    for delta_text in _text_chunks(response.choices[0].message.content):
        choice = CompatObject(
            index=0,
            delta=CompatObject(content=delta_text),
            finish_reason=None,
            logprobs=None,
        )
        yield CompatObject(**chunk_base, choices=[choice], usage=None)
    final_choice = CompatObject(index=0, delta=CompatObject(), finish_reason="stop", logprobs=None)
    yield CompatObject(
        **chunk_base,
        choices=[final_choice],
        usage=None,
    )
    if include_usage:
        yield CompatObject(**chunk_base, choices=[], usage=response.usage)


def _text_chunks(text: str, *, max_chars: int = 120) -> Iterator[str]:
    if not text:
        return
    for start in range(0, len(text), max_chars):
        yield text[start : start + max_chars]


def _usage_object(result: CrupierResult) -> CompatObject:
    calls = result.provider_metadata.get("calls", []) if result.provider_metadata else []
    usage: dict[str, Any] = {}
    for call in calls:
        for key, value in (call.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return CompatObject(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )


def _embedding_usage_object(usage: dict[str, Any]) -> CompatObject:
    prompt_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens) or 0)
    return CompatObject(prompt_tokens=prompt_tokens, total_tokens=total_tokens)


def _crupier_metadata(result: CrupierResult) -> CompatObject:
    route = result.route.to_dict() if result.route else None
    trace = result.trace.to_dict(summary=True) if result.trace else None
    return CompatObject(
        route=route,
        trace=trace,
        warnings=list(result.warnings),
        provider_metadata=dict(result.provider_metadata),
    )


def _operation_metadata(result: OperationResult) -> CompatObject:
    return CompatObject(
        operation=result.operation,
        route=result.route.to_dict() if result.route else None,
        trace=result.trace.to_dict(summary=True) if result.trace else None,
        warnings=list(result.warnings),
        provider_metadata=dict(result.provider_metadata),
    )


def _image_object(result: OperationResult) -> CompatObject:
    data = [CompatObject(**item) for item in result.data or [] if isinstance(item, dict)]
    return CompatObject(
        created=int(time.time()),
        model=result.model,
        data=data,
        crupier=_operation_metadata(result),
    )


def _response_model(result: CrupierResult, requested_model: str | None) -> str:
    if result.route and result.route.models:
        return result.route.models[0]
    return requested_model or "crupier"


def _model_ref_for_openai(model: str) -> str:
    return model if ":" in model else f"openai:{model}"


def _looks_like_embedding_model(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in ["embed", "embedding", "all-minilm", "bge-", "e5-", "gte-"])


def _plain(value: Any) -> Any:
    if isinstance(value, CompatObject):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value

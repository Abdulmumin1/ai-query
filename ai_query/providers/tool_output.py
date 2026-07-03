"""Provider wire mappings for explicit rich tool outputs."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ai_query.types import (
    FilePart,
    ImagePart,
    TextPart,
    ToolOutput,
    UnsupportedToolOutputError,
)


FetchResource = Callable[[str], Awaitable[tuple[str, str]]]


def has_tool_output(messages: list[Any]) -> bool:
    for message in messages:
        if isinstance(message.content, str):
            continue
        for part in message.content:
            tool_result = getattr(part, "tool_result", None)
            if tool_result is not None and isinstance(tool_result.result, ToolOutput):
                return True
    return False


def unsupported(provider: str, endpoint: str) -> UnsupportedToolOutputError:
    return UnsupportedToolOutputError(
        f"{provider} {endpoint} cannot represent ToolOutput rich content without "
        "loss; use a provider endpoint and model with multimodal tool-result support"
    )


def google_supports_multimodal_tool_output(model: str) -> bool:
    match = re.match(r"^gemini-(\d+)(?:\D|$)", model)
    return match is not None and int(match.group(1)) >= 3


async def _base64_data(
    value: str | bytes,
    media_type: str | None,
    fetch: FetchResource,
) -> tuple[str, str | None]:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii"), media_type
    if value.startswith(("http://", "https://")):
        data, fetched_type = await fetch(value)
        return data, media_type or fetched_type
    if value.startswith("data:"):
        header, separator, data = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("ToolOutput data URLs must contain base64 data")
        detected_type = header[5:].split(";", 1)[0] or None
        return data, media_type or detected_type
    return value, media_type


async def openai_responses_content(
    output: ToolOutput,
    fetch: FetchResource,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in output.content:
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            if isinstance(part.image, str) and part.image.startswith(
                ("http://", "https://", "data:")
            ):
                image_url = part.image
            else:
                data, media_type = await _base64_data(
                    part.image, part.media_type or "image/png", fetch
                )
                image_url = f"data:{media_type or 'image/png'};base64,{data}"
            content.append(
                {"type": "input_image", "image_url": image_url, "detail": "auto"}
            )
        elif isinstance(part, FilePart):
            if isinstance(part.data, str) and part.data.startswith(
                ("http://", "https://")
            ):
                item: dict[str, Any] = {
                    "type": "input_file",
                    "file_url": part.data,
                }
            else:
                data, _ = await _base64_data(part.data, part.media_type, fetch)
                item = {"type": "input_file", "file_data": data}
            if part.filename:
                item["filename"] = part.filename
            content.append(item)
    return content


async def anthropic_tool_result_content(
    output: ToolOutput,
    fetch: FetchResource,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in output.content:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            data, media_type = await _base64_data(
                part.image, part.media_type or "image/png", fetch
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type or "image/png",
                        "data": data,
                    },
                }
            )
        elif isinstance(part, FilePart):
            data, media_type = await _base64_data(
                part.data, part.media_type or "application/pdf", fetch
            )
            source: dict[str, Any] = {
                "type": "base64",
                "media_type": media_type or "application/pdf",
                "data": data,
            }
            content.append({"type": "document", "source": source})
    return content


async def google_function_response(
    output: ToolOutput,
    fetch: FetchResource,
) -> dict[str, Any]:
    text = "\n".join(
        part.text for part in output.content if isinstance(part, TextPart)
    )
    response: dict[str, Any] = {"result": text} if text else {"result": "success"}
    parts: list[dict[str, Any]] = []
    for index, part in enumerate(output.content):
        if isinstance(part, TextPart):
            continue
        raw = part.image if isinstance(part, ImagePart) else part.data
        default_type = "image/png" if isinstance(part, ImagePart) else "application/pdf"
        data, media_type = await _base64_data(
            raw, part.media_type or default_type, fetch
        )
        display_name = (
            part.filename
            if isinstance(part, FilePart) and part.filename
            else f"tool-output-{index}"
        )
        parts.append(
            {
                "inlineData": {
                    "mimeType": media_type or default_type,
                    "data": data,
                    "displayName": display_name,
                }
            }
        )
    result: dict[str, Any] = {"response": response}
    if parts:
        result["parts"] = parts
    return result


def _bytes_data(value: str | bytes, provider: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if value.startswith(("http://", "https://")):
        raise unsupported(provider, "Converse URL tool results")
    if value.startswith("data:"):
        _, separator, value = value.partition(",")
        if not separator:
            raise ValueError("Invalid ToolOutput data URL")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Bedrock ToolOutput binary strings must be base64") from exc


def bedrock_tool_result_content(output: ToolOutput) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, part in enumerate(output.content):
        if isinstance(part, TextPart):
            content.append({"text": part.text})
        elif isinstance(part, ImagePart):
            media_type = part.media_type or "image/png"
            content.append(
                {
                    "image": {
                        "format": media_type.split("/", 1)[-1],
                        "source": {"bytes": _bytes_data(part.image, "bedrock")},
                    }
                }
            )
        elif isinstance(part, FilePart):
            media_type = part.media_type or "application/pdf"
            content.append(
                {
                    "document": {
                        "format": media_type.split("/", 1)[-1],
                        "name": part.filename or f"tool-output-{index}",
                        "source": {"bytes": _bytes_data(part.data, "bedrock")},
                    }
                }
            )
    return content

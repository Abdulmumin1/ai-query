"""Provider wire mappings for explicit rich tool outputs."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from ai_query.model import LanguageModel
from ai_query.types import (
    ContentPart,
    FilePart,
    ImagePart,
    Message,
    TextPart,
    ToolOutput,
    ToolOutputPart,
    ToolResultPart,
    UnsupportedToolOutputError,
)


FetchResource = Callable[[str], Awaitable[tuple[str, str]]]

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"
NON_FILE_USER_PLACEHOLDER = "(file omitted: model does not support files)"
NON_FILE_TOOL_PLACEHOLDER = "(tool file omitted: model does not support files)"
CHAT_FILE_TOOL_PLACEHOLDER = (
    "(tool file omitted: provider endpoint does not support file tool results)"
)
CHAT_TOOL_IMAGE_PROMPT = "Attached image(s) from tool result:"


def has_tool_output(messages: list[Any]) -> bool:
    for message in messages:
        if isinstance(message.content, str):
            continue
        for part in message.content:
            tool_result = getattr(part, "tool_result", None)
            if tool_result is not None and isinstance(tool_result.result, ToolOutput):
                return True
    return False


def transform_messages_for_model(
    messages: list[Message], model: LanguageModel
) -> list[Message]:
    """Project canonical history into the target model and endpoint capabilities."""
    modalities = model.input_modalities
    if modalities is None:
        return messages

    projected: list[Message] = []
    pending_images: list[ImagePart] = []
    native_tool_output = model.provider.supports_native_tool_output(model.model_id)

    for message in messages:
        if message.role != "tool" and pending_images:
            projected.append(_tool_images_message(pending_images))
            pending_images = []

        message, images = _project_message(
            message,
            modalities=modalities,
            native_tool_output=native_tool_output,
        )
        projected.append(message)
        if message.role == "tool":
            pending_images.extend(images)
        elif images:
            projected.append(_tool_images_message(images))

    if pending_images:
        projected.append(_tool_images_message(pending_images))

    return projected


def _project_message(
    message: Message,
    *,
    modalities: tuple[str, ...],
    native_tool_output: bool,
) -> tuple[Message, list[ImagePart]]:
    if isinstance(message.content, str):
        return message, []

    changed = False
    images: list[ImagePart] = []
    content: list[ContentPart] = []
    for part in message.content:
        replacement: ContentPart = part
        if isinstance(part, ImagePart) and "image" not in modalities:
            replacement = TextPart(text=NON_VISION_USER_IMAGE_PLACEHOLDER)
        elif isinstance(part, FilePart) and "file" not in modalities:
            replacement = TextPart(text=NON_FILE_USER_PLACEHOLDER)
        elif (
            isinstance(part, ToolResultPart)
            and part.tool_result is not None
            and isinstance(part.tool_result.result, ToolOutput)
        ):
            output = part.tool_result.result
            if native_tool_output:
                result = _project_native_tool_output(output, modalities)
            else:
                result, output_images = _project_chat_tool_output(output, modalities)
                images.extend(output_images)
            if result is not output:
                replacement = replace(
                    part,
                    tool_result=replace(part.tool_result, result=result),
                )
        changed = changed or replacement is not part
        content.append(replacement)

    if changed:
        return replace(message, content=content), images
    return message, images


def _project_native_tool_output(
    output: ToolOutput, modalities: tuple[str, ...]
) -> ToolOutput:
    content: list[ToolOutputPart] = []
    changed = False
    for part in output.content:
        replacement: ToolOutputPart = part
        if isinstance(part, ImagePart) and "image" not in modalities:
            replacement = TextPart(text=NON_VISION_TOOL_IMAGE_PLACEHOLDER)
        elif isinstance(part, FilePart) and "file" not in modalities:
            replacement = TextPart(text=NON_FILE_TOOL_PLACEHOLDER)
        changed = changed or replacement is not part
        content.append(replacement)
    return ToolOutput(content=content) if changed else output


def _project_chat_tool_output(
    output: ToolOutput, modalities: tuple[str, ...]
) -> tuple[str, list[ImagePart]]:
    text: list[str] = []
    images: list[ImagePart] = []
    for part in output.content:
        if isinstance(part, TextPart) and part.text:
            text.append(part.text)
        elif isinstance(part, ImagePart):
            if "image" in modalities:
                images.append(part)
            else:
                text.append(NON_VISION_TOOL_IMAGE_PLACEHOLDER)
        elif isinstance(part, FilePart):
            if "file" in modalities:
                text.append(CHAT_FILE_TOOL_PLACEHOLDER)
            else:
                text.append(NON_FILE_TOOL_PLACEHOLDER)

    if not text and images:
        text.append("(see attached image)")
    return "\n".join(text) or "(no tool output)", images


def _tool_images_message(images: list[ImagePart]) -> Message:
    return Message(
        role="user",
        content=[
            TextPart(text=CHAT_TOOL_IMAGE_PROMPT),
            *images,
        ],
    )


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

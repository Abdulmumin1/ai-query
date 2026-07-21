from __future__ import annotations

import json

import pytest

from ai_query import (
    FilePart,
    ImagePart,
    TextPart,
    ToolOutput,
    UnsupportedToolOutputError,
    generate_text,
    step_count_is,
    tool,
)
from ai_query.agents import Agent, SQLiteStorage
from ai_query.agents.turn_codec import turn_event_from_dict, turn_event_to_dict
from ai_query.model import LanguageModel
from ai_query.providers.anthropic import AnthropicProvider
from ai_query.providers.bedrock.provider import BedrockProvider
from ai_query.providers.google import GoogleProvider
from ai_query.providers.openai import OpenAIProvider
from ai_query.types import (
    Message,
    ToolCall,
    ToolCallPart,
    ToolResult,
    ToolResultEvent,
    ToolResultPart,
)
from tests.conftest import (
    MockProvider,
    make_response,
    make_stream_chunks,
    make_tool_call,
)


def rich_output() -> ToolOutput:
    return ToolOutput(
        content=[
            TextPart(text="chart generated"),
            ImagePart(image=b"png-bytes", media_type="image/png"),
            FilePart(
                data=b"pdf-bytes",
                media_type="application/pdf",
                filename="report.pdf",
            ),
        ]
    )


def tool_message(output: ToolOutput | object) -> Message:
    return Message(
        role="tool",
        content=[
            ToolResultPart(
                tool_result=ToolResult(
                    tool_call_id="call_1",
                    tool_name="render",
                    result=output,
                )
            )
        ],
    )


def test_tool_output_message_round_trip_is_json_safe() -> None:
    stored = tool_message(rich_output()).to_dict()
    encoded = json.dumps(stored)
    restored = Message.from_dict(json.loads(encoded))

    result = restored.content[0].tool_result.result
    assert isinstance(result, ToolOutput)
    assert result.content[1].image == b"png-bytes"
    assert result.content[2].data == b"pdf-bytes"
    assert result.content[2].filename == "report.pdf"


def test_tool_output_rejects_non_content_parts_and_redacts_repr() -> None:
    secret = b"highly-sensitive-image-payload"
    output = ToolOutput(content=[ImagePart(image=secret, media_type="image/png")])

    assert secret.decode() not in repr(output)
    assert secret.decode() not in repr(
        ToolResult("call_1", "render", output)
    )
    with pytest.raises(TypeError, match="only supports"):
        ToolOutput(content=[ToolCallPart(tool_call=ToolCall("1", "x", {}))])  # type: ignore[list-item]


def test_tool_output_round_trips_through_turn_event_codec() -> None:
    call = ToolCall(id="call_1", name="render", arguments={})
    event = ToolResultEvent(
        type="tool_result",
        step_number=1,
        index=0,
        tool_call=call,
        tool_result=ToolResult("call_1", "render", rich_output()),
    )

    restored = turn_event_from_dict(turn_event_to_dict(event))

    assert restored == event


@pytest.mark.asyncio
async def test_generate_text_preserves_tool_output_into_second_request() -> None:
    @tool(description="Render a report")
    def render() -> ToolOutput:
        return rich_output()

    provider = MockProvider(
        responses=[
            make_response(
                text="",
                finish_reason="tool_use",
                tool_calls=[make_tool_call("render", {}, id="call_1")],
            ),
            make_response(text="The report is ready.", finish_reason="stop"),
        ]
    )
    result = await generate_text(
        model=LanguageModel(provider=provider, model_id="mock"),
        prompt="Render it",
        tools={"render": render},
        stop_when=step_count_is(3),
    )

    second_result = provider.last_messages[-1].content[0].tool_result.result
    assert result.text == "The report is ready."
    assert isinstance(second_result, ToolOutput)
    assert result.steps[0].tool_results[0].result is second_result


@pytest.mark.asyncio
async def test_openai_responses_sends_rich_output_on_second_request() -> None:
    class CaptureTransport:
        def __init__(self) -> None:
            self.requests: list[dict] = []
            self.urls: list[str] = []

        async def post(self, url, json, headers=None):
            self.requests.append(json)
            self.urls.append(url)
            if len(self.requests) == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "render",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            return {
                "id": "resp_2",
                "status": "completed",
                "output_text": "I can see the report.",
                "output": [],
            }

        async def get(self, url, headers=None):
            raise AssertionError("inline bytes must not be fetched")

    @tool(description="Render a report")
    def render() -> ToolOutput:
        return rich_output()

    transport = CaptureTransport()
    result = await generate_text(
        model=LanguageModel(
            provider=OpenAIProvider(api_key="test", transport=transport),
            model_id="gpt-5",
        ),
        prompt="Render it",
        tools={"render": render},
        stop_when=step_count_is(3),
    )

    output = next(
        item["output"]
        for item in transport.requests[1]["input"]
        if item.get("type") == "function_call_output"
    )
    assert result.text == "I can see the report."
    assert transport.urls[0].endswith("/chat/completions")
    assert transport.urls[1].endswith("/responses")
    assert any(
        item.get("type") == "function_call" for item in transport.requests[1]["input"]
    )
    assert [part["type"] for part in output] == [
        "input_text",
        "input_image",
        "input_file",
    ]
    assert output[1]["image_url"].startswith("data:image/png;base64,")
    assert output[2]["filename"] == "report.pdf"


@pytest.mark.asyncio
async def test_openai_chat_and_compatible_providers_fail_without_stringifying() -> None:
    openai = OpenAIProvider(api_key="test")
    with pytest.raises(UnsupportedToolOutputError, match="chat/completions"):
        await openai._convert_messages([tool_message(rich_output())])

    from ai_query.providers.deepseek import DeepSeekProvider

    deepseek = DeepSeekProvider(api_key="test")
    with pytest.raises(UnsupportedToolOutputError, match="deepseek chat/completions"):
        await deepseek._convert_messages([tool_message(rich_output())])


@pytest.mark.asyncio
async def test_anthropic_maps_nested_tool_result_blocks() -> None:
    provider = AnthropicProvider(api_key="test")
    _, messages = await provider._convert_messages([tool_message(rich_output())])
    result = messages[0]["content"][0]

    assert result["type"] == "tool_result"
    assert [part["type"] for part in result["content"]] == [
        "text",
        "image",
        "document",
    ]
    assert result["content"][1]["source"]["data"] == "cG5nLWJ5dGVz"


@pytest.mark.asyncio
async def test_google_maps_supported_models_and_rejects_older_models() -> None:
    provider = GoogleProvider(api_key="test")
    _, messages = await provider._convert_messages(
        [tool_message(rich_output())], model="gemini-3-flash-preview"
    )
    response = messages[0]["parts"][0]["functionResponse"]

    assert response["response"] == {"result": "chart generated"}
    assert len(response["parts"]) == 2
    assert response["parts"][0]["inlineData"]["mimeType"] == "image/png"

    with pytest.raises(UnsupportedToolOutputError, match="gemini-2.5-flash"):
        await provider._convert_messages(
            [tool_message(rich_output())], model="gemini-2.5-flash"
        )


def test_bedrock_maps_typed_tool_result_blocks() -> None:
    provider = object.__new__(BedrockProvider)
    _, messages = provider._convert_messages(
        [tool_message(rich_output())], model="amazon.nova-pro-v1:0"
    )
    content = messages[0]["content"][0]["toolResult"]["content"]

    assert content[0] == {"text": "chart generated"}
    assert content[1]["image"]["source"]["bytes"] == b"png-bytes"
    assert content[2]["document"]["name"] == "report.pdf"

    with pytest.raises(UnsupportedToolOutputError, match="meta.llama"):
        provider._convert_messages(
            [tool_message(rich_output())], model="meta.llama3-70b-instruct-v1:0"
        )


@pytest.mark.asyncio
async def test_agent_sqlite_persistence_restores_tool_output(tmp_path) -> None:
    @tool(description="Render a report")
    def render() -> ToolOutput:
        return rich_output()

    provider = MockProvider(
        stream_chunks=[
            make_stream_chunks(
                "", [make_tool_call("render", {}, id="call_1")]
            ),
            make_stream_chunks("done"),
        ]
    )
    db_path = str(tmp_path / "agent.db")
    storage = SQLiteStorage(db_path)
    agent = Agent(
        "rich",
        model=LanguageModel(provider=provider, model_id="mock"),
        storage=storage,
        tools={"render": render},
        stop_when=step_count_is(3),
    )
    await agent.start()
    await agent.chat("render")
    await agent.stop()
    storage.close()

    restored_storage = SQLiteStorage(db_path)
    restored = Agent("rich", storage=restored_storage)
    await restored.start()
    output = restored.messages[2].content[0].tool_result.result

    assert isinstance(output, ToolOutput)
    assert output.content[1].image == b"png-bytes"
    await restored.stop()
    restored_storage.close()


def test_scalar_tool_results_remain_unchanged() -> None:
    message = tool_message({"temperature": 24})
    stored = message.to_dict()
    restored = Message.from_dict(stored)

    assert stored["content"][0]["tool_result"]["result"] == {"temperature": 24}
    assert restored.content[0].tool_result.result == {"temperature": 24}

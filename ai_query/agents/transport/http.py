"""HTTP Transport for agent-to-agent communication."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Union, TYPE_CHECKING

import httpx

from .base import AgentTransport

if TYPE_CHECKING:
    from ai_query.types import AbortSignal


class HTTPTransport(AgentTransport):
    """Transport that communicates with agents via HTTP/REST.

    This transport allows calling agents hosted on remote servers or serverless
    functions. It implements the standard Wire Protocol defined in the
    Transport Plan.

    Example:
        transport = HTTPTransport(
            base_url="https://api.myapp.com/agents",
            headers={"Authorization": "Bearer token"}
        )
        # Calls https://api.myapp.com/agents/researcher
        result = await transport.invoke("researcher", {"method": "search"})
    """

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        """Initialize the HTTP transport.

        Args:
            base_url: Base URL for agent endpoints. If provided, agent_id is appended.
                      If empty, agent_id is treated as a full URL.
            headers: Default headers to send with requests (e.g., auth).
            client: Optional existing httpx client to reuse.
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers)
        return self._client

    def _get_url(self, agent_id: str) -> str:
        if self.base_url:
            return f"{self.base_url}/{agent_id}"
        # If no base_url, assume agent_id is the full URL or handle appropriately
        if agent_id.startswith("http"):
            return agent_id
        raise ValueError("No base_url provided and agent_id is not a URL")

    async def invoke(
        self,
        agent_id: str,
        payload: dict[str, Any],
        timeout: float = 30.0,
        signal: Union["AbortSignal", None] = None,
    ) -> dict[str, Any]:
        """Invoke a remote agent via HTTP POST.

        This sends a standardized request:
        POST {url}
        {
            "action": "invoke",
            "payload": { ... }
        }
        """
        client = await self._get_client()
        url = self._get_url(agent_id)

        # Wire Protocol: Wrap the payload in an 'invoke' action
        request_body = {"action": "invoke", "payload": payload}

        # If signal is provided, set up cancellation
        # We wrap the request in a task so we can cancel it
        async def _make_request():
            response = await client.post(url, json=request_body, timeout=timeout)
            response.raise_for_status()
            return response.json()

        request_task = asyncio.create_task(_make_request())

        if signal:
            signal.add_listener(lambda: request_task.cancel())

        try:
            data = await request_task

            # Unpack result from Wire Protocol response
            if "error" in data:
                # The agent returned a logic error
                return {"error": data["error"]}

            if "result" in data:
                # Success
                return {"result": data["result"]}

            # Fallback for unexpected formats
            return data

        except asyncio.CancelledError:
            if signal and signal.aborted:
                raise asyncio.TimeoutError(f"Request aborted: {signal.reason}")
            raise
        except httpx.HTTPStatusError as e:
            # Try to read error message from body
            try:
                err_data = e.response.json()
                msg = err_data.get("error", str(e))
            except Exception:
                msg = str(e)
            return {"error": f"HTTP Error: {msg}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}

    async def chat(
        self,
        agent_id: str,
        message: str,
        timeout: float = 30.0,
        signal: Union["AbortSignal", None] = None,
    ) -> str:
        """Send a chat message to a remote agent."""
        client = await self._get_client()
        url = self._get_url(agent_id)

        request_body = {"action": "chat", "message": message}

        async def _make_request():
            response = await client.post(url, json=request_body, timeout=timeout)
            response.raise_for_status()
            return response.json()

        request_task = asyncio.create_task(_make_request())

        if signal:
            signal.add_listener(lambda: request_task.cancel())

        try:
            data = await request_task
            return data.get("response", "")
        except asyncio.CancelledError:
            if signal and signal.aborted:
                raise asyncio.TimeoutError(f"Request aborted: {signal.reason}")
            raise

    async def stream(
        self,
        agent_id: str,
        message: str,
        timeout: float = 30.0,
        signal: Union["AbortSignal", None] = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response from a remote agent via SSE (POST)."""
        client = await self._get_client()
        url = self._get_url(agent_id)

        # Wire Protocol: Streaming Request via POST
        request_body = {"action": "chat", "message": message, "stream": True}

        try:
            async with client.stream(
                "POST",
                url,
                json=request_body,
                headers={"Accept": "text/event-stream"},
                timeout=timeout,
            ) as response:
                response.raise_for_status()

                # Setup signal listener to close response if aborted
                if signal:
                    signal.add_listener(lambda: asyncio.create_task(response.aclose()))

                current_event = None
                async for line in response.aiter_lines():
                    if signal:
                        signal.throw_if_aborted()

                    if not line.strip():
                        # Empty line marks end of SSE event
                        current_event = None
                        continue

                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                    elif line.startswith("data: "):
                        data_str = line[6:]
                        # Only yield chunks, not start/end/error events
                        if current_event == "chunk" and data_str:
                            try:
                                # Decode JSON data
                                chunk = json.loads(data_str)
                                if isinstance(chunk, str):
                                    yield chunk
                            except json.JSONDecodeError:
                                # Fallback: yield raw data if not valid JSON
                                yield data_str
                        elif current_event == "error":
                            try:
                                error_msg = json.loads(data_str)
                                raise Exception(f"Stream error: {error_msg}")
                            except json.JSONDecodeError:
                                raise Exception(f"Stream error: {data_str}")
        except asyncio.CancelledError:
            if signal and signal.aborted:
                raise asyncio.TimeoutError(f"Request aborted: {signal.reason}")
            raise

    async def close(self) -> None:
        """Close the underlying client if we own it."""
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

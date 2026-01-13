"""ChatAgent mixin for AI-powered conversations."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Generic, TypeVar, TYPE_CHECKING

from ai_query.agents.base import Agent
from ai_query.agents.output import AgentOutput, NullOutput, PersistingOutput, BroadcastOutput, QueueOutput
from ai_query.types import Message, StopCondition, StepStartEvent, StepFinishEvent
from ai_query.model import LanguageModel

if TYPE_CHECKING:
    from ai_query.types import ProviderOptions

State = TypeVar("State")


class ChatAgent(Agent[State], Generic[State]):
    """
    Agent mixin that adds AI chat capabilities.

    Combine with a storage backend to create a chat bot:

        class MyBot(ChatAgent, InMemoryAgent):
            system = "You are a helpful assistant."

        async with MyBot("bot-1") as bot:
            response = await bot.chat("Hello!")
            print(response)

    Attributes:
        model: The LanguageModel to use (default: google("gemini-2.0-flash")).
        stop_when: Stop condition(s) for tool execution loops.
        system: The system prompt for the AI.
        tools: Dict of tools available to the AI.
    """

    from ai_query.providers.google import google

    model: LanguageModel = google('gemini-2.0-flash')
    stop_when: StopCondition | list[StopCondition] | None = None
    system: str = "You are a helpful assistant."
    provider_options: ProviderOptions | None = None
    tools: dict[str, Any] = {}

    _output: AgentOutput | None = None

    @property
    def output(self) -> AgentOutput:
        """Safe access to current output channel (no-op if None)."""
        return self._output or BroadcastOutput(self)

    def on_step_start(self, event: StepStartEvent) -> None:
        """
        Called when a generation step starts.

        Override this to hook into the AI generation process.

        Args:
            event: StepStartEvent with step information.
        """
        pass

    def on_step_finish(self, event: StepFinishEvent) -> None:
        """
        Called when a generation step finishes.

        Override this to hook into the AI generation process.

        Args:
            event: StepFinishEvent with step results.
        """
        pass

    async def chat(self, message: str, output: AgentOutput | None = None) -> str:
        """
        Send a message and get an AI response.

        This method:
        1. Adds the user message to history
        2. Calls the AI model with the conversation
        3. Adds the assistant response to history
        4. Persists the updated history

        If output is provided (or defaults to BroadcastOutput), it will stream
        start/chunk/end events to it, even though this method awaits the full result.

        Args:
            message: The user's message.
            output: Optional output channel for intermediate feedback.

        Returns:
            The AI's response text.
        """
        # Import here to avoid circular imports
        from ai_query import stream_text

        # Set output context
        previous_output = self._output

        # Use provided output or default broadcast output
        effective_output = output if output is not None else BroadcastOutput(self)

        # Wrap output if event logging is enabled
        if self.enable_event_log:
            effective_output = PersistingOutput(effective_output, self)

        self._output = effective_output

        try:
            # Add user message
            self._messages.append(Message(role="user", content=message))

            # Send start event
            await self.output.send_event("ai_start", {})

            # Stream response to capture chunks for broadcasting
            result = stream_text(
                model=self.model,
                system=self.system,
                messages=self._messages,
                tools=self.tools if self.tools else None,
                on_step_start=self.on_step_start,
                on_step_finish=self.on_step_finish,
                stop_when=self.stop_when,
                provider_options=self.provider_options,
            )

            full_response = ""
            async for chunk in result.text_stream:
                full_response += chunk
                # Broadcast chunk
                await self.output.send_event("ai_chunk", {"content": chunk})

            # Send end event
            await self.output.send_event("ai_end", {"content": full_response})

            # Add assistant response
            self._messages.append(Message(role="assistant", content=full_response))

            # Persist
            await self._save_messages(self._messages)

            return full_response
        finally:
            # Restore previous output context
            self._output = previous_output

    async def handle_request_stream(self, request: dict[str, Any]) -> AsyncIterator[str]:
        """Serverless streaming request handler.

        Handles streaming requests, primarily for 'chat' action.
        Yields SSE-formatted events:
        - event: start (empty data)
        - event: chunk (text delta)
        - event: status (status update)
        - event: message (message event)
        - event: end (full accumulated text)
        - event: error (error message)

        Args:
            request: The request with 'action' and action-specific fields.

        Yields:
            SSE formatted strings.
        """
        # Ensure agent is started
        if self._state is None:
            await self.start()

        action = request.get("action", "chat")

        if action == "chat":
            message = request.get("message", "")

            # Create a queue-based output to capture side effects
            queue = asyncio.Queue()
            output = QueueOutput(queue)

            # Define the background task
            async def run_chat():
                try:
                    # Manually send start event
                    await output.send_event("start", {})

                    full_text = ""
                    # Stream chunks and push to queue as chunks
                    # Note: we pass output to stream_chat so it sets up the context
                    # for hooks to use.
                    async for chunk in self.stream_chat(message, output=output):
                        full_text += chunk
                        # Send text chunk event
                        await output.send_event("chunk", {"content": chunk})

                    # Send end event
                    await output.send_event("end", {"content": full_text})

                except Exception as e:
                    await output.send_error(str(e))
                finally:
                    # Signal end of stream
                    await queue.put(None)

            # Start chat in background
            asyncio.create_task(run_chat())

            # Yield events from queue
            while True:
                event = await queue.get()
                if event is None:
                    break

                # Format as SSE
                event_type = event.get("type", "message")
                # Remove type from data payload to avoid duplication if desired,
                # but AgentOutput adapters usually keep it or flatten it.
                # QueueOutput stores {"type": ..., ...data...}

                # We need to extract data part.
                # QueueOutput.send_message -> {"type": "message", "content": "..."}
                # QueueOutput.send_status -> {"type": "status", "status": "...", "details": ...}
                # QueueOutput.send_event -> {"type": event, ...data}

                # Copy dict to avoid modifying original
                data = event.copy()
                if "type" in data:
                    del data["type"]

                # Escape newlines for SSE data
                safe_data = json.dumps(data)
                yield f"event: {event_type}\ndata: {safe_data}\n\n"

        else:
            yield f"event: error\ndata: Streaming not supported for action: {action}\n\n"

    async def stream_chat(self, message: str, output: AgentOutput | None = None) -> AsyncIterator[str]:
        """
        Stream an AI response.

        Similar to chat(), but yields response chunks as they arrive.

        Args:
            message: The user's message.
            output: Optional output channel for intermediate feedback.

        Yields:
            Response text chunks.
        """
        # Import here to avoid circular imports
        from ai_query import stream_text

        # Set output context
        previous_output = self._output
        self._output = output

        try:
            # Add user message
            self._messages.append(Message(role="user", content=message))

            # Stream response with step hooks
            result = stream_text(
                model=self.model,
                system=self.system,
                messages=self._messages,
                tools=self.tools if self.tools else None,
                on_step_start=self.on_step_start,
                on_step_finish=self.on_step_finish,
                stop_when=self.stop_when,
            )

            full_response = ""
            async for chunk in result.text_stream:
                full_response += chunk
                yield chunk

            # Add assistant response
            self._messages.append(Message(role="assistant", content=full_response))

            # Persist
            await self._save_messages(self._messages)
        finally:
            # Restore previous output context
            self._output = previous_output

    async def stream_chat_sse(self, message: str) -> str:
        """
        Stream an AI response via SSE to connected clients.

        Uses SSE (Server-Sent Events) for efficient streaming to clients
        while processing the AI response. Returns the full response.

        Args:
            message: The user's message.

        Returns:
            The complete AI response text.
        """
        # Set output context to ensure events go through the proper channel (and are persisted)
        previous_output = self._output
        output = self.output

        # Wrap output if event logging is enabled
        if self.enable_event_log and output:
            output = PersistingOutput(output, self)

        self._output = output

        try:
            # Use send_event to ensure persistence
            await output.send_event("ai_start", {})

            full_response = ""
            async for chunk in self.stream_chat(message):
                full_response += chunk
                await output.send_event("ai_chunk", {"content": chunk})

            await output.send_event("ai_end", {"content": full_response})
            return full_response
        finally:
            self._output = previous_output

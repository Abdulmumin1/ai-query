"""ChatAgent mixin for AI-powered conversations."""

from __future__ import annotations

from typing import Any, AsyncIterator, Generic, TypeVar, TYPE_CHECKING

from ai_query.agents.base import Agent
from ai_query.agents.output import AgentOutput, NullOutput
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
        return self._output or NullOutput()

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

        Args:
            message: The user's message.
            output: Optional output channel for intermediate feedback.

        Returns:
            The AI's response text.
        """
        # Import here to avoid circular imports
        from ai_query import generate_text

        # Set output context
        previous_output = self._output
        self._output = output

        try:
            # Add user message
            self._messages.append(Message(role="user", content=message))

            # Generate response with step hooks
            result = await generate_text(
                model=self.model,
                system=self.system,
                messages=self._messages,
                tools=self.tools if self.tools else None,
                on_step_start=self.on_step_start,
                on_step_finish=self.on_step_finish,
                stop_when=self.stop_when,
                provider_options=self.provider_options,
            )

            # Add assistant response
            self._messages.append(Message(role="assistant", content=result.text))

            # Persist
            await self._save_messages(self._messages)

            return result.text
        finally:
            # Restore previous output context
            self._output = previous_output

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
        await self.stream_to_sse("ai_start", "")
        
        full_response = ""
        async for chunk in self.stream_chat(message):
            full_response += chunk
            await self.stream_to_sse("ai_chunk", chunk)
        
        await self.stream_to_sse("ai_end", full_response)
        return full_response

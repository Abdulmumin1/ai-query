"""Example: Using Agent with stream_text directly.

This shows how to use stream_text directly within an Agent for full control
over AI interactions, while using emit() for events.

Usage:
    uv run examples/agent_direct.py

Connect:
    curl http://localhost:8080/events  # SSE stream
    curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"message": "Hello @ai"}'
"""

from ai_query import Agent, MemoryStorage, stream_text, google, tool, Field


class CustomAIAgent(Agent):
    """Agent using stream_text directly instead of agent.stream()."""

    def __init__(self):
        super().__init__(
            "custom-agent",
            storage=MemoryStorage(),
            initial_state={"query_count": 0, "last_topic": None},
        )

    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        username = ctx.metadata.get("username", "Anonymous")
        connection.username = username
        await connection.send(f"Welcome {username}! Ask me anything with @ai")

    async def on_message(self, connection, message):
        username = getattr(connection, "username", "Anonymous")

        if "@ai" in message.lower():
            await self._handle_ai_query(connection, username, message)
        else:
            await connection.send(f"Echo: {message}")

    async def _handle_ai_query(self, connection, username: str, message: str):
        """Handle AI query using stream_text directly."""

        # Update state
        await self.update_state(
            query_count=self.state.get("query_count", 0) + 1,
            last_topic=message[:50],
        )

        # Define tools with access to agent via closure
        @tool(description="Save a note for the user")
        async def save_note(note: str = Field(description="Note to save")) -> str:
            notes = self.state.get("notes", []) + [note]
            await self.update_state(notes=notes)
            await self.emit("note_saved", {"note": note})
            return f"Saved: {note}"

        @tool(description="Get the current query count")
        def get_stats() -> str:
            return f"Total queries: {self.state.get('query_count', 0)}"

        # Emit start event
        await self.emit("ai_start", {"username": username, "query": message})

        # Use stream_text directly for full control
        result = stream_text(
            model=google("gemini-2.0-flash"),
            system=f"""You are a helpful AI assistant.
            You're talking to {username}.
            Be concise and helpful.
            Use tools when appropriate.""",
            prompt=message,
            tools={"save_note": save_note, "get_stats": get_stats},
        )

        # Stream chunks via emit
        full_response = ""
        async for chunk in result.text_stream:
            full_response += chunk
            await self.emit("chunk", {"content": chunk})

        # Emit completion
        await self.emit("ai_complete", {"response": full_response})

        # Send to WebSocket client
        await connection.send(full_response)

        print(f"AI responded to {username} (query #{self.state.get('query_count', 0)})")


if __name__ == "__main__":
    print("Custom AI Agent Server")
    print("=" * 40)
    print("This example shows using Agent + stream_text directly")
    print("for full control over AI interactions.")
    print()
    print("Endpoints:")
    print("  POST /chat   - Send message")
    print("  GET  /events - SSE stream")
    print("  WS   /ws     - WebSocket")
    print("  GET  /state  - Agent state")
    print()
    CustomAIAgent().serve(port=8080)

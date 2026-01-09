"""Example: Using Agent (not ChatAgent) with WebSocket + SSE.

This shows how to use generate_text/stream_text directly within an Agent
for full control over AI interactions, while using WebSocket for messages
and SSE for streaming.

Usage:
    uv run examples/agent_direct.py
    
Connect: wscat -c 'ws://localhost:8080/ws?username=Alice'
SSE: curl http://localhost:8080/events
"""

from ai_query.agents import InMemoryAgent
from ai_query import stream_text, google, tool, Field


class CustomAIAgent(InMemoryAgent):
    """Agent using stream_text directly instead of ChatAgent."""
    
    initial_state = {
        "query_count": 0,
        "last_topic": None,
    }
    
    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        connection.username = ctx.metadata.get("username", "Anonymous")
        await connection.send(f"Welcome {connection.username}! Ask me anything with @ai")
        await self.broadcast(f"[System] {connection.username} joined")
    
    async def on_message(self, connection, message):
        username = getattr(connection, "username", "Anonymous")
        await self.broadcast(f"{username}: {message}")
        
        if "@ai" in message.lower():
            await self._handle_ai_query(username, message)
    
    async def on_close(self, connection, code, reason):
        username = getattr(connection, "username", "Anonymous")
        await super().on_close(connection, code, reason)
        await self.broadcast(f"[System] {username} left")
    
    async def _handle_ai_query(self, username: str, message: str):
        """Handle AI query using stream_text directly."""
        
        # Update state
        await self.set_state({
            **self.state,
            "query_count": self.state["query_count"] + 1,
            "last_topic": message[:50],
        })
        
        # Define tools with access to agent via closure
        @tool(description="Save a note for the user")
        async def save_note(note: str = Field(description="Note to save")) -> str:
            # Could save to state, database, etc.
            print(f"  Saved note: {note[:30]}...")
            return f"Saved: {note}"
        
        @tool(description="Get the current query count")
        def get_stats() -> str:
            return f"Total queries: {self.state['query_count']}"
        
        # Signal AI is starting via SSE
        await self.stream_to_sse("ai_start", "")
        
        # Use stream_text directly for full control
        result = stream_text(
            model=google("gemini-2.0-flash"),
            system=f"""You are a helpful AI assistant.
            You're talking to {username}.
            Be concise and helpful.
            Use tools when appropriate.""",
            prompt=message,
            tools={
                "save_note": save_note,
                "get_stats": get_stats,
            },
        )
        
        # Stream chunks via SSE
        full_response = ""
        async for chunk in result.text_stream:
            full_response += chunk
            await self.stream_to_sse("ai_chunk", chunk)
        
        # Signal AI is done
        await self.stream_to_sse("ai_end", full_response)
        
        # Also broadcast the final response via WebSocket for clients not using SSE
        await self.broadcast(f"[AI] {full_response}")
        
        print(f"  AI responded to {username} (query #{self.state['query_count']})")


if __name__ == "__main__":
    print("Custom AI Agent Server")
    print("=" * 40)
    print("This example shows using Agent + stream_text directly")
    print("instead of ChatAgent for full control.")
    print()
    print("Connect: wscat -c 'ws://localhost:8080/ws?username=Alice'")
    print("SSE: curl http://localhost:8080/events")
    print()
    CustomAIAgent("custom-agent").serve(port=8080)

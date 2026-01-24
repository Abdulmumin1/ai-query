"""Example: Stateful Research Agent with real-time events."""

import os
from ai_query import tool, Field, step_count_is
from ai_query.agents import Agent, SQLiteStorage
from ai_query.providers import google


class ResearchAgent(Agent):
    """A research agent that emits real-time status events.

    Features demonstrated:
    - Persistent storage with SQLite
    - Real-time event emission with emit()
    - Event replay on reconnection (enable_event_log)
    - Custom state management
    - Tool definitions with type hints
    """

    enable_event_log = True  # Persist events for replay on reconnect

    def __init__(self):
        @tool(description="Search the web for information")
        async def search(query: str = Field(description="Search query")) -> str:
            await self.emit("status", {"text": f"Searching: {query}"})
            # In production, use a real search API
            return f"Found results for: {query}"

        @tool(description="Save a finding to the research notes")
        async def save_finding(
            title: str = Field(description="Finding title"),
            content: str = Field(description="Finding content")
        ) -> str:
            findings = self.state.get("findings", [])
            findings.append({"title": title, "content": content})
            await self.update_state(findings=findings)
            await self.emit("finding_saved", {"title": title})
            return f"Saved: {title}"

        super().__init__(
            "researcher",
            model=google("gemini-2.0-flash"),
            system="""You are a research assistant. When asked to research a topic:
1. Search for relevant information
2. Save important findings
3. Summarize what you learned""",
            storage=SQLiteStorage("./data/research.db"),
            initial_state={"findings": []},
            tools={"search": search, "save_finding": save_finding},
            stop_when=step_count_is(10),
        )

    async def on_start(self):
        findings = self.state.get("findings", [])
        print(f"Agent started. Previous findings: {len(findings)}")

    async def on_message(self, connection, message):
        """Handle WebSocket messages - stream response with events."""
        await self.emit("research_start", {"query": message})

        async for chunk in self.stream(message):
            await self.emit("chunk", {"content": chunk})

        await self.emit("research_complete", {
            "findings_count": len(self.state.get("findings", []))
        })


if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)

    print("Research Agent running on http://localhost:8081")
    print("Endpoints:")
    print("  POST /chat        - Send message, get response")
    print("  GET  /events      - SSE stream of events")
    print("  WS   /ws          - WebSocket connection")
    print("  GET  /state       - Get agent state")
    print()

    ResearchAgent().serve(port=8081)

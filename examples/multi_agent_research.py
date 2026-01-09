"""Per-User AI Research Assistant with Multi-Agent Routing.

This example demonstrates AgentServer for running independent AI assistants.
Each user gets their own isolated agent instance with persistent state.

Usage:
    uv run examples/multi_agent_research.py

Then connect:
    ws://localhost:8080/agent/user-alice/ws
    ws://localhost:8080/agent/user-bob/ws
    
Each user's research history and findings are isolated.
"""

import json
from ai_query.agents import ChatAgent, InMemoryAgent, AgentServer, AgentServerConfig
from ai_query import tool, Field
from ai_query.providers.google import google


class ResearchAssistant(ChatAgent, InMemoryAgent):
    """Per-user AI research assistant with persistent findings."""
    
    model = google("gemini-2.0-flash")
    system = """You are a research assistant. Help users research topics, 
    save important findings, and recall previous research. Use tools when helpful."""
    
    initial_state = {
        "research_topics": [],
        "findings": [],
        "total_queries": 0,
    }
    
    @property
    def tools(self):
        @tool(description="Save an important research finding for later")
        async def save_finding(
            topic: str = Field(description="Topic of the finding"),
            finding: str = Field(description="The finding to save"),
        ) -> str:
            findings = self.state["findings"] + [{
                "topic": topic,
                "finding": finding,
            }]
            topics = list(set(self.state["research_topics"] + [topic]))
            await self.set_state({
                **self.state, 
                "findings": findings,
                "research_topics": topics,
            })
            return f"Saved finding #{len(findings)} under topic '{topic}'"
        
        @tool(description="List all saved findings, optionally filtered by topic")
        async def list_findings(
            topic: str = Field(default="", description="Optional topic filter"),
        ) -> str:
            findings = self.state["findings"]
            if topic:
                findings = [f for f in findings if f["topic"] == topic]
            if not findings:
                return "No findings saved yet."
            return "\n".join([
                f"- [{f['topic']}] {f['finding']}" 
                for f in findings
            ])
        
        @tool(description="Get summary of research activity")
        def get_stats() -> str:
            return json.dumps({
                "total_queries": self.state["total_queries"],
                "topics_researched": self.state["research_topics"],
                "findings_count": len(self.state["findings"]),
            }, indent=2)
        
        return {
            "save_finding": save_finding,
            "list_findings": list_findings,
            "get_stats": get_stats,
        }
    
    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        user_id = self.id.replace("user-", "")
        
        # Send personalized welcome
        findings_count = len(self.state["findings"])
        if findings_count > 0:
            await connection.send(
                f"Welcome back! You have {findings_count} saved findings. "
                f"Ask me to recall them or continue your research."
            )
        else:
            await connection.send(
                f"Hello! I'm your research assistant. "
                f"I can help you research topics and save important findings."
            )
        print(f"[{self.id}] User connected")
    
    async def on_message(self, connection, message):
        # Track query count
        await self.set_state({
            **self.state,
            "total_queries": self.state["total_queries"] + 1,
        })
        
        # Stream AI response via SSE
        print(f"[{self.id}] Query: {message[:50]}...")
        response = await self.stream_chat_sse(message)
        await connection.send(response)
    
    async def on_close(self, connection, code, reason):
        await super().on_close(connection, code, reason)
        print(f"[{self.id}] User disconnected")


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Research Assistant Server")
    print("=" * 60)
    print()
    print("Each user gets their own AI assistant instance with:")
    print("  - Isolated state and research history")
    print("  - Persistent findings (saved between sessions)")
    print("  - Tools for saving and recalling research")
    print()
    print("Endpoints:")
    print("  WebSocket: ws://localhost:8080/agent/{user_id}/ws")
    print("  SSE:       http://localhost:8080/agent/{user_id}/events")
    print("  State:     http://localhost:8080/agent/{user_id}/state")
    print()
    print("Example user IDs: user-alice, user-bob, user-research-team")
    print()
    
    config = AgentServerConfig(
        idle_timeout=600,          # Evict after 10 min idle
        max_agents=50,             # Max 50 concurrent users
        enable_rest_api=True,      # Enable state inspection
        enable_list_agents=True,   # Enable /agents (for demo)
    )
    
    AgentServer(ResearchAssistant, config=config).serve(port=8080)

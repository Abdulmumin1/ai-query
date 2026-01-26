"""
Example Cloudflare Worker script using ai-query Durable Objects.

This example demonstrates a complete setup for a stateful "Researcher" agent
running on Cloudflare Workers.

Prerequisites:
- A Cloudflare account
- Wrangler CLI installed (`npm install -g wrangler`)

Setup:
1. Create a `wrangler.toml` (see below)
2. Run `wrangler deploy`

"""

# --- 1. The Agent Logic (Platform Agnostic) ---
from ai_query import Agent, action

class ResearcherAgent(Agent):
    """An agent that performs research and remembers it."""
    
    @action
    async def research(self, topic: str):
        # Emit status update (sent via WebSocket if connected)
        await self.emit("status", {"text": f"Starting research on {topic}..."})
        
        # Persist state automatically using Durable Object storage
        history = self.state.get("history", [])
        history.append(topic)
        self.state["history"] = history
        
        # Simulate work
        import asyncio
        await asyncio.sleep(1) 
        
        result = f"Detailed report on {topic}"
        
        await self.emit("result", {"summary": result})
        return result

    @action
    async def get_history(self):
        return self.state.get("history", [])

# --- 2. The Cloudflare Adapter ---
from ai_query.adapters.cloudflare import AgentDO, CloudflareRegistry

class ResearcherDO(AgentDO):
    """The Durable Object wrapper.
    
    This class bridges the Cloudflare DO lifecycle with the Agent lifecycle.
    It inherits from `workers.DurableObject` and handles storage/networking.
    """
    agent_class = ResearcherAgent


# --- 3. The Worker Entry Point ---
async def fetch(request, env):
    """Main Worker fetch handler.
    
    Routes incoming HTTP and WebSocket requests to the correct Durable Object.
    """
    
    # The Registry knows how to find DOs based on bindings
    registry = CloudflareRegistry(env)
    
    # Register the route.
    # Pattern: "researcher-.*" matches agent IDs like "researcher-123"
    # Binding: env.RESEARCHER (defined in wrangler.toml)
    registry.register("researcher-.*", env.RESEARCHER)
    
    # Handles:
    # - POST /agent/researcher-123/chat (Chat)
    # - POST /agent/researcher-123/invoke (Action)
    # - WebSocket /agent/researcher-123 (Real-time)
    return await registry.handle_request(request)


# --- Appendix: Example wrangler.toml ---
"""
name = "ai-query-worker"
main = "worker.py"
compatibility_date = "2024-04-03"

[durable_objects]
bindings = [
  { name = "RESEARCHER", class_name = "ResearcherDO" }
]

[[migrations]]
tag = "v1"
new_classes = ["ResearcherDO"]

[observability]
enabled = true
"""

"""Battle Test: End-to-end multi-agent workflow."""

import asyncio
import os
from ai_query import Agent, action, AgentRegistry, AgentServer, HTTPTransport, connect, generate_text
from ai_query.providers import google

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

# --- Agent Definitions ---

class ResearcherAgent(Agent):
    """Researches a topic using a real LLM."""
    @action
    async def research(self, topic: str) -> str:
        print(f"[{self.id}] Researching topic: {topic}")
        res = await generate_text(
            model=google("gemini-2.5-flash", api_key=GOOGLE_API_KEY),
            prompt=f"Provide a concise, 3-sentence summary on the topic: {topic}"
        )
        print(f"[{self.id}] Research complete.")
        return res.text

class WriterAgent(Agent):
    """Writes a summary based on research."""
    @action
    async def write(self, research_summary: str) -> str:
        print(f"[{self.id}] Writing content based on research...")
        res = await generate_text(
            model=google("gemini-2.5-flash", api_key=GOOGLE_API_KEY),
            prompt=f"Write a short blog post intro based on this summary:\n\n{research_summary}"
        )
        print(f"[{self.id}] Writing complete.")
        return res.text

class CoordinatorAgent(Agent):
    """Orchestrates the research and writing workflow."""
    @action
    async def create_article(self, topic: str) -> dict:
        print(f"[{self.id}] Starting workflow for topic: {topic}")
        
        # Call the remote researcher
        researcher_proxy = self.call("researcher", agent_cls=ResearcherAgent)
        research_summary = await researcher_proxy.research(topic=topic)
        print(f"[{self.id}] Got research summary: '{research_summary[:30]}...'")
        
        # Call the remote writer
        writer_proxy = self.call("writer", agent_cls=WriterAgent)
        article_intro = await writer_proxy.write(research_summary=research_summary)
        print(f"[{self.id}] Got article intro: '{article_intro[:30]}...'")
        
        return {
            "topic": topic,
            "research": research_summary,
            "article": article_intro
        }

# --- Remote Server (Simulates a separate microservice) ---

def run_remote_server():
    """Runs the remote agent server."""
    print("Starting remote server on port 8091...")
    registry = AgentRegistry()
    registry.register("researcher", ResearcherAgent)
    registry.register("writer", WriterAgent)
    
    server = AgentServer(registry)
    server.serve(port=8091)

# --- Main Server (Runs the coordinator) ---

def run_main_server():
    """Runs the main coordinator agent server."""
    print("Starting main server on port 8090...")
    registry = AgentRegistry()
    registry.register("coordinator", CoordinatorAgent)
    
    # Register the remote agents so the coordinator knows where to find them
    remote_transport = HTTPTransport(base_url="http://localhost:8091/agent")
    registry.register("researcher", remote_transport)
    registry.register("writer", remote_transport)
    
    server = AgentServer(registry)
    server.serve(port=8090)

# --- Test Client ---

async def run_test_client():
    """Connects to the main server and runs the workflow."""
    print("\n--- Running Battle Test Client ---")
    await asyncio.sleep(2) # Wait for servers to start
    
    try:
        coordinator = connect("http://localhost:8090/agent/coordinator")
        
        topic = "The future of artificial intelligence in education"
        print(f"Client: Requesting article on '{topic}'...")
        
        result = await coordinator.call().create_article(topic=topic)
        
        print("\n--- Battle Test SUCCESS ---")
        print(f"Final Article Intro:\n{result['article']}")
        
        assert "education" in result["article"].lower()
        
    except Exception as e:
        print(f"\n--- Battle Test FAILED ---")
        print(f"Error: {e}")
    finally:
        # In a real script, we'd have a way to shut down the servers.
        # For this test, we'll just exit.
        print("\n--- Battle Test Complete ---")
        # This is a bit abrupt, but necessary for an automated script.
        import os
        os._exit(0)


if __name__ == "__main__":
    import multiprocessing
    
    # Run servers in separate processes
    p_remote = multiprocessing.Process(target=run_remote_server)
    p_main = multiprocessing.Process(target=run_main_server)
    
    p_remote.start()
    p_main.start()
    
    # Run client
    asyncio.run(run_test_client())

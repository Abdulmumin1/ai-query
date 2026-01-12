"""Example using ChatAgent with MCP server tools."""

import asyncio
from ai_query.agents import ChatAgent, SQLiteAgent
from ai_query.providers.openrouter import openrouter
from ai_query import connect_mcp_http, step_count_is


class MCPBot(ChatAgent, SQLiteAgent):
    """Chatbot that uses tools from an MCP server."""
    
    db_path = "./mcp_bot.db"
    model = openrouter("z-ai/glm-4.5-air:free")
    system = "You are a helpful assistant. Use tools when needed."
    initial_state = {"message_count": 0}
    
    # Will be set after connecting to MCP
    _mcp_tools: dict = {}
    
    @property
    def stop_when(self):
        return step_count_is(10)
    
    @property
    def tools(self):
        return self._mcp_tools
    
    def on_step_finish(self, event):
        if event.step.tool_calls:
            for tc in event.step.tool_calls:
                print(f"  [Tool] {tc.name}")
    
    async def on_start(self):
        print(f"Bot started! Previous messages: {len(self.messages)}")


async def main():
    # Connect to MCP server first
    print("Connecting to MCP server...")
    mcp_server = await connect_mcp_http("https://ai-query.dev/mcp")
    
    try:
        print(f"Connected! Available tools: {list(mcp_server.tools.keys())}")
        
        async with MCPBot("mcp-bot2") as bot:
            # Inject MCP tools into the bot
            bot._mcp_tools = mcp_server.tools
            
            while True:
                print("\n--- Chat ---")
                question = input("Enter your message (or 'quit'): ")
                
                if question.lower() in ('quit', 'exit', 'q'):
                    break
                
                print("\nBot: ", end="", flush=True)
                async for chunk in bot.stream_chat(question):
                    print(chunk, end="", flush=True)
                print()
                
                print(f"\nTotal messages: {len(bot.messages)}")
    finally:
        # Always close the MCP connection
        await mcp_server.close()
        print("MCP server disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
"""Real-time chat room server with AI assistant.

Usage:
    uv run examples/realtime/server.py

Then connect with the client:
    uv run examples/realtime/client.py --username Alice
"""

from ai_query.agents import ChatAgent, InMemoryAgent
from ai_query.providers.google import google


class ChatRoom(ChatAgent, InMemoryAgent):
    """Real-time chat room with AI assistant."""
    
    model = google("gemini-2.0-flash")
    system = """You are an AI assistant in a group chat room.
    Be helpful, friendly, and concise.
    When someone mentions @ai, respond to them."""
    
    initial_state = {
        "participants": [],
        "message_count": 0,
    }
    
    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        
        # Get username from query params
        username = ctx.metadata.get("username", "Anonymous")
        connection.username = username
        
        # Update participants
        participants = self.state["participants"] + [username]
        await self.set_state({**self.state, "participants": participants})
        
        # Welcome message
        await connection.send(f"[System] Welcome {username}! {len(participants)} online.")
        await self.broadcast(f"[System] {username} joined the chat")
        print(f"+ {username} connected ({len(participants)} total)")
    
    async def on_message(self, connection, message):
        username = getattr(connection, "username", "Anonymous")
        
        # Broadcast user message
        await self.broadcast(f"{username}: {message}")
        
        # Track message count
        await self.set_state({
            **self.state,
            "message_count": self.state["message_count"] + 1
        })
        
        # Respond if @ai is mentioned
        if "@ai" in message.lower():
            print(f"  AI responding to: {message[:50]}...")
            # Use SSE for efficient AI streaming
            await self.stream_chat_sse(f"{username} says: {message}")
    
    async def on_close(self, connection, code, reason):
        username = getattr(connection, "username", "Anonymous")
        await super().on_close(connection, code, reason)
        
        # Update participants
        participants = [p for p in self.state["participants"] if p != username]
        await self.set_state({**self.state, "participants": participants})
        
        await self.broadcast(f"[System] {username} left the chat")
        print(f"- {username} disconnected ({len(participants)} remaining)")


if __name__ == "__main__":
    print("Starting chat room server...")
    print("Connect with: uv run examples/realtime/client.py --username YourName")
    print()
    ChatRoom("main-room").serve(host="localhost", port=8080, path="/ws")

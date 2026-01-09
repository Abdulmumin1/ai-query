"""FastAPI integration example with WebSocket + SSE for AI streaming.

This shows how to integrate ai-query agents with FastAPI.

Usage:
    pip install fastapi uvicorn
    uv run examples/fastapi_realtime.py

Connect:
    - WebSocket: ws://localhost:8000/ws?username=Alice
    - SSE: http://localhost:8000/events
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ai_query.agents import ChatAgent, InMemoryAgent, Connection, ConnectionContext


# ─── Custom FastAPI Connection ──────────────────────────────────────────

class FastAPIConnection(Connection):
    """Wraps FastAPI WebSocket in our Connection interface."""
    
    def __init__(self, ws: WebSocket):
        self._ws = ws
        self.username: str | None = None
    
    async def send(self, message: str | bytes) -> None:
        await self._ws.send_text(message)
    
    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code)


# ─── Agent Definition ───────────────────────────────────────────────────

class ChatRoom(ChatAgent, InMemoryAgent):
    """Chat room with AI assistant."""
    
    initial_state = {"participants": [], "message_count": 0}
    system = "You are a helpful AI assistant. Be concise."
    
    async def on_connect(self, connection, ctx):
        await super().on_connect(connection, ctx)
        connection.username = ctx.metadata.get("username", "Anonymous")
        await self.set_state({
            **self.state,
            "participants": self.state["participants"] + [connection.username]
        })
        await self.broadcast(f"[System] {connection.username} joined")
        print(f"+ {connection.username} connected")
    
    async def on_message(self, connection, message):
        username = connection.username
        await self.broadcast(f"{username}: {message}")
        
        if "@ai" in message.lower():
            print(f"  AI responding to: {message[:40]}...")
            await self.stream_chat_sse(f"{username} says: {message}")
    
    async def on_close(self, connection, code, reason):
        username = getattr(connection, "username", "Anonymous")
        await super().on_close(connection, code, reason)
        await self.set_state({
            **self.state,
            "participants": [p for p in self.state["participants"] if p != username]
        })
        await self.broadcast(f"[System] {username} left")
        print(f"- {username} disconnected")


# ─── FastAPI App ────────────────────────────────────────────────────────

room = ChatRoom("main-room")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start agent on app startup."""
    await room.start()
    print("Chat room started")
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, username: str = "Anonymous"):
    """WebSocket endpoint for chat messages."""
    await websocket.accept()
    
    # Create Connection and Context
    connection = FastAPIConnection(websocket)
    ctx = ConnectionContext(request=websocket, metadata={"username": username})
    
    await room.on_connect(connection, ctx)
    
    try:
        while True:
            message = await websocket.receive_text()
            await room.on_message(connection, message)
    except WebSocketDisconnect:
        pass
    finally:
        await room.on_close(connection, 1000, "Disconnected")


@app.get("/events")
async def sse_endpoint():
    """SSE endpoint for AI streaming."""
    
    async def event_generator():
        # Create a queue for this connection
        queue: asyncio.Queue[str] = asyncio.Queue()
        
        # Wrapper to intercept SSE messages
        original_stream_to_sse = room.stream_to_sse
        
        async def capture_sse(event: str, data: str):
            await original_stream_to_sse(event, data)
            await queue.put(f"event: {event}\ndata: {data}\n\n")
        
        room.stream_to_sse = capture_sse
        
        try:
            while True:
                try:
                    # Wait for events with timeout for keepalive
                    message = await asyncio.wait_for(queue.get(), timeout=30)
                    yield message
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            room.stream_to_sse = original_stream_to_sse
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("WebSocket: ws://localhost:8000/ws?username=YourName")
    print("SSE: http://localhost:8000/events")
    uvicorn.run(app, host="localhost", port=8000)

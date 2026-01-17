"""FastAPI integration example with Agent events.

This shows how to integrate ai-query agents with FastAPI using the emit() pattern.

Usage:
    pip install fastapi uvicorn
    uv run examples/fastapi_realtime.py

Connect:
    - SSE: http://localhost:8000/events
    - Chat: POST http://localhost:8000/chat
"""

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_query import Agent, MemoryStorage, tool, Field, google


# ─── Agent Definition ───────────────────────────────────────────────────

class ChatRoom(Agent):
    """Chat room with AI assistant using emit() for events."""

    def __init__(self):
        @tool(description="Save a note")
        async def save_note(note: str = Field(description="Note to save")) -> str:
            notes = self.state.get("notes", [])
            notes.append(note)
            await self.update_state(notes=notes)
            await self.emit("note_saved", {"note": note, "count": len(notes)})
            return f"Saved note #{len(notes)}"

        super().__init__(
            "chat-room",
            model=google("gemini-2.0-flash"),
            system="You are a helpful AI assistant. Be concise.",
            storage=MemoryStorage(),
            initial_state={"notes": [], "message_count": 0},
            tools={"save_note": save_note},
        )

    async def on_start(self):
        print(f"Chat room started. Notes: {len(self.state.get('notes', []))}")


# ─── FastAPI App ────────────────────────────────────────────────────────

room = ChatRoom()

# Queue for SSE connections
sse_queues: list[asyncio.Queue] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start agent on app startup."""
    await room.start()

    # Inject emit handler to broadcast to SSE queues
    async def emit_to_sse(event: str, data: dict, event_id: int):
        sse_msg = f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data)}\n\n"
        for queue in sse_queues:
            await queue.put(sse_msg)

    room._emit_handler = emit_to_sse

    print("Chat room ready")
    yield
    await room.stop()


app = FastAPI(lifespan=lifespan)

# Add CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Send a message and get a response. Events stream via SSE."""
    # Emit start event
    await room.emit("chat_start", {"message": request.message})

    # Update message count
    await room.update_state(message_count=room.state.get("message_count", 0) + 1)

    # Stream response, emitting chunks
    full_response = ""
    async for chunk in room.stream(request.message):
        full_response += chunk
        await room.emit("chunk", {"content": chunk})

    # Emit completion
    await room.emit("chat_complete", {"response": full_response})

    return {"response": full_response}


@app.get("/events")
async def sse_endpoint():
    """SSE endpoint for real-time events."""

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        sse_queues.append(queue)

        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30)
                    yield message
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            sse_queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/state")
async def state_endpoint():
    """Get current agent state."""
    return {"state": room.state}


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print("SSE:   http://localhost:8000/events")
    print("Chat:  POST http://localhost:8000/chat")
    print("State: GET http://localhost:8000/state")
    uvicorn.run(app, host="localhost", port=8000)

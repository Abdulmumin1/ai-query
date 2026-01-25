from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web

from ai_query.agents import (
    Agent,
    action,
    AgentServer,
    MemoryStorage,
    ConnectionContext,
    Connection,
)


class ChatRoom(Agent):
    enable_event_log = True
    def __init__(self, id: str, **kwargs: Any):
        super().__init__(
            id,
            storage=MemoryStorage(),
            initial_state={"messages": [], "reputation": {}},
            **kwargs
        )
    

    async def on_connect(self, connection: Connection, ctx: ConnectionContext):
        """Set user_id and notify room of arrival."""
        user_id = ctx.request.headers.get("X-User-Id", "anonymous")
        connection.state["user_id"] = user_id
        await self.emit("status", {"text": f"{user_id} has joined."})
        await super().on_connect(connection, ctx)

    async def on_message(self, connection: Connection, message: str | bytes | dict[str, Any]):
        """Broadcast incoming chat messages to all users."""
        if isinstance(message, (bytes, str)):
            text = message.decode("utf-8") if isinstance(message, bytes) else message
        elif isinstance(message, dict):
            text = message.get("text", "")
        else:
            return

        user_id = connection.state.get("user_id", "anonymous")
        message_data = {"user_id": user_id, "text": text}
        
        # Add to history
        self.state["messages"].append(message_data)
        
        # Broadcast to all clients
        await self.emit("message", message_data)

    @action
    async def add_reaction(self, message_id: str, reaction: str) -> dict[str, Any]:
        """Add a reaction to a message."""
        assert self.context.connection is not None
        user_id = self.context.connection.state.get("user_id", "anonymous")
        await self.emit("reaction", {"user_id": user_id, "reaction": reaction})
        return {"status": "ok", "user": user_id}

    @action
    async def get_history(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Get the chat history, optionally filtering by user_id."""
        if user_id:
            return [
                msg for msg in self.state["messages"] if msg.get("user_id") == user_id
            ]
        return self.state["messages"]

    @action
    async def greet(self, name: str) -> str:
        """A simple greeting action."""
        return f"Hello, {name}!"


if __name__ == "__main__":
    server = AgentServer(ChatRoom)
    server.serve(port=8080)

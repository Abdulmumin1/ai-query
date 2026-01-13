"""SQLite-based persistent storage agent using aiosqlite for async operations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generic, TypeVar

import aiosqlite

from ai_query.agents.base import Agent
from ai_query.types import Message, AgentEvent

State = TypeVar("State")


class SQLiteAgent(Agent[State], Generic[State]):
    """
    Agent with async SQLite persistence.

    Provides persistent storage for state and messages using aiosqlite for
    non-blocking database operations. Access to an embedded SQLite database
    is available via the sql() method.

    Attributes:
        db_path: Path to the SQLite database file. Override in subclass
                 or set ":memory:" for in-memory database. Default: "agents.db"

    Example:
        class MyBot(ChatAgent, SQLiteAgent):
            db_path = "./data/my_bot.db"
            initial_state = {"user_prefs": {}}

        async with MyBot("bot-123") as bot:
            # Custom SQL queries
            await bot.sql("CREATE TABLE IF NOT EXISTS logs (msg TEXT)")
            await bot.sql("INSERT INTO logs VALUES (?)", "Hello")
    """

    db_path: str = "agents.db"

    def __init__(self, agent_id: str, *, env: Any = None, db_path: str | None = None):
        """
        Initialize the SQLite agent.

        Args:
            agent_id: Unique identifier for this agent.
            env: Optional environment bindings.
            db_path: Override the database path (optional).
        """
        super().__init__(agent_id, env=env)

        if db_path is not None:
            self.db_path = db_path

        # Ensure parent directory exists
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn: aiosqlite.Connection | None = None

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Ensure database connection is open."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._init_schema()
        return self._conn

    async def _init_schema(self) -> None:
        """Initialize the agent tables if they don't exist."""
        if self._conn is None:
            return
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                messages TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_events (
                agent_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (agent_id, event_id)
            )
        """)
        await self._conn.commit()

    async def sql(self, query: str, *params: Any) -> list[dict[str, Any]]:
        """
        Execute a SQL query against the agent's database.

        Args:
            query: The SQL query to execute.
            *params: Query parameters for safe substitution.

        Returns:
            List of rows as dictionaries.

        Example:
            # Create a custom table
            await agent.sql("CREATE TABLE IF NOT EXISTS users (id TEXT, name TEXT)")

            # Insert data
            await agent.sql("INSERT INTO users VALUES (?, ?)", "1", "Alice")

            # Query data
            users = await agent.sql("SELECT * FROM users WHERE name LIKE ?", "%Ali%")
        """
        conn = await self._ensure_connection()
        cursor = await conn.execute(query, params)
        await conn.commit()

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = await cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        return []

    async def _load_state(self) -> State | None:
        """Load state from SQLite."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            "SELECT state FROM agent_state WHERE id = ?",
            (self._id,)
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def _save_state(self, state: State) -> None:
        """Save state to SQLite."""
        conn = await self._ensure_connection()
        await conn.execute(
            "INSERT OR REPLACE INTO agent_state (id, state) VALUES (?, ?)",
            (self._id, json.dumps(state))
        )
        await conn.commit()

    async def _load_messages(self) -> list[Message]:
        """Load messages from SQLite."""
        conn = await self._ensure_connection()
        cursor = await conn.execute(
            "SELECT messages FROM agent_messages WHERE id = ?",
            (self._id,)
        )
        row = await cursor.fetchone()
        if row:
            messages_data = json.loads(row[0])
            return [Message(**msg) for msg in messages_data]
        return []

    async def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to SQLite."""
        conn = await self._ensure_connection()
        messages_data = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        await conn.execute(
            "INSERT OR REPLACE INTO agent_messages (id, messages) VALUES (?, ?)",
            (self._id, json.dumps(messages_data))
        )
        await conn.commit()

    async def _save_event(self, event: AgentEvent) -> None:
        """Save an event to the persistent log."""
        conn = await self._ensure_connection()

        # Prune expired events (lazy cleanup)
        if self.event_retention > 0:
            cutoff = time.time() - self.event_retention
            await conn.execute(
                "DELETE FROM agent_events WHERE agent_id = ? AND created_at < ?",
                (self._id, cutoff)
            )

        await conn.execute(
            """
            INSERT INTO agent_events (agent_id, event_id, event_type, data, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self._id, event.id, event.type, json.dumps(event.data), event.created_at)
        )
        await conn.commit()

    async def _get_events(self, after_id: int | None = None) -> list[AgentEvent]:
        """Get events from the persistent log."""
        conn = await self._ensure_connection()
        query = "SELECT * FROM agent_events WHERE agent_id = ?"
        params = [self._id]

        if after_id is not None:
            query += " AND event_id > ?"
            params.append(after_id)

        query += " ORDER BY event_id ASC"

        cursor = await conn.execute(query, tuple(params))
        rows = await cursor.fetchall()

        events = []
        for row in rows:
            events.append(AgentEvent(
                id=row["event_id"],
                type=row["event_type"],
                data=json.loads(row["data"]),
                created_at=row["created_at"]
            ))
        return events

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aexit__(self, *args: Any) -> None:
        """Close connections on exit."""
        await super().__aexit__(*args)
        await self.close()

"""SQLite-based persistent storage agent."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Generic, TypeVar

from ai_query.agents.base import Agent
from ai_query.types import Message

State = TypeVar("State")


class SQLiteAgent(Agent[State], Generic[State]):
    """
    Agent with SQLite persistence.
    
    Provides persistent storage for state and messages, plus access to
    an embedded SQLite database via the sql() method.
    
    Attributes:
        db_path: Path to the SQLite database file. Override in subclass
                 or set ":memory:" for in-memory database. Default: "agents.db"
    
    Example:
        class MyBot(ChatAgent, SQLiteAgent):
            db_path = "./data/my_bot.db"
            initial_state = {"user_prefs": {}}
        
        async with MyBot("bot-123") as bot:
            # Custom SQL queries
            bot.sql("CREATE TABLE IF NOT EXISTS logs (msg TEXT)")
            bot.sql("INSERT INTO logs VALUES (?)", "Hello")
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
        
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize the agent tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                id TEXT PRIMARY KEY,
                state TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                messages TEXT NOT NULL
            )
        """)
        self._conn.commit()
    
    def sql(self, query: str, *params: Any) -> list[dict[str, Any]]:
        """
        Execute a SQL query against the agent's database.
        
        Args:
            query: The SQL query to execute.
            *params: Query parameters for safe substitution.
            
        Returns:
            List of rows as dictionaries.
            
        Example:
            # Create a custom table
            agent.sql("CREATE TABLE IF NOT EXISTS users (id TEXT, name TEXT)")
            
            # Insert data
            agent.sql("INSERT INTO users VALUES (?, ?)", "1", "Alice")
            
            # Query data
            users = agent.sql("SELECT * FROM users WHERE name LIKE ?", "%Ali%")
        """
        cursor = self._conn.execute(query, params)
        self._conn.commit()
        
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        return []
    
    async def _load_state(self) -> State | None:
        """Load state from SQLite."""
        cursor = self._conn.execute(
            "SELECT state FROM agent_state WHERE id = ?",
            (self._id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    async def _save_state(self, state: State) -> None:
        """Save state to SQLite."""
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_state (id, state) VALUES (?, ?)",
            (self._id, json.dumps(state))
        )
        self._conn.commit()
    
    async def _load_messages(self) -> list[Message]:
        """Load messages from SQLite."""
        cursor = self._conn.execute(
            "SELECT messages FROM agent_messages WHERE id = ?",
            (self._id,)
        )
        row = cursor.fetchone()
        if row:
            messages_data = json.loads(row[0])
            return [Message(**msg) for msg in messages_data]
        return []
    
    async def _save_messages(self, messages: list[Message]) -> None:
        """Save messages to SQLite."""
        messages_data = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_messages (id, messages) VALUES (?, ?)",
            (self._id, json.dumps(messages_data))
        )
        self._conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
    
    async def __aexit__(self, *args: Any) -> None:
        """Close connections on exit."""
        await super().__aexit__(*args)
        self.close()

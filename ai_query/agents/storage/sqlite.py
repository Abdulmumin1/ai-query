"""SQLite storage implementation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStorage:
    """SQLite-based persistent storage.

    Stores data in a SQLite database file for persistence across restarts.

    Example:
        agent = Agent("assistant", storage=SQLiteStorage("agents.db"))
    """

    def __init__(self, path: str = "agents.db") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        return self._conn

    async def get(self, key: str) -> Any | None:
        conn = self._get_connection()
        cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def set(self, key: str, value: Any) -> None:
        conn = self._get_connection()
        json_value = json.dumps(value, default=str)
        conn.execute(
            "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
            (key, json_value),
        )
        conn.commit()

    async def delete(self, key: str) -> None:
        conn = self._get_connection()
        conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        conn.commit()

    async def keys(self, prefix: str = "") -> list[str]:
        conn = self._get_connection()
        if not prefix:
            cursor = conn.execute("SELECT key FROM kv_store")
        else:
            cursor = conn.execute(
                "SELECT key FROM kv_store WHERE key LIKE ?",
                (f"{prefix}%",),
            )
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                key TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (key, event_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_log_meta (
                key TEXT PRIMARY KEY,
                event_counter INTEGER NOT NULL
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

    async def append_event(
        self,
        key: str,
        event: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> None:
        conn = self._get_connection()
        event_id = int(event["id"])
        value = json.dumps(event, default=str)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO event_log (key, event_id, value) VALUES (?, ?, ?)",
                (key, event_id, value),
            )
            self._set_event_counter(conn, key, event_id)
            self._prune_events(conn, key, limit)

    async def set_event_counter(self, key: str, event_id: int) -> None:
        conn = self._get_connection()
        with conn:
            self._set_event_counter(conn, key, event_id)

    async def load_events(
        self,
        key: str,
        *,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        conn = self._get_connection()
        self._migrate_legacy_event_log(conn, key, limit)
        with conn:
            self._prune_events(conn, key, limit)
        rows = conn.execute(
            "SELECT value FROM event_log WHERE key = ? ORDER BY event_id",
            (key,),
        ).fetchall()
        counter_row = conn.execute(
            "SELECT event_counter FROM event_log_meta WHERE key = ?",
            (key,),
        ).fetchone()
        events = [json.loads(row[0]) for row in rows]
        counter = int(counter_row[0]) if counter_row else 0
        if events:
            counter = max(counter, max(int(event.get("id", 0)) for event in events))
        return events, counter

    async def delete_events(self, key: str) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM event_log WHERE key = ?", (key,))
            conn.execute("DELETE FROM event_log_meta WHERE key = ?", (key,))
            conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))

    @staticmethod
    def _set_event_counter(
        conn: sqlite3.Connection,
        key: str,
        event_id: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO event_log_meta (key, event_counter) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                event_counter = MAX(event_log_meta.event_counter, excluded.event_counter)
            """,
            (key, event_id),
        )

    @staticmethod
    def _prune_events(
        conn: sqlite3.Connection,
        key: str,
        limit: int | None,
    ) -> None:
        if limit is None:
            return
        if limit <= 0:
            conn.execute("DELETE FROM event_log WHERE key = ?", (key,))
            return
        threshold = conn.execute(
            """
            SELECT event_id
            FROM event_log
            WHERE key = ?
            ORDER BY event_id DESC
            LIMIT 1 OFFSET ?
            """,
            (key, limit - 1),
        ).fetchone()
        if threshold is not None:
            conn.execute(
                "DELETE FROM event_log WHERE key = ? AND event_id < ?",
                (key, int(threshold[0])),
            )

    def _migrate_legacy_event_log(
        self,
        conn: sqlite3.Connection,
        key: str,
        limit: int | None,
    ) -> None:
        legacy = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?",
            (key,),
        ).fetchone()
        if legacy is None:
            return

        try:
            events = json.loads(legacy[0])
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(events, list):
            return

        retained = events if limit is None else events[-max(0, limit):]
        max_event_id = 0
        with conn:
            for event in retained:
                if not isinstance(event, dict):
                    continue
                event_id = event.get("id")
                if not isinstance(event_id, int):
                    continue
                max_event_id = max(max_event_id, event_id)
                conn.execute(
                    "INSERT OR REPLACE INTO event_log (key, event_id, value) VALUES (?, ?, ?)",
                    (key, event_id, json.dumps(event, default=str)),
                )
            for event in events:
                if isinstance(event, dict) and isinstance(event.get("id"), int):
                    max_event_id = max(max_event_id, int(event["id"]))
            if max_event_id:
                self._set_event_counter(conn, key, max_event_id)
            conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

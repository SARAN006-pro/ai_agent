"""
SQLite-backed memory store for agent task results.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent / "agent.db"
_DB_LOCK = threading.RLock()
_CONN: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    """Return a shared thread-safe SQLite connection."""
    global _CONN
    with _DB_LOCK:
        if _CONN is None:
            _CONN = sqlite3.connect(_DB_PATH, check_same_thread=False)
            _CONN.row_factory = sqlite3.Row
            _CONN.execute("PRAGMA journal_mode=WAL;")
        return _CONN


def init_db() -> None:
    """Initialize database schema if it does not already exist."""
    conn = _get_connection()
    with _DB_LOCK:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_memory(task: str, result: str) -> None:
    """Persist one task and its result."""
    conn = _get_connection()
    with _DB_LOCK:
        conn.execute(
            "INSERT INTO logs (task, result) VALUES (?, ?)",
            (task, result),
        )
        conn.commit()


def get_recent_memories(limit: int = 5) -> list[dict[str, Any]]:
    """Return the latest saved memories ordered from newest to oldest."""
    if limit <= 0:
        return []

    conn = _get_connection()
    with _DB_LOCK:
        rows = conn.execute(
            """
            SELECT id, task, result, created_at
            FROM logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [dict(row) for row in rows]

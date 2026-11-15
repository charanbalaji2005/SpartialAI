"""SQLite project metadata (local-first, no server needed)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,           -- created | uploaded | queued | running | completed | failed | cancelled
    stage TEXT,
    progress REAL DEFAULT 0,
    message TEXT,
    error TEXT,
    original_filename TEXT,
    image_ext TEXT,
    image_width INTEGER,
    image_height INTEGER,
    quality TEXT DEFAULT 'balanced',
    scene_category TEXT,
    thumbnail TEXT,
    report TEXT
);
"""

JSON_FIELDS = {"report"}


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def create(self, pid: str, name: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO projects (id, name, created_at, updated_at, status, progress) "
                "VALUES (?, ?, ?, ?, 'created', 0)", (pid, name, now, now))
            self._conn.commit()
        return self.get(pid)  # type: ignore[return-value]

    def update(self, pid: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        for key in JSON_FIELDS & fields.keys():
            fields[key] = json.dumps(fields[key])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE projects SET {cols} WHERE id = ?", (*fields.values(), pid))
            self._conn.commit()

    def get(self, pid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
        return _row(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [_row(r) for r in rows]

    def delete(self, pid: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
            self._conn.commit()

    def mark_interrupted(self) -> int:
        """Jobs that were running when the server stopped can't resume mid-model."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE projects SET status = 'failed', error = 'Interrupted: the server stopped "
                "during reconstruction. Generate again to retry.' WHERE status IN ('running', 'queued')")
            self._conn.commit()
            return cur.rowcount


def _row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in JSON_FIELDS:
        if data.get(key):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                data[key] = None
    return data

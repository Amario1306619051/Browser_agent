"""Per-thread short-term memory: the agent remembers earlier tasks (and their
results) within a thread_id, so follow-up tasks keep context ("add IT to cart"
resolves to what the previous task found).

Default store: SQLite (zero-setup) at config.MEMORY_DB_PATH. Set
AGENT_DATABASE_URL=postgresql://user:pass@host/db to use Postgres (needs the
`psycopg` package). Same idea as LangGraph's thread_id checkpointer, kept native
so we don't pull a framework into a vanilla codebase.

If the store can't be opened (e.g. bad Postgres URL), memory is disabled and the
agent still runs — every method becomes a no-op.
"""
from __future__ import annotations

import logging
import threading
import time

import config

log = logging.getLogger(__name__)


class Memory:
    def __init__(self) -> None:
        self.url = (config.DATABASE_URL or "").strip()
        self.is_pg = self.url.lower().startswith(("postgres://", "postgresql://"))
        self._lock = threading.Lock()
        self._conn = None
        self.ok = False
        try:
            self._connect()
            self._init()
            self.ok = True
            log.info("Memory store ready (%s)", "postgres" if self.is_pg else "sqlite")
        except Exception as e:  # noqa: BLE001 — never block the app on memory
            log.warning("Memory disabled (%s): %s", "postgres" if self.is_pg else "sqlite", e)

    # ---- backend plumbing ----------------------------------------------------
    def _connect(self) -> None:
        if self.is_pg:
            import psycopg  # lazy: only needed for the Postgres path

            self._conn = psycopg.connect(self.url, autocommit=True)
        else:
            import sqlite3

            config.MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(config.MEMORY_DB_PATH), check_same_thread=False)

    @property
    def _ph(self) -> str:
        return "%s" if self.is_pg else "?"

    def _commit(self) -> None:
        if not self.is_pg:
            self._conn.commit()

    def _init(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            if self.is_pg:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS thread_memory ("
                    "id SERIAL PRIMARY KEY, thread_id TEXT, ts DOUBLE PRECISION, "
                    "task TEXT, result TEXT)"
                )
            else:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS thread_memory ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT, ts REAL, "
                    "task TEXT, result TEXT)"
                )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_thread_memory_tid ON thread_memory(thread_id)")
            self._commit()

    @staticmethod
    def norm(thread_id: str | None) -> str:
        return (thread_id or "").strip()[:200]

    # ---- public API ----------------------------------------------------------
    def count(self, thread_id: str | None) -> int:
        tid = self.norm(thread_id)
        if not self.ok or not tid:
            return 0
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM thread_memory WHERE thread_id={self._ph}", (tid,))
                return int(cur.fetchone()[0])
        except Exception as e:  # noqa: BLE001
            log.warning("memory.count failed: %s", e)
            return 0

    def load(self, thread_id: str | None, limit: int | None = None) -> list[dict]:
        """Most recent `limit` turns for the thread, oldest → newest."""
        tid = self.norm(thread_id)
        if not self.ok or not tid:
            return []
        limit = int(limit or config.MEMORY_TURNS)
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    f"SELECT task, result, ts FROM thread_memory WHERE thread_id={self._ph} "
                    f"ORDER BY id DESC LIMIT {self._ph}",
                    (tid, limit),
                )
                rows = cur.fetchall()
            rows.reverse()
            return [{"task": r[0], "result": r[1], "ts": r[2]} for r in rows]
        except Exception as e:  # noqa: BLE001
            log.warning("memory.load failed: %s", e)
            return []

    def append(self, thread_id: str | None, task: str, result: str) -> None:
        tid = self.norm(thread_id)
        if not self.ok or not tid:
            return
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    f"INSERT INTO thread_memory(thread_id, ts, task, result) "
                    f"VALUES ({self._ph},{self._ph},{self._ph},{self._ph})",
                    (tid, time.time(), str(task)[:4000], str(result)[:8000]),
                )
                self._commit()
        except Exception as e:  # noqa: BLE001
            log.warning("memory.append failed: %s", e)


# Singleton used across the app.
store = Memory()

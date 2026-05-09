from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from .sqlite import SQLiteMemoryStore


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value):
    return json.dumps({} if value is None else value, sort_keys=True)


class AsyncSQLiteMemoryStore:
    """Async SQLite access layer for concurrent Robits runtimes and observers.

    This class intentionally starts as a small async boundary around the hot
    event/message paths that the runtime and TUI need first. The synchronous
    ``SQLiteMemoryStore`` remains the canonical schema owner while this async
    layer grows method coverage.
    """

    def __init__(self, path, connection):
        self.path = Path(path)
        self.connection = connection
        self._db_lock = asyncio.Lock()

    @classmethod
    async def open(cls, path, *, busy_timeout_ms=5000, wal=True):
        db_path = Path(path)
        if db_path == Path(":memory:"):
            raise ValueError(
                "AsyncSQLiteMemoryStore requires a file-backed SQLite database; "
                "':memory:' would create separate databases per connection."
            )

        # Keep schema creation and migration centralized in the synchronous store
        # until this async layer fully owns schema management.
        bootstrap = SQLiteMemoryStore(db_path)
        bootstrap.close()

        connection = await aiosqlite.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        if wal:
            await connection.execute("PRAGMA journal_mode = WAL")
        await connection.commit()
        return cls(db_path, connection)

    async def close(self):
        await self.connection.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def _rows(self, statement, params=()):
        async with self._db_lock:
            cursor = await self.connection.execute(statement, params)
            try:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                await cursor.close()

    async def _index(
        self,
        kind,
        record_id,
        agent_id,
        related_agent_id,
        session_id,
        source,
        relationship_type,
        conversation_type,
        created_at,
        content,
    ):
        await self.connection.execute(
            """
            INSERT INTO memory_fts(
                kind, record_id, agent_id, related_agent_id, session_id,
                source, relationship_type, conversation_type, created_at, content
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                str(record_id),
                agent_id,
                related_agent_id,
                session_id,
                source,
                relationship_type,
                conversation_type,
                created_at,
                content,
            ),
        )

    async def create_session(self, session_id, title=None, started_at=None, metadata=None):
        timestamp = started_at or _utc_now()
        async with self._db_lock:
            await self.connection.execute(
                """
                INSERT OR IGNORE INTO sessions(session_id, title, started_at, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, title, timestamp, _json_dumps(metadata)),
            )
            await self.connection.commit()
        return session_id

    async def upsert_agent(
        self,
        agent_id,
        role,
        display_name=None,
        lifecycle_state="active",
        created_at=None,
        metadata=None,
    ):
        timestamp = created_at or _utc_now()
        async with self._db_lock:
            await self.connection.execute(
                """
                INSERT INTO agents(
                    agent_id, role, display_name, lifecycle_state, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    role = excluded.role,
                    display_name = excluded.display_name,
                    lifecycle_state = excluded.lifecycle_state,
                    metadata_json = excluded.metadata_json
                """,
                (
                    agent_id,
                    role,
                    display_name,
                    lifecycle_state,
                    timestamp,
                    _json_dumps(metadata),
                ),
            )
            await self.connection.commit()
        return agent_id

    async def append_message(
        self,
        session_id,
        sender_agent_id,
        receiver_agent_id,
        content,
        kind="message",
        visibility="public",
        relationship_type=None,
        conversation_type=None,
        source=None,
        created_at=None,
        metadata=None,
    ):
        timestamp = created_at or _utc_now()
        async with self._db_lock:
            cursor = await self.connection.execute(
                """
                INSERT INTO messages(
                    session_id, sender_agent_id, receiver_agent_id, content, kind,
                    visibility, relationship_type, conversation_type, source,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    sender_agent_id,
                    receiver_agent_id,
                    content,
                    kind,
                    visibility,
                    relationship_type,
                    conversation_type,
                    source,
                    timestamp,
                    _json_dumps(metadata),
                ),
            )
            record_id = cursor.lastrowid
            await cursor.close()
            await self._index(
                "message",
                record_id,
                sender_agent_id,
                receiver_agent_id,
                session_id,
                source,
                relationship_type,
                conversation_type,
                timestamp,
                content,
            )
            await self.connection.commit()
        return record_id

    async def append_thought(
        self,
        agent_id,
        content,
        session_id=None,
        visibility="private",
        relationship_type=None,
        conversation_type=None,
        source=None,
        created_at=None,
        metadata=None,
    ):
        timestamp = created_at or _utc_now()
        async with self._db_lock:
            cursor = await self.connection.execute(
                """
                INSERT INTO thoughts(
                    session_id, agent_id, content, visibility, relationship_type,
                    conversation_type, source, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    agent_id,
                    content,
                    visibility,
                    relationship_type,
                    conversation_type,
                    source,
                    timestamp,
                    _json_dumps(metadata),
                ),
            )
            record_id = cursor.lastrowid
            await cursor.close()
            await self._index(
                "thought",
                record_id,
                agent_id,
                None,
                session_id,
                source,
                relationship_type,
                conversation_type,
                timestamp,
                content,
            )
            await self.connection.commit()
        return record_id

    async def append_runtime_event(
        self,
        session_id,
        event_type,
        payload=None,
        visibility="public",
        sequence=None,
        created_at=None,
    ):
        timestamp = created_at or _utc_now()
        async with self._db_lock:
            cursor = await self.connection.execute(
                """
                INSERT INTO runtime_events(
                    session_id, sequence, event_type, visibility, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    sequence,
                    event_type,
                    visibility,
                    _json_dumps(payload),
                    timestamp,
                ),
            )
            record_id = cursor.lastrowid
            await cursor.close()
            await self.connection.commit()
        return record_id

    async def list_messages(
        self,
        session_id=None,
        agent_id=None,
        conversation_type=None,
        visibility=None,
        limit=100,
    ):
        clauses = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if agent_id is not None:
            clauses.append("(sender_agent_id = ? OR receiver_agent_id = ?)")
            params.extend([agent_id, agent_id])
        if conversation_type is not None:
            clauses.append("conversation_type = ?")
            params.append(conversation_type)
        if visibility is not None:
            clauses.append("visibility = ?")
            params.append(visibility)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return await self._rows(
            f"""
            SELECT * FROM messages
            {where}
            ORDER BY created_at, message_id
            LIMIT ?
            """,
            params + [limit],
        )

    async def list_thoughts(
        self,
        session_id=None,
        agent_id=None,
        conversation_type=None,
        visibility=None,
        limit=100,
    ):
        clauses = []
        params: list[Any] = []
        for column, value in (
            ("session_id", session_id),
            ("agent_id", agent_id),
            ("conversation_type", conversation_type),
            ("visibility", visibility),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return await self._rows(
            f"""
            SELECT * FROM thoughts
            {where}
            ORDER BY created_at, thought_id
            LIMIT ?
            """,
            params + [limit],
        )

    async def list_runtime_events(
        self,
        session_id=None,
        event_type=None,
        visibility=None,
        limit=100,
    ):
        clauses = []
        params: list[Any] = []
        for column, value in (
            ("session_id", session_id),
            ("event_type", event_type),
            ("visibility", visibility),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return await self._rows(
            f"""
            SELECT * FROM runtime_events
            {where}
            ORDER BY COALESCE(sequence, event_id), event_id
            LIMIT ?
            """,
            params + [limit],
        )

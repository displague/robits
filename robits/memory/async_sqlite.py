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

    The synchronous ``SQLiteMemoryStore`` remains the schema owner, and this
    class mirrors its repository API for async callers. Hot-path methods are
    implemented natively with ``aiosqlite`` while remaining methods delegate to
    the synchronous store through ``asyncio.to_thread``.
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

    async def _channel_type(self, channel_id):
        if channel_id is None:
            return None
        cursor = await self.connection.execute(
            "SELECT channel_type FROM channels WHERE channel_id = ?", (channel_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row)["channel_type"] if row else None

    async def _run_sync(self, method_name, *args, **kwargs):
        def runner():
            store = SQLiteMemoryStore(self.path)
            try:
                return getattr(store, method_name)(*args, **kwargs)
            finally:
                store.close()

        return await asyncio.to_thread(runner)

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
        channel_id=None,
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
                    visibility, relationship_type, channel_id, source,
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
                    channel_id,
                    source,
                    timestamp,
                    _json_dumps(metadata),
                ),
            )
            record_id = cursor.lastrowid
            await cursor.close()
            channel_type = await self._channel_type(channel_id)
            await self._index(
                "message",
                record_id,
                sender_agent_id,
                receiver_agent_id,
                session_id,
                source,
                relationship_type,
                channel_type,
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
        channel_id=None,
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
                    channel_id, source, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    agent_id,
                    content,
                    visibility,
                    relationship_type,
                    channel_id,
                    source,
                    timestamp,
                    _json_dumps(metadata),
                ),
            )
            record_id = cursor.lastrowid
            await cursor.close()
            channel_type = await self._channel_type(channel_id)
            await self._index(
                "thought",
                record_id,
                agent_id,
                None,
                session_id,
                source,
                relationship_type,
                channel_type,
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
        channel_id=None,
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
        if channel_id is not None:
            clauses.append("channel_id = ?")
            params.append(channel_id)
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
        channel_id=None,
        visibility=None,
        limit=100,
    ):
        clauses = []
        params: list[Any] = []
        for column, value in (
            ("session_id", session_id),
            ("agent_id", agent_id),
            ("channel_id", channel_id),
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

    async def ensure_schema(self):
        return await self._run_sync("ensure_schema")

    async def list_recent_message_ids(self, session_id, limit):
        return await self._run_sync("list_recent_message_ids", session_id, limit)

    async def list_recent_message_ids_by_channel(self, session_id, limit, channel_id):
        return await self._run_sync(
            "list_recent_message_ids_by_channel",
            session_id,
            limit,
            channel_id,
        )

    async def end_session(self, session_id, ended_at=None):
        return await self._run_sync("end_session", session_id, ended_at=ended_at)

    async def add_contact(
        self,
        owner_agent_id,
        contact_agent_id,
        relationship_type,
        notes=None,
        created_at=None,
        metadata=None,
    ):
        return await self._run_sync(
            "add_contact",
            owner_agent_id,
            contact_agent_id,
            relationship_type,
            notes=notes,
            created_at=created_at,
            metadata=metadata,
        )

    async def append_todo(
        self,
        agent_id,
        title,
        session_id=None,
        status="open",
        content=None,
        due_at=None,
        created_at=None,
        metadata=None,
    ):
        return await self._run_sync(
            "append_todo",
            agent_id,
            title,
            session_id=session_id,
            status=status,
            content=content,
            due_at=due_at,
            created_at=created_at,
            metadata=metadata,
        )

    async def append_tool_call(
        self,
        tool_call_id,
        agent_id,
        tool_name,
        arguments=None,
        result_content=None,
        status="requested",
        session_id=None,
        relationship_type=None,
        conversation_type=None,
        source=None,
        created_at=None,
        metadata=None,
    ):
        return await self._run_sync(
            "append_tool_call",
            tool_call_id,
            agent_id,
            tool_name,
            arguments=arguments,
            result_content=result_content,
            status=status,
            session_id=session_id,
            relationship_type=relationship_type,
            conversation_type=conversation_type,
            source=source,
            created_at=created_at,
            metadata=metadata,
        )

    async def list_todos(self, agent_id=None, session_id=None, status=None, limit=100):
        return await self._run_sync(
            "list_todos",
            agent_id=agent_id,
            session_id=session_id,
            status=status,
            limit=limit,
        )

    async def append_memory_entry(
        self,
        kind,
        content,
        agent_id=None,
        session_id=None,
        source_table=None,
        source_id=None,
        relationship_type=None,
        conversation_type=None,
        source=None,
        created_at=None,
        metadata=None,
    ):
        return await self._run_sync(
            "append_memory_entry",
            kind,
            content,
            agent_id=agent_id,
            session_id=session_id,
            source_table=source_table,
            source_id=source_id,
            relationship_type=relationship_type,
            conversation_type=conversation_type,
            source=source,
            created_at=created_at,
            metadata=metadata,
        )

    async def append_memory_digest(
        self,
        content,
        source_refs,
        agent_id=None,
        session_id=None,
        digest_type="episodic",
        generation=None,
        version=1,
        supersedes_digest_ids=None,
        system_only=False,
        accessibility=None,
        prompt_version="memory-digest-v1",
        source_start_at=None,
        source_end_at=None,
        relationship_type=None,
        conversation_type=None,
        source="digest",
        created_at=None,
        metadata=None,
    ):
        return await self._run_sync(
            "append_memory_digest",
            content,
            source_refs,
            agent_id=agent_id,
            session_id=session_id,
            digest_type=digest_type,
            generation=generation,
            version=version,
            supersedes_digest_ids=supersedes_digest_ids,
            system_only=system_only,
            accessibility=accessibility,
            prompt_version=prompt_version,
            source_start_at=source_start_at,
            source_end_at=source_end_at,
            relationship_type=relationship_type,
            conversation_type=conversation_type,
            source=source,
            created_at=created_at,
            metadata=metadata,
        )

    async def seed_memory_digest(
        self,
        digest_type,
        content,
        agent_id=None,
        session_id=None,
        prompt_version="memory-digest-seed-v1",
        source="system_seed",
        created_at=None,
        metadata=None,
        system_only=False,
        accessibility=None,
        relationship_type=None,
    ):
        return await self._run_sync(
            "seed_memory_digest",
            digest_type,
            content,
            agent_id=agent_id,
            session_id=session_id,
            prompt_version=prompt_version,
            source=source,
            created_at=created_at,
            metadata=metadata,
            system_only=system_only,
            accessibility=accessibility,
            relationship_type=relationship_type,
        )

    async def seed_identity_and_goal_digests(
        self,
        agent_id,
        identity_content,
        long_term_goal_content,
        short_term_goal_content=None,
        session_id=None,
        created_at=None,
        metadata=None,
        identity_relationship_type=None,
    ):
        return await self._run_sync(
            "seed_identity_and_goal_digests",
            agent_id,
            identity_content,
            long_term_goal_content,
            short_term_goal_content=short_term_goal_content,
            session_id=session_id,
            created_at=created_at,
            metadata=metadata,
            identity_relationship_type=identity_relationship_type,
        )

    async def append_runtime_event_object(self, event):
        return await self._run_sync("append_runtime_event_object", event)

    async def get_memory_digest(self, digest_id):
        return await self._run_sync("get_memory_digest", digest_id)

    async def get_memory_digest_sources(self, digest_id):
        return await self._run_sync("get_memory_digest_sources", digest_id)

    async def expand_memory_digest_sources(
        self,
        digest_id,
        recursive=False,
        include_digest_records=True,
        max_depth=None,
    ):
        return await self._run_sync(
            "expand_memory_digest_sources",
            digest_id,
            recursive=recursive,
            include_digest_records=include_digest_records,
            max_depth=max_depth,
        )

    async def expand_memory_digest_source_tree(self, digest_id):
        return await self._run_sync("expand_memory_digest_source_tree", digest_id)

    async def list_memory_digests(
        self,
        agent_id=None,
        session_id=None,
        digest_type=None,
        current_only=False,
        accessible_only=False,
        include_system_only=True,
        limit=100,
        relationship_type=None,
    ):
        return await self._run_sync(
            "list_memory_digests",
            agent_id=agent_id,
            session_id=session_id,
            digest_type=digest_type,
            current_only=current_only,
            accessible_only=accessible_only,
            include_system_only=include_system_only,
            limit=limit,
            relationship_type=relationship_type,
        )

    async def list_agent_records(self, agent_id, limit=100):
        return await self._run_sync("list_agent_records", agent_id, limit=limit)

    async def search(
        self,
        query,
        agent_id=None,
        session_id=None,
        relationship_type=None,
        conversation_type=None,
        source=None,
        start_at=None,
        end_at=None,
        limit=20,
    ):
        return await self._run_sync(
            "search",
            query,
            agent_id=agent_id,
            session_id=session_id,
            relationship_type=relationship_type,
            conversation_type=conversation_type,
            source=source,
            start_at=start_at,
            end_at=end_at,
            limit=limit,
        )

    async def search_cascade(
        self,
        query,
        agent_id=None,
        session_id=None,
        relationship_type=None,
        conversation_type=None,
        source=None,
        start_at=None,
        end_at=None,
        limit=20,
    ):
        return await self._run_sync(
            "search_cascade",
            query,
            agent_id=agent_id,
            session_id=session_id,
            relationship_type=relationship_type,
            conversation_type=conversation_type,
            source=source,
            start_at=start_at,
            end_at=end_at,
            limit=limit,
        )

    async def get_or_create_channel(
        self,
        channel_type,
        participants=None,
        *,
        visibility="public",
        social_distance=None,
        clock_phase_hint=None,
        memory_policy="default",
        digest_policy="default",
        prompt_injection_policy="default",
        metadata=None,
        created_at=None,
    ):
        return await self._run_sync(
            "get_or_create_channel",
            channel_type,
            participants,
            visibility=visibility,
            social_distance=social_distance,
            clock_phase_hint=clock_phase_hint,
            memory_policy=memory_policy,
            digest_policy=digest_policy,
            prompt_injection_policy=prompt_injection_policy,
            metadata=metadata,
            created_at=created_at,
        )

    async def list_channels(self, channel_type=None, participant=None, limit=100):
        return await self._run_sync(
            "list_channels",
            channel_type=channel_type,
            participant=participant,
            limit=limit,
        )

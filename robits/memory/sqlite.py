from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value):
    return json.dumps({} if value is None else value, sort_keys=True)


@dataclass(frozen=True)
class MemorySearchResult:
    kind: str
    record_id: str
    content: str
    agent_id: str | None
    session_id: str | None
    source: str | None
    relationship_type: str | None
    conversation_type: str | None
    created_at: str


@dataclass(frozen=True)
class MemoryDigestSource:
    source_table: str
    source_id: str
    source_created_at: str | None
    position: int
    metadata: dict[str, Any]


class SQLiteMemoryStore:
    """Small repository API for durable local Robits memory records."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.ensure_schema()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def ensure_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                display_name TEXT,
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS contacts (
                owner_agent_id TEXT NOT NULL,
                contact_agent_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (owner_agent_id, contact_agent_id),
                FOREIGN KEY (owner_agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (contact_agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender_agent_id TEXT NOT NULL,
                receiver_agent_id TEXT,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'message',
                visibility TEXT NOT NULL DEFAULT 'public',
                relationship_type TEXT,
                conversation_type TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (sender_agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (receiver_agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS thoughts (
                thought_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                relationship_type TEXT,
                conversation_type TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS todos (
                todo_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                session_id TEXT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                content TEXT,
                due_at TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                tool_call_id TEXT PRIMARY KEY,
                session_id TEXT,
                agent_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL DEFAULT '{}',
                result_content TEXT,
                status TEXT NOT NULL DEFAULT 'requested',
                relationship_type TEXT,
                conversation_type TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS memory_entries (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                session_id TEXT,
                source_table TEXT,
                source_id TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                relationship_type TEXT,
                conversation_type TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS memory_digests (
                digest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                session_id TEXT,
                content TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                source_start_at TEXT,
                source_end_at TEXT,
                relationship_type TEXT,
                conversation_type TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS memory_digest_sources (
                digest_id INTEGER NOT NULL,
                source_table TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_created_at TEXT,
                position INTEGER NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (digest_id, source_table, source_id),
                FOREIGN KEY (digest_id) REFERENCES memory_digests(digest_id)
                    ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                kind UNINDEXED,
                record_id UNINDEXED,
                agent_id UNINDEXED,
                related_agent_id UNINDEXED,
                session_id UNINDEXED,
                source UNINDEXED,
                relationship_type UNINDEXED,
                conversation_type UNINDEXED,
                created_at UNINDEXED,
                content
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_agent_id);
            CREATE INDEX IF NOT EXISTS idx_messages_receiver ON messages(receiver_agent_id);
            CREATE INDEX IF NOT EXISTS idx_thoughts_agent ON thoughts(agent_id);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memory_entries_agent ON memory_entries(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memory_digests_agent ON memory_digests(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memory_digest_sources_digest
                ON memory_digest_sources(digest_id);
            """
        )
        self.connection.commit()

    def create_session(self, session_id, title=None, started_at=None, metadata=None):
        created_at = started_at or _utc_now()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO sessions(session_id, title, started_at, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, title, created_at, _json_dumps(metadata)),
        )
        self.connection.commit()
        return session_id

    def upsert_agent(
        self,
        agent_id,
        role,
        display_name=None,
        lifecycle_state="active",
        created_at=None,
        metadata=None,
    ):
        timestamp = created_at or _utc_now()
        self.connection.execute(
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
        self.connection.commit()
        return agent_id

    def add_contact(
        self,
        owner_agent_id,
        contact_agent_id,
        relationship_type,
        notes=None,
        created_at=None,
        metadata=None,
    ):
        timestamp = created_at or _utc_now()
        self.connection.execute(
            """
            INSERT INTO contacts(
                owner_agent_id, contact_agent_id, relationship_type, notes,
                created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_agent_id, contact_agent_id) DO UPDATE SET
                relationship_type = excluded.relationship_type,
                notes = excluded.notes,
                metadata_json = excluded.metadata_json
            """,
            (
                owner_agent_id,
                contact_agent_id,
                relationship_type,
                notes,
                timestamp,
                _json_dumps(metadata),
            ),
        )
        self.connection.commit()

    def append_message(
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
        cursor = self.connection.execute(
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
        self._index(
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
        self.connection.commit()
        return record_id

    def append_thought(
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
        cursor = self.connection.execute(
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
        self._index(
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
        self.connection.commit()
        return record_id

    def append_todo(
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
        timestamp = created_at or _utc_now()
        cursor = self.connection.execute(
            """
            INSERT INTO todos(
                agent_id, session_id, title, status, content, due_at,
                created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                session_id,
                title,
                status,
                content,
                due_at,
                timestamp,
                _json_dumps(metadata),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def append_tool_call(
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
        timestamp = created_at or _utc_now()
        self.connection.execute(
            """
            INSERT INTO tool_calls(
                tool_call_id, session_id, agent_id, tool_name, arguments_json,
                result_content, status, relationship_type, conversation_type,
                source, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tool_call_id) DO UPDATE SET
                session_id = excluded.session_id,
                agent_id = excluded.agent_id,
                tool_name = excluded.tool_name,
                arguments_json = excluded.arguments_json,
                result_content = COALESCE(excluded.result_content, tool_calls.result_content),
                status = excluded.status,
                relationship_type = excluded.relationship_type,
                conversation_type = excluded.conversation_type,
                source = excluded.source,
                metadata_json = excluded.metadata_json
            """,
            (
                tool_call_id,
                session_id,
                agent_id,
                tool_name,
                _json_dumps(arguments),
                result_content,
                status,
                relationship_type,
                conversation_type,
                source,
                timestamp,
                _json_dumps(metadata),
            ),
        )
        if result_content:
            self._delete_index("tool_call", tool_call_id)
            self._index(
                "tool_call",
                tool_call_id,
                agent_id,
                None,
                session_id,
                source,
                relationship_type,
                conversation_type,
                timestamp,
                result_content,
            )
        self.connection.commit()
        return tool_call_id

    def list_todos(self, agent_id=None, session_id=None, status=None, limit=100):
        clauses = []
        params: list[Any] = []
        for column, value in (
            ("agent_id", agent_id),
            ("session_id", session_id),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self._rows(
            f"""
            SELECT * FROM todos
            {where}
            ORDER BY created_at, todo_id
            LIMIT ?
            """,
            params + [limit],
        )

    def append_memory_entry(
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
        if kind == "memory_digest":
            raise ValueError("Use append_memory_digest for compacted memory artifacts.")
        timestamp = created_at or _utc_now()
        cursor = self.connection.execute(
            """
            INSERT INTO memory_entries(
                agent_id, session_id, source_table, source_id, kind, content,
                relationship_type, conversation_type, source, created_at,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                session_id,
                source_table,
                None if source_id is None else str(source_id),
                kind,
                content,
                relationship_type,
                conversation_type,
                source,
                timestamp,
                _json_dumps(metadata),
            ),
        )
        record_id = cursor.lastrowid
        self._index(
            kind,
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
        self.connection.commit()
        return record_id

    def append_memory_digest(
        self,
        content,
        source_refs,
        agent_id=None,
        session_id=None,
        prompt_version="memory-digest-v1",
        source_start_at=None,
        source_end_at=None,
        relationship_type=None,
        conversation_type=None,
        source="digest",
        created_at=None,
        metadata=None,
    ):
        if not source_refs:
            raise ValueError("Memory digest requires at least one source reference.")
        normalized_sources = self._normalize_digest_source_refs(source_refs)
        timestamp = created_at or _utc_now()
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO memory_digests(
                    agent_id, session_id, content, prompt_version, source_start_at,
                    source_end_at, relationship_type, conversation_type, source,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    session_id,
                    content,
                    prompt_version,
                    source_start_at,
                    source_end_at,
                    relationship_type,
                    conversation_type,
                    source,
                    timestamp,
                    _json_dumps(metadata),
                ),
            )
            digest_id = cursor.lastrowid
            for position, source_ref in enumerate(normalized_sources):
                self.connection.execute(
                    """
                    INSERT INTO memory_digest_sources(
                        digest_id, source_table, source_id, source_created_at,
                        position, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        digest_id,
                        source_ref["source_table"],
                        source_ref["source_id"],
                        source_ref.get("source_created_at"),
                        position,
                        _json_dumps(source_ref.get("metadata")),
                    ),
                )
            self._index(
                "memory_digest",
                digest_id,
                agent_id,
                None,
                session_id,
                source,
                relationship_type,
                conversation_type,
                timestamp,
                content,
            )
        return digest_id

    def get_memory_digest(self, digest_id):
        row = self.connection.execute(
            "SELECT * FROM memory_digests WHERE digest_id = ?",
            (digest_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_memory_digest_sources(self, digest_id):
        rows = self.connection.execute(
            """
            SELECT source_table, source_id, source_created_at, position, metadata_json
            FROM memory_digest_sources
            WHERE digest_id = ?
            ORDER BY position
            """,
            (digest_id,),
        ).fetchall()
        return [
            MemoryDigestSource(
                source_table=row["source_table"],
                source_id=row["source_id"],
                source_created_at=row["source_created_at"],
                position=row["position"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def expand_memory_digest_sources(self, digest_id):
        expanded = []
        for source_ref in self.get_memory_digest_sources(digest_id):
            row = self._fetch_source_record(source_ref.source_table, source_ref.source_id)
            if row is not None:
                row["source_table"] = source_ref.source_table
                row["source_id"] = source_ref.source_id
                row["digest_position"] = source_ref.position
                expanded.append(row)
        return expanded

    def list_messages(self, session_id=None, agent_id=None, limit=100):
        clauses = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if agent_id is not None:
            clauses.append("(sender_agent_id = ? OR receiver_agent_id = ?)")
            params.extend([agent_id, agent_id])
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self._rows(
            f"""
            SELECT * FROM messages
            {where}
            ORDER BY created_at, message_id
            LIMIT ?
            """,
            params + [limit],
        )

    def list_agent_records(self, agent_id, limit=100):
        records = []
        records.extend(
            self._rows(
                """
                SELECT 'message' AS record_type, message_id AS record_id, content,
                       session_id, ? AS agent_id, sender_agent_id,
                       receiver_agent_id, created_at
                FROM messages
                WHERE sender_agent_id = ? OR receiver_agent_id = ?
                ORDER BY created_at, message_id
                LIMIT ?
                """,
                (agent_id, agent_id, agent_id, limit),
            )
        )
        records.extend(
            self._rows(
                """
                SELECT 'thought' AS record_type, thought_id AS record_id, content,
                       session_id, agent_id, NULL AS sender_agent_id,
                       NULL AS receiver_agent_id, created_at
                FROM thoughts
                WHERE agent_id = ?
                ORDER BY created_at, thought_id
                LIMIT ?
                """,
                (agent_id, limit),
            )
        )
        records.extend(
            self._rows(
                """
                SELECT 'tool_call' AS record_type, tool_call_id AS record_id,
                       result_content AS content, session_id, agent_id,
                       NULL AS sender_agent_id, NULL AS receiver_agent_id,
                       created_at
                FROM tool_calls
                WHERE agent_id = ?
                ORDER BY created_at, tool_call_id
                LIMIT ?
                """,
                (agent_id, limit),
            )
        )
        return sorted(records, key=lambda row: row["created_at"])[:limit]

    def search(
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
        clauses = ["memory_fts MATCH ?"]
        params: list[Any] = [query]
        for column, value in (
            ("session_id", session_id),
            ("relationship_type", relationship_type),
            ("conversation_type", conversation_type),
            ("source", source),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if agent_id is not None:
            clauses.append("(agent_id = ? OR related_agent_id = ?)")
            params.extend([agent_id, agent_id])
        if start_at is not None:
            clauses.append("created_at >= ?")
            params.append(start_at)
        if end_at is not None:
            clauses.append("created_at <= ?")
            params.append(end_at)
        rows = self.connection.execute(
            f"""
            SELECT kind, record_id, agent_id, session_id, source,
                   relationship_type, conversation_type, created_at, content
            FROM memory_fts
            WHERE {" AND ".join(clauses)}
            ORDER BY bm25(memory_fts)
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        return [MemorySearchResult(**dict(row)) for row in rows]

    def _index(
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
        self.connection.execute(
            """
            INSERT INTO memory_fts(
                kind, record_id, agent_id, related_agent_id, session_id, source,
                relationship_type, conversation_type, created_at, content
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

    def _delete_index(self, kind, record_id):
        self.connection.execute(
            "DELETE FROM memory_fts WHERE kind = ? AND record_id = ?",
            (kind, str(record_id)),
        )

    def _validate_source_table(self, source_table):
        if source_table not in {
            "messages",
            "thoughts",
            "tool_calls",
            "memory_entries",
            "memory_digests",
        }:
            raise ValueError(f"Unsupported memory digest source table: {source_table}")

    def _normalize_digest_source_refs(self, source_refs):
        normalized = []
        for source_ref in source_refs:
            if "source_table" not in source_ref or "source_id" not in source_ref:
                raise ValueError("Memory digest sources require source_table and source_id.")
            source_table = source_ref["source_table"]
            self._validate_source_table(source_table)
            metadata = source_ref.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError("Memory digest source metadata must be an object.")
            normalized.append(
                {
                    "source_table": source_table,
                    "source_id": str(source_ref["source_id"]),
                    "source_created_at": source_ref.get("source_created_at"),
                    "metadata": metadata,
                }
            )
        return normalized

    def _fetch_source_record(self, source_table, source_id):
        self._validate_source_table(source_table)
        id_column = {
            "messages": "message_id",
            "thoughts": "thought_id",
            "tool_calls": "tool_call_id",
            "memory_entries": "memory_id",
            "memory_digests": "digest_id",
        }[source_table]
        row = self.connection.execute(
            f"SELECT * FROM {source_table} WHERE {id_column} = ?",
            (source_id,),
        ).fetchone()
        return dict(row) if row else None

    def _rows(self, query, params=()):
        return [dict(row) for row in self.connection.execute(query, params).fetchall()]

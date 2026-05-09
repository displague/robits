from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Canonical channel types for relationship-scoped conversations
CHANNEL_ORG_CHAT = "org_chat"
CHANNEL_AGENT_THOUGHT = "agent_thought"
CHANNEL_AGENT_DM = "agent_dm"
CHANNEL_THERAPIST = "therapist"
CHANNEL_FAMILY_MEMBER = "family_member"
CHANNEL_FAMILY_GROUP = "family_group"
CHANNEL_WORK_PEER = "work_peer"
CHANNEL_SYSTEM = "system"

# Social-distance scale (float labels for use in channels.social_distance)
SOCIAL_SELF = 0.0
SOCIAL_FAMILIAL = 1.0
SOCIAL_FRIENDLY = 2.0
SOCIAL_PROFESSIONAL = 3.0
SOCIAL_UNKNOWN = 4.0
SOCIAL_HOSTILE = 5.0


SOCIAL_SELF = 0
SOCIAL_FAMILIAL = 1
SOCIAL_FRIENDLY = 2
SOCIAL_PROFESSIONAL = 3
SOCIAL_UNKNOWN = 4
SOCIAL_HOSTILE = 5


def compute_phase_shift(current_phase: float, event_phase: float, social_distance: float) -> float:
    """Shift current_phase toward event_phase, weighted by closeness (inverse of social_distance+1)."""
    weight = 1.0 / (social_distance + 1.0)
    shifted = current_phase + weight * (event_phase - current_phase)
    return max(0.0, min(1.0, shifted))


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


@dataclass(frozen=True)
class ExpandedMemoryDigestSource:
    source_table: str
    source_id: str
    source_created_at: str | None
    position: int
    depth: int
    parent_digest_id: int
    source_path: tuple[int, ...]
    metadata: dict[str, Any]
    record: dict[str, Any]


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

            CREATE TABLE IF NOT EXISTS channels (
                channel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_type TEXT NOT NULL,
                participants_json TEXT NOT NULL DEFAULT '[]',
                visibility TEXT NOT NULL DEFAULT 'public',
                social_distance REAL CHECK (social_distance IS NULL OR (social_distance >= 0.0 AND social_distance <= 5.0)),
                clock_phase_hint REAL,
                memory_policy TEXT NOT NULL DEFAULT 'default',
                digest_policy TEXT NOT NULL DEFAULT 'default',
                prompt_injection_policy TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(channel_type, participants_json)
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
                channel_id INTEGER,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (sender_agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (receiver_agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            );

            CREATE TABLE IF NOT EXISTS thoughts (
                thought_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                relationship_type TEXT,
                channel_id INTEGER,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
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
                digest_type TEXT NOT NULL DEFAULT 'episodic',
                generation INTEGER NOT NULL DEFAULT 1,
                version INTEGER NOT NULL DEFAULT 1,
                superseded_by_digest_id INTEGER,
                system_only INTEGER NOT NULL DEFAULT 0,
                accessibility TEXT NOT NULL DEFAULT 'agent',
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

            CREATE TABLE IF NOT EXISTS runtime_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sequence INTEGER,
                event_type TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'public',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
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
            CREATE INDEX IF NOT EXISTS idx_runtime_events_session ON runtime_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_events_type ON runtime_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_channels_type ON channels(channel_type);
            """
        )
        self._ensure_memory_digest_columns()
        self._ensure_channel_schema()
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_thoughts_channel ON thoughts(channel_id)"
        )
        self._ensure_agent_phase_column()
        self._ensure_message_sender_phase_column()
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_digests_current
                ON memory_digests(digest_type, superseded_by_digest_id, accessibility, system_only)
            """
        )
        self.connection.commit()

    def _ensure_channel_schema(self):
        for table in ("messages", "thoughts"):
            cols = {
                row["name"]
                for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "channel_id" not in cols:
                self.connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN channel_id INTEGER REFERENCES channels(channel_id)"
                )

    def _ensure_memory_digest_columns(self):
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(memory_digests)").fetchall()
        }
        migrations = {
            "digest_type": "ALTER TABLE memory_digests ADD COLUMN digest_type TEXT NOT NULL DEFAULT 'episodic'",
            "generation": "ALTER TABLE memory_digests ADD COLUMN generation INTEGER NOT NULL DEFAULT 1",
            "version": "ALTER TABLE memory_digests ADD COLUMN version INTEGER NOT NULL DEFAULT 1",
            "superseded_by_digest_id": "ALTER TABLE memory_digests ADD COLUMN superseded_by_digest_id INTEGER",
            "system_only": "ALTER TABLE memory_digests ADD COLUMN system_only INTEGER NOT NULL DEFAULT 0",
            "accessibility": "ALTER TABLE memory_digests ADD COLUMN accessibility TEXT NOT NULL DEFAULT 'agent'",
        }
        for column, statement in migrations.items():
            if column not in columns:
                self.connection.execute(statement)

    def _ensure_agent_phase_column(self):
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(agents)").fetchall()
        }
        if "circadian_phase" not in columns:
            self.connection.execute(
                "ALTER TABLE agents ADD COLUMN circadian_phase REAL DEFAULT 0.5"
            )

    def _ensure_message_sender_phase_column(self):
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "sender_phase" not in columns:
            self.connection.execute(
                "ALTER TABLE messages ADD COLUMN sender_phase REAL"
            )

    def get_agent_phase(self, agent_id) -> "float | None":
        """Return the current circadian phase for an agent, or None if not found."""
        row = self.connection.execute(
            "SELECT circadian_phase FROM agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        if row is None:
            return None
        return row["circadian_phase"]

    def set_agent_phase(self, agent_id, phase: float) -> None:
        """Update the circadian phase for an agent; clamps to [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(phase)))
        self.connection.execute(
            "UPDATE agents SET circadian_phase = ? WHERE agent_id = ?",
            (clamped, agent_id),
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

    def list_recent_message_ids(self, session_id, limit):
        """Return the last ``limit`` message IDs for a session in chronological order."""
        rows = self.connection.execute(
            """
            SELECT message_id FROM messages
            WHERE session_id = ?
            ORDER BY message_id DESC LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [row["message_id"] for row in reversed(rows)]

    def list_recent_message_ids_by_channel(self, session_id, limit, channel_id):
        """Return the last ``limit`` message IDs for a session filtered by channel."""
        rows = self.connection.execute(
            """
            SELECT message_id FROM messages
            WHERE session_id = ? AND channel_id = ?
            ORDER BY message_id DESC LIMIT ?
            """,
            (session_id, channel_id, limit),
        ).fetchall()
        return [row["message_id"] for row in reversed(rows)]

    def end_session(self, session_id, ended_at=None):
        self.connection.execute(
            "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
            (ended_at or _utc_now(), session_id),
        )
        self.connection.commit()

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

    def get_or_create_channel(
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
        normalized_participants = self._normalize_participants(participants)
        participants_json = json.dumps(normalized_participants)
        self._validate_social_distance(social_distance)
        timestamp = created_at or _utc_now()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO channels(
                channel_type, participants_json, visibility, social_distance, clock_phase_hint,
                memory_policy, digest_policy, prompt_injection_policy, created_at, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel_type,
                participants_json,
                visibility,
                social_distance,
                clock_phase_hint,
                memory_policy,
                digest_policy,
                prompt_injection_policy,
                timestamp,
                _json_dumps(metadata),
            ),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT channel_id FROM channels WHERE channel_type = ? AND participants_json = ?",
            (channel_type, participants_json),
        ).fetchone()
        return row["channel_id"]

    def list_channels(self, channel_type=None, participant=None, limit=100):
        clauses = []
        params: list[Any] = []
        if channel_type is not None:
            clauses.append("channel_type = ?")
            params.append(channel_type)
        if participant is not None:
            if not isinstance(participant, str):
                raise TypeError("participant must be a string")
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(channels.participants_json) WHERE json_each.value = ?)"
            )
            params.append(participant)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM channels {where} ORDER BY channel_id LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _normalize_participants(participants):
        values = participants or []
        for participant in values:
            if not isinstance(participant, str):
                raise TypeError("participants must contain only strings")
        return sorted(set(values))

    @staticmethod
    def _validate_social_distance(social_distance):
        if social_distance is None:
            return
        # bool is a subclass of int in Python; reject it as a semantic scale value.
        if not isinstance(social_distance, (int, float)) or isinstance(
            social_distance, bool
        ):
            raise TypeError("social_distance must be a number between 0.0 and 5.0")
        if social_distance < SOCIAL_SELF or social_distance > SOCIAL_HOSTILE:
            raise ValueError(
                f"social_distance must be between {SOCIAL_SELF:.1f} and {SOCIAL_HOSTILE:.1f}"
            )

    def _channel_type(self, channel_id):
        if channel_id is None:
            return None
        row = self.connection.execute(
            "SELECT channel_type FROM channels WHERE channel_id = ?", (channel_id,)
        ).fetchone()
        return row["channel_type"] if row else None

    def append_message(
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
        sender_phase=None,
    ):
        timestamp = created_at or _utc_now()
        cursor = self.connection.execute(
            """
            INSERT INTO messages(
                session_id, sender_agent_id, receiver_agent_id, content, kind,
                visibility, relationship_type, channel_id, source,
                created_at, metadata_json, sender_phase
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                sender_phase,
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
            self._channel_type(channel_id),
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
        channel_id=None,
        source=None,
        created_at=None,
        metadata=None,
    ):
        timestamp = created_at or _utc_now()
        cursor = self.connection.execute(
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
        self._index(
            "thought",
            record_id,
            agent_id,
            None,
            session_id,
            source,
            relationship_type,
            self._channel_type(channel_id),
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
        if not source_refs:
            raise ValueError("Memory digest requires at least one source reference.")
        self._validate_digest_type(digest_type)
        if version < 1:
            raise ValueError("Memory digest version must be at least 1.")
        if accessibility is None:
            accessibility = "system" if system_only else "agent"
        self._validate_digest_accessibility(accessibility)
        normalized_sources = self._normalize_digest_source_refs(source_refs)
        if generation is None:
            generation = self._next_digest_generation(normalized_sources)
        if generation < 0:
            raise ValueError("Memory digest generation must be non-negative.")
        supersedes_digest_ids = [
            int(digest_id) for digest_id in (supersedes_digest_ids or [])
        ]
        timestamp = created_at or _utc_now()
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO memory_digests(
                    agent_id, session_id, content, digest_type, generation, version,
                    system_only, accessibility, prompt_version, source_start_at,
                    source_end_at, relationship_type, conversation_type, source,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    session_id,
                    content,
                    digest_type,
                    generation,
                    version,
                    1 if system_only else 0,
                    accessibility,
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
            for superseded_digest_id in supersedes_digest_ids:
                self.connection.execute(
                    """
                    UPDATE memory_digests
                    SET superseded_by_digest_id = ?
                    WHERE digest_id = ?
                    """,
                    (digest_id, superseded_digest_id),
                )
        return digest_id

    def seed_memory_digest(
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
        self._validate_digest_type(digest_type)
        seed_id = self.append_memory_entry(
            f"{digest_type}_digest_seed",
            content,
            agent_id=agent_id,
            session_id=session_id,
            source=source,
            created_at=created_at,
            metadata=metadata,
            relationship_type=relationship_type,
        )
        return self.append_memory_digest(
            content,
            [{"source_table": "memory_entries", "source_id": seed_id}],
            agent_id=agent_id,
            session_id=session_id,
            digest_type=digest_type,
            generation=0,
            version=1,
            system_only=system_only,
            accessibility=accessibility,
            prompt_version=prompt_version,
            source=source,
            created_at=created_at,
            metadata=metadata,
            relationship_type=relationship_type,
        )

    def seed_identity_and_goal_digests(
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
        if short_term_goal_content is None:
            short_term_goal_content = long_term_goal_content
        return {
            "identity": self.seed_memory_digest(
                "identity",
                identity_content,
                agent_id=agent_id,
                session_id=session_id,
                created_at=created_at,
                metadata=metadata,
                relationship_type=identity_relationship_type,
            ),
            "goal_long_term": self.seed_memory_digest(
                "goal_long_term",
                long_term_goal_content,
                agent_id=agent_id,
                session_id=session_id,
                created_at=created_at,
                metadata=metadata,
            ),
            "goal_short_term": self.seed_memory_digest(
                "goal_short_term",
                short_term_goal_content,
                agent_id=agent_id,
                session_id=session_id,
                created_at=created_at,
                metadata=metadata,
            ),
        }

    def append_runtime_event(
        self,
        session_id,
        event_type,
        payload=None,
        visibility="public",
        sequence=None,
        created_at=None,
    ):
        timestamp = created_at or _utc_now()
        cursor = self.connection.execute(
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
        self.connection.commit()
        return cursor.lastrowid

    def append_runtime_event_object(self, event):
        return self.append_runtime_event(
            event.session_id,
            event.event_type,
            payload=event.payload,
            visibility=event.visibility,
            sequence=event.sequence,
            created_at=event.created_at,
        )

    def list_runtime_events(
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
        return self._rows(
            f"""
            SELECT * FROM runtime_events
            {where}
            ORDER BY COALESCE(sequence, event_id), event_id
            LIMIT ?
            """,
            params + [limit],
        )

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

    def expand_memory_digest_sources(
        self,
        digest_id,
        recursive=False,
        include_digest_records=True,
        max_depth=None,
    ):
        if recursive:
            return self._expand_memory_digest_sources_recursive(
                int(digest_id),
                include_digest_records=include_digest_records,
                max_depth=max_depth,
            )
        expanded = []
        for source_ref in self.get_memory_digest_sources(digest_id):
            row = self._fetch_source_record(source_ref.source_table, source_ref.source_id)
            if row is not None:
                row["source_table"] = source_ref.source_table
                row["source_id"] = source_ref.source_id
                row["digest_position"] = source_ref.position
                expanded.append(row)
        return expanded

    def expand_memory_digest_source_tree(self, digest_id):
        digest_id = int(digest_id)
        digest = self.get_memory_digest(digest_id)
        if digest is None:
            return None
        return {
            "digest": digest,
            "sources": self._build_memory_digest_source_tree(digest_id, visited=set()),
        }

    def list_memory_digests(
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
        clauses = []
        params: list[Any] = []
        for column, value in (
            ("agent_id", agent_id),
            ("session_id", session_id),
            ("digest_type", digest_type),
            ("relationship_type", relationship_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if current_only:
            clauses.append("superseded_by_digest_id IS NULL")
        if accessible_only:
            clauses.append("accessibility = 'agent'")
            clauses.append("system_only = 0")
        elif not include_system_only:
            clauses.append("system_only = 0")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self._rows(
            f"""
            SELECT * FROM memory_digests
            {where}
            ORDER BY created_at, digest_id
            LIMIT ?
            """,
            params + [limit],
        )

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

    def search_cascade(
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
        """Search outer FTS records and surface parent digests for inner hits.

        Fetch more than ``limit`` inner records so that cascade-surfaced parent
        digests have room after the final ``outer[:limit]`` trim.
        """
        inner_limit = max(limit, limit * 2)
        outer = self.search(
            query,
            agent_id=agent_id,
            session_id=session_id,
            relationship_type=relationship_type,
            conversation_type=conversation_type,
            source=source,
            start_at=start_at,
            end_at=end_at,
            limit=inner_limit,
        )

        # Map FTS kind values to their canonical source_table names so we can
        # look up whether each hit is a source of a parent digest.
        _kind_to_table: dict[str, str] = {
            "message": "messages",
            "thought": "thoughts",
            "tool_call": "tool_calls",
        }

        # Group non-digest outer hits by their source_table.
        source_ids_by_table: dict[str, list[str]] = {
            "messages": [],
            "thoughts": [],
            "tool_calls": [],
            "memory_entries": [],
        }
        outer_digest_ids: set[str] = set()
        for result in outer:
            if result.kind == "memory_digest":
                outer_digest_ids.add(result.record_id)
                continue
            table = _kind_to_table.get(result.kind, "memory_entries")
            source_ids_by_table[table].append(result.record_id)

        # For each non-digest outer hit, find accessible parent digests that
        # reference it as a source.  Surface those digests at the outer level.
        for source_table, source_ids in source_ids_by_table.items():
            if not source_ids:
                continue
            placeholders = ", ".join("?" * len(source_ids))
            params: list[Any] = [source_table] + source_ids
            agent_clauses = []
            if agent_id is not None:
                agent_clauses.append("md.agent_id = ?")
                params.append(agent_id)
            params.append(limit)
            agent_filter = f"AND ({' OR '.join(agent_clauses)})" if agent_clauses else ""
            rows = self.connection.execute(
                f"""
                SELECT DISTINCT md.digest_id, md.agent_id, md.session_id,
                       md.source, md.relationship_type, md.conversation_type,
                       md.created_at, md.content
                FROM memory_digest_sources mds
                JOIN memory_digests md ON md.digest_id = mds.digest_id
                WHERE mds.source_table = ?
                  AND mds.source_id IN ({placeholders})
                  AND md.accessibility = 'agent'
                  AND md.system_only = 0
                  {agent_filter}
                LIMIT ?
                """,
                params,
            ).fetchall()
            for row in rows:
                digest_id_str = str(row["digest_id"])
                if digest_id_str not in outer_digest_ids:
                    outer.append(
                        MemorySearchResult(
                            kind="memory_digest",
                            record_id=digest_id_str,
                            content=row["content"],
                            agent_id=row["agent_id"],
                            session_id=row["session_id"],
                            source=row["source"],
                            relationship_type=row["relationship_type"],
                            conversation_type=row["conversation_type"],
                            created_at=row["created_at"],
                        )
                    )
                    outer_digest_ids.add(digest_id_str)

        return outer[:limit]

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

    def _validate_digest_type(self, digest_type):
        if digest_type not in {"episodic", "identity", "goal_long_term", "goal_short_term"}:
            raise ValueError(
                "Memory digest type must be episodic, identity, goal_long_term, or goal_short_term."
            )

    def _validate_digest_accessibility(self, accessibility):
        if accessibility not in {"agent", "system"}:
            raise ValueError("Memory digest accessibility must be agent or system.")

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

    def _next_digest_generation(self, normalized_sources):
        source_digest_ids = [
            source_ref["source_id"]
            for source_ref in normalized_sources
            if source_ref["source_table"] == "memory_digests"
        ]
        if not source_digest_ids:
            return 1
        placeholders = ", ".join("?" for _ in source_digest_ids)
        row = self.connection.execute(
            f"""
            SELECT MAX(generation) AS max_generation
            FROM memory_digests
            WHERE digest_id IN ({placeholders})
            """,
            source_digest_ids,
        ).fetchone()
        max_generation = row["max_generation"] if row else None
        return (max_generation or 0) + 1

    def _expand_memory_digest_sources_recursive(
        self,
        digest_id,
        include_digest_records=True,
        max_depth=None,
    ):
        expanded: list[ExpandedMemoryDigestSource] = []

        def visit(current_digest_id, depth, path, visited):
            if current_digest_id in visited:
                return
            if max_depth is not None and depth > max_depth:
                return
            next_visited = visited | {current_digest_id}
            for source_ref in self.get_memory_digest_sources(current_digest_id):
                row = self._fetch_source_record(source_ref.source_table, source_ref.source_id)
                if row is None:
                    continue
                source_digest_id = (
                    int(source_ref.source_id)
                    if source_ref.source_table == "memory_digests"
                    else None
                )
                next_path = path + ((source_digest_id,) if source_digest_id else ())
                if source_ref.source_table != "memory_digests" or include_digest_records:
                    expanded.append(
                        ExpandedMemoryDigestSource(
                            source_table=source_ref.source_table,
                            source_id=source_ref.source_id,
                            source_created_at=source_ref.source_created_at,
                            position=source_ref.position,
                            depth=depth,
                            parent_digest_id=current_digest_id,
                            source_path=next_path,
                            metadata=source_ref.metadata,
                            record=row,
                        )
                    )
                if source_digest_id is not None:
                    visit(source_digest_id, depth + 1, next_path, next_visited)

        visit(digest_id, 0, (digest_id,), set())
        return expanded

    def _build_memory_digest_source_tree(self, digest_id, visited):
        if digest_id in visited:
            return []
        next_visited = visited | {digest_id}
        tree = []
        for source_ref in self.get_memory_digest_sources(digest_id):
            row = self._fetch_source_record(source_ref.source_table, source_ref.source_id)
            if row is None:
                continue
            node = {
                "source_table": source_ref.source_table,
                "source_id": source_ref.source_id,
                "source_created_at": source_ref.source_created_at,
                "position": source_ref.position,
                "metadata": source_ref.metadata,
                "record": row,
                "sources": [],
            }
            if source_ref.source_table == "memory_digests":
                node["sources"] = self._build_memory_digest_source_tree(
                    int(source_ref.source_id),
                    next_visited,
                )
            tree.append(node)
        return tree

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

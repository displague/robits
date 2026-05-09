from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Synchronous memory store interface used by the runtime."""

    def close(self): ...
    def ensure_schema(self): ...
    def create_session(self, session_id, title=None, started_at=None, metadata=None): ...
    def end_session(self, session_id, ended_at=None): ...
    def upsert_agent(
        self,
        agent_id,
        role,
        display_name=None,
        lifecycle_state="active",
        created_at=None,
        metadata=None,
    ): ...
    def add_contact(
        self,
        owner_agent_id,
        contact_agent_id,
        relationship_type,
        notes=None,
        created_at=None,
        metadata=None,
    ): ...
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
    ): ...
    def list_channels(self, channel_type=None, participant=None, limit=100): ...
    def get_agent_phase(self, agent_id) -> float | None: ...
    def set_agent_phase(self, agent_id, phase: float) -> None: ...
    def get_channel_social_distance(self, channel_id) -> float | None: ...
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
    ): ...
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
    ): ...
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
    ): ...
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
    ): ...
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
    ): ...
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
    ): ...
    def append_runtime_event(
        self,
        session_id,
        event_type,
        payload=None,
        visibility="public",
        sequence=None,
        created_at=None,
    ): ...
    def append_runtime_event_object(self, event): ...
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
    ): ...
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
    ): ...
    def list_recent_message_ids(self, session_id, limit): ...
    def list_recent_message_ids_by_channel(self, session_id, limit, channel_id): ...
    def list_todos(self, agent_id=None, session_id=None, status=None, limit=100): ...
    def list_runtime_events(self, session_id=None, event_type=None, visibility=None, limit=100): ...
    def get_memory_digest(self, digest_id): ...
    def get_memory_digest_sources(self, digest_id): ...
    def expand_memory_digest_sources(
        self,
        digest_id,
        recursive=False,
        include_digest_records=True,
        max_depth=None,
    ): ...
    def expand_memory_digest_source_tree(self, digest_id): ...
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
    ): ...
    def list_messages(self, session_id=None, agent_id=None, limit=100): ...
    def list_agent_records(self, agent_id, limit=100): ...
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
    ): ...
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
    ): ...


@runtime_checkable
class AsyncMemoryStoreProtocol(Protocol):
    """Async equivalent of MemoryStoreProtocol.

    Implementations should accept the same arguments as the sync store and
    return awaitable equivalents.
    """

    async def close(self): ...
    async def ensure_schema(self): ...
    async def create_session(self, session_id, title=None, started_at=None, metadata=None): ...
    async def end_session(self, session_id, ended_at=None): ...
    async def upsert_agent(
        self,
        agent_id,
        role,
        display_name=None,
        lifecycle_state="active",
        created_at=None,
        metadata=None,
    ): ...
    async def add_contact(
        self,
        owner_agent_id,
        contact_agent_id,
        relationship_type,
        notes=None,
        created_at=None,
        metadata=None,
    ): ...
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
    ): ...
    async def list_channels(self, channel_type=None, participant=None, limit=100): ...
    async def get_agent_phase(self, agent_id) -> float | None: ...
    async def set_agent_phase(self, agent_id, phase: float) -> None: ...
    async def get_channel_social_distance(self, channel_id) -> float | None: ...
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
        sender_phase=None,
    ): ...
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
    ): ...
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
    ): ...
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
    ): ...
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
    ): ...
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
    ): ...
    async def append_runtime_event(
        self,
        session_id,
        event_type,
        payload=None,
        visibility="public",
        sequence=None,
        created_at=None,
    ): ...
    async def append_runtime_event_object(self, event): ...
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
    ): ...
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
    ): ...
    async def list_recent_message_ids(self, session_id, limit): ...
    async def list_recent_message_ids_by_channel(self, session_id, limit, channel_id): ...
    async def list_todos(self, agent_id=None, session_id=None, status=None, limit=100): ...
    async def list_runtime_events(self, session_id=None, event_type=None, visibility=None, limit=100): ...
    async def get_memory_digest(self, digest_id): ...
    async def get_memory_digest_sources(self, digest_id): ...
    async def expand_memory_digest_sources(
        self,
        digest_id,
        recursive=False,
        include_digest_records=True,
        max_depth=None,
    ): ...
    async def expand_memory_digest_source_tree(self, digest_id): ...
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
    ): ...
    async def list_messages(self, session_id=None, agent_id=None, limit=100): ...
    async def list_agent_records(self, agent_id, limit=100): ...
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
    ): ...
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
    ): ...

"""Persona seeding: preload per-agent identity memories from a YAML configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_VALID_KINDS = {"thought", "message", "entry", "digest"}
_DEFAULT_PERSONAS_FILE = "personas.yaml"


def load_personas(path: str | None = None) -> dict[str, dict]:
    """Load personas.yaml and return {username: {role, full_name, entries}}.

    Supports both the new schema (username/role/full_name) and the legacy
    schema (agent) for backwards compatibility.  Returns an empty dict if
    the file is absent or unparseable.
    """
    try:
        import yaml
    except ImportError:
        return {}

    resolved = Path(path or _DEFAULT_PERSONAS_FILE)
    if not resolved.exists():
        return {}
    try:
        data = yaml.safe_load(resolved.read_text())
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    result: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        memories = item.get("memories")
        if not isinstance(memories, list):
            continue
        valid_entries = [m for m in memories if isinstance(m, dict) and m.get("kind") in _VALID_KINDS]

        # New schema: username + role + optional full_name
        username = item.get("username")
        role = item.get("role")
        if username and role:
            full_name = item.get("full_name", username)
            result[username] = {
                "role": role,
                "full_name": full_name,
                "entries": valid_entries,
            }
            continue

        # Legacy schema: agent key
        agent = item.get("agent")
        if agent:
            result[agent] = {
                "role": agent,
                "full_name": agent,
                "entries": valid_entries,
            }
    return result


def _resolve_channel_id(store: Any, channel: str | None) -> int | None:
    """Resolve a channel name to its DB id, or return None."""
    if not channel:
        return None
    try:
        from robits.memory.sqlite import SOCIAL_PROFESSIONAL
        return store.get_or_create_channel(channel, social_distance=SOCIAL_PROFESSIONAL)
    except Exception:
        return None


def seed_persona(
    store: Any,
    agent_name: str,
    entries: list[dict],
    session_id: str = "persona-seed",
) -> int:
    """Write persona memories for agent_name; skip entirely if any memory exists.

    Returns the number of records written (0 if skipped).
    """
    if not entries:
        return 0

    # Ensure a session record exists for FK constraints.
    try:
        store.create_session(session_id)
    except Exception:
        pass  # session may already exist

    # Idempotency: skip if the agent already has *any* indexed memory record.
    try:
        rows = store._rows(
            "SELECT 1 FROM memory_fts WHERE agent_id = ? LIMIT 1",
            [agent_name],
        )
        if rows:
            return 0
    except Exception:
        return 0

    written = 0
    for entry in entries:
        kind = entry.get("kind")
        content = entry.get("content", "")
        if not content:
            continue
        try:
            if kind == "thought":
                channel_id = _resolve_channel_id(store, entry.get("channel"))
                store.append_thought(
                    agent_id=agent_name,
                    content=content,
                    session_id=session_id,
                    visibility=entry.get("visibility", "private"),
                    channel_id=channel_id,
                )
                written += 1
            elif kind == "message":
                channel_id = _resolve_channel_id(store, entry.get("channel"))
                store.append_message(
                    session_id=session_id,
                    sender_agent_id=agent_name,
                    receiver_agent_id=agent_name,
                    content=content,
                    kind="message",
                    visibility=entry.get("visibility", "public"),
                    channel_id=channel_id,
                )
                written += 1
            elif kind == "entry":
                store.append_memory_entry(
                    kind=entry.get("digest_type", "identity"),
                    content=content,
                    agent_id=agent_name,
                    session_id=session_id,
                    source="persona",
                    relationship_type=entry.get("relationship_type"),
                    conversation_type=entry.get("conversation_type"),
                )
                written += 1
            elif kind == "digest":
                seed_entry_id = store.append_memory_entry(
                    kind="identity",
                    content=content,
                    agent_id=agent_name,
                    session_id=session_id,
                    source="persona",
                )
                store.append_memory_digest(
                    content=content,
                    source_refs=[{"source_table": "memory_entries", "source_id": seed_entry_id}],
                    agent_id=agent_name,
                    session_id=session_id,
                    digest_type=entry.get("digest_type", "identity"),
                    accessibility="agent",
                    system_only=False,
                    source="persona",
                    relationship_type=entry.get("relationship_type"),
                )
                written += 1
        except Exception:
            pass

    return written

"""Persona seeding: preload per-agent identity memories from a YAML configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_VALID_KINDS = {"thought", "message", "entry", "digest"}
_DEFAULT_PERSONAS_FILE = "personas.yaml"


def load_personas(path: str | None = None) -> dict[str, list[dict]]:
    """Load personas.yaml and return {agent_name: [entry_dicts]}.

    Returns an empty dict if the file is absent or unparseable.
    """
    try:
        import yaml
    except ImportError:
        return {}

    resolved = Path(path or os.environ.get("ROBITS_PERSONAS_FILE", _DEFAULT_PERSONAS_FILE))
    if not resolved.exists():
        return {}
    try:
        data = yaml.safe_load(resolved.read_text())
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    result: dict[str, list[dict]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        agent = item.get("agent")
        memories = item.get("memories")
        if not agent or not isinstance(memories, list):
            continue
        result.setdefault(agent, []).extend(
            m for m in memories if isinstance(m, dict) and m.get("kind") in _VALID_KINDS
        )
    return result


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

    # Ensure a session record exists for FK constraints; persona seeds use their own session.
    try:
        store.create_session(session_id)
    except Exception:
        pass  # session may already exist

    # Idempotency: skip if the agent already has any memory
    try:
        existing = store.list_memory_digests(
            agent_id=agent_name, digest_type="identity", current_only=True, limit=1
        )
        if existing:
            return 0
        # Also check raw memory entries
        rows = store._rows(
            "SELECT 1 FROM memory_entries WHERE agent_id = ? LIMIT 1",
            [agent_name],
        )
        if rows:
            return 0
        rows = store._rows(
            "SELECT 1 FROM thoughts WHERE agent_id = ? LIMIT 1",
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
                store.append_thought(
                    agent_id=agent_name,
                    content=content,
                    session_id=session_id,
                    visibility=entry.get("visibility", "private"),
                )
                written += 1
            elif kind == "message":
                channel = entry.get("channel")
                channel_id = None
                if channel:
                    try:
                        from robits.memory.sqlite import SOCIAL_PROFESSIONAL
                        channel_id = store.get_or_create_channel(
                            channel,
                            social_distance=SOCIAL_PROFESSIONAL,
                        )
                    except Exception:
                        pass
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

"""Persona seeding: preload per-agent identity memories from a YAML configuration."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

_VALID_KINDS = {"thought", "message", "entry", "digest"}
_DEFAULT_PERSONAS_FILE = "personas.yaml"


def load_personas(path: str | None = None) -> dict[str, dict]:
    """Load personas from a YAML file or a directory; return {username: {role, full_name, entries, ...}}.

    A file holds a list of persona entries. A directory holds one persona per
    ``*.yaml``/``*.yml`` file (a mapping; ``username`` defaults to the file
    name), so personas can be added or removed without editing a shared file.

    Optional per-persona settings (``tools``, ``tool_budget``,
    ``max_tool_rounds``, ``briefings``) are passed through as-is (None when
    absent) and validated by ``build_employee_dict``. Briefings named by string
    are read from a ``briefings/`` directory next to the file, or inside the
    persona directory.

    Supports both the new schema (username/role/full_name) and the legacy
    schema (agent) for backwards compatibility.  Returns an empty dict if
    the path is absent or unparseable.
    """
    try:
        import yaml
    except ImportError:
        return {}

    resolved = Path(path or _DEFAULT_PERSONAS_FILE)
    if resolved.is_dir():
        items = []
        for persona_file in sorted([*resolved.glob("*.yaml"), *resolved.glob("*.yml")]):
            try:
                data = yaml.safe_load(persona_file.read_text(encoding="utf-8"))
            except Exception as error:
                logging.warning("robits: cannot read persona file %s (%s) - skipping", persona_file, error)
                continue
            if isinstance(data, dict):
                if not data.get("username") and not data.get("agent"):
                    data = {**data, "username": persona_file.stem}
                items.append(data)
            elif isinstance(data, list):
                items.extend(data)
            else:
                logging.warning("robits: persona file %s holds no persona - skipping", persona_file)
        return _parse_persona_items(items, resolved / "briefings")
    if not resolved.exists():
        return {}
    try:
        data = yaml.safe_load(resolved.read_text())
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    return _parse_persona_items(data, resolved.parent / "briefings")


def _parse_persona_items(items: list, briefing_dir: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        memories = item.get("memories")
        if memories is None:
            memories = []
        if not isinstance(memories, list):
            continue
        valid_entries = [m for m in memories if isinstance(m, dict) and m.get("kind") in _VALID_KINDS]

        # New schema: username + role + optional full_name; legacy schema: agent key.
        username = item.get("username")
        role = item.get("role")
        if not (username and role):
            username = role = item.get("agent")
        if not username:
            continue
        result[username] = {
            "role": role,
            "full_name": item.get("full_name", username),
            "entries": valid_entries,
            "preprompt_work": item.get("preprompt_work"),
            "preprompt_personal": item.get("preprompt_personal"),
            "tools": item.get("tools"),
            "tool_budget": item.get("tool_budget"),
            "max_tool_rounds": item.get("max_tool_rounds"),
            "briefings": item.get("briefings"),
            "briefing_dir": str(briefing_dir),
        }
    return result


_BRIEFING_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SUMMARY_LIMIT = 200


def read_briefing_file(path: Path) -> tuple[str, str]:
    """Return (summary, text) for a Markdown briefing.

    The summary comes from ``summary:`` (or ``description:``) in optional YAML
    front matter, else from the first non-empty line.
    """
    import yaml
    text = path.read_text(encoding="utf-8")
    summary = ""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            meta = yaml.safe_load(text[4:end]) or {}
            if isinstance(meta, dict):
                summary = str(meta.get("summary") or meta.get("description") or "")
            text = text[end + 4:]
    body = text.strip()
    if not summary:
        summary = next((line.strip().lstrip("#").strip() for line in body.splitlines() if line.strip()), "")
    return summary, body


def resolve_briefings(spec: Any, briefing_dir: str | Path | None) -> dict[str, dict]:
    """Turn a persona's ``briefings`` setting into {name: {summary, text}}; raise ValueError if malformed.

    Entries are a briefing name (read from ``<briefing_dir>/<name>.md``) or an
    inline mapping with ``name``, ``summary`` and ``text``.
    """
    if spec is None:
        return {}
    if isinstance(spec, str) or not isinstance(spec, list):
        raise ValueError("briefings must be a list of briefing names or {name, summary, text} entries")
    briefings: dict[str, dict] = {}
    for entry in spec:
        if isinstance(entry, str):
            name = entry.strip()
            if not _BRIEFING_NAME.match(name):
                raise ValueError(f"briefing name {entry!r} must be letters, digits, '.', '_' or '-'")
            path = Path(briefing_dir or "briefings") / f"{name}.md"
            if not path.is_file():
                raise ValueError(f"briefing {name!r} not found at {path}")
            summary, text = read_briefing_file(path)
        elif isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if not _BRIEFING_NAME.match(name):
                raise ValueError(f"inline briefing needs a name of letters, digits, '.', '_' or '-' (got {name!r})")
            text = str(entry.get("text") or "").strip()
            if not text:
                raise ValueError(f"inline briefing {name!r} needs text")
            summary = str(entry.get("summary") or "").strip() or text.splitlines()[0]
        else:
            raise ValueError("briefings entries must be names or mappings")
        if name in briefings:
            raise ValueError(f"briefing {name!r} is listed twice")
        summary = " ".join(summary.split())
        if len(summary) > _SUMMARY_LIMIT:
            summary = summary[:_SUMMARY_LIMIT - 1] + "…"
        briefings[name] = {"summary": summary, "text": text}
    return briefings


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

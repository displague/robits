#!/usr/bin/env python3
"""Seed a memory DB with personal + work context, then run a short session.

Usage:
  OPENAI_BASE_URL=http://127.0.0.1:11434/v1/ OPENAI_API_KEY=ollama \
  OPENAI_MODEL=granite4.1:3b ROBITS_PROVIDER_API=chat \
  ROBITS_MEMORY_DB=/tmp/robits_local.db python3 run_local.py \
  --prompt "Alex, have you had any personal or work experience with Python or cooking?"

The script detects the SE persona username from personas.yaml (via Config) and seeds
memory under that key so directed routing works correctly after the persona redesign.
"""
import sys


def _se_username():
    """Return the username of the first SE persona, or 'SE' if none configured."""
    from robits.core.config import _config as _m
    for username, info in (_m.persona_entries or {}).items():
        if isinstance(info, dict) and info.get("role") in ("SE", "SoftwareEngineer"):
            return username, info.get("full_name", username)
    return "SE", "SE"


def seed_memory(db_path, agent_id, full_name):
    from robits.memory.sqlite import (
        SQLiteMemoryStore,
        CHANNEL_AGENT_THOUGHT, CHANNEL_ORG_CHAT,
        SOCIAL_PROFESSIONAL,
    )
    store = SQLiteMemoryStore(db_path)
    store.create_session("seed-session")
    store.upsert_agent(agent_id, "SoftwareEngineer", full_name,
                       username=agent_id, full_name=full_name)
    store.upsert_agent("CEO", "Human", "CEO")

    org_ch = store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
    thought_ch = store.get_or_create_channel(
        CHANNEL_AGENT_THOUGHT, participants=[agent_id],
        visibility="private", social_distance=0.0,
    )

    first = full_name.split()[0] if full_name else agent_id

    # Work memories — org_chat channel
    store.append_message("seed-session", "CEO", agent_id,
        f"{first}, can you refactor the auth module to use Python async/await?",
        channel_id=org_ch)
    store.append_message("seed-session", agent_id, "CEO",
        "Sure, I'll migrate it to asyncio. I've done this pattern several times — "
        "Python async has become second nature to me after years of backend work.",
        channel_id=org_ch)
    store.append_message("seed-session", "CEO", agent_id,
        "What's your take on the new SQLite FTS5 integration?",
        channel_id=org_ch)
    store.append_message("seed-session", agent_id, "CEO",
        "I find it elegant. The cascade-search approach is something I proposed "
        "based on prior experience building search pipelines in Python.",
        channel_id=org_ch)

    # Personal memories — agent_thought channel
    store.append_thought(agent_id,
        "I really enjoy cooking on weekends — especially Italian food. "
        "Last Saturday I made a proper carbonara from scratch; took about an hour.",
        session_id="seed-session", channel_id=thought_ch, visibility="private")
    store.append_thought(agent_id,
        "Been meaning to pick up sourdough baking. I've been reading about "
        "fermentation science — there's a surprising amount of chemistry involved.",
        session_id="seed-session", channel_id=thought_ch, visibility="private")
    store.append_thought(agent_id,
        "Python was actually my first serious language — started with it in college "
        "writing small data scripts. Feels personal, not just a work tool.",
        session_id="seed-session", channel_id=thought_ch, visibility="private")

    store.close()
    print(f"[run_local] Memory seeded for {agent_id!r} ({full_name}) at {db_path}",
          file=sys.stderr)


if __name__ == "__main__":
    from robits.core.config import _config as _m
    db_path = _m.memory_db_path or "/tmp/robits_local.db"
    agent_id, full_name = _se_username()
    seed_memory(db_path, agent_id, full_name)
    # Delegate to main.py's entry point
    import main
    main.main(argv=sys.argv[1:])

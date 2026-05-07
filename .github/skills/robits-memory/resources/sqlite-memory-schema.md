# SQLite Memory Schema

The memory store lives in `robits/memory/sqlite.py`.

Core tables:

- `sessions`: run metadata and timestamps.
- `agents`: durable agent identity, role, display name, and lifecycle state.
- `contacts`: per-agent relationship records such as coworker, family, therapy, or personal project.
- `messages`: public or policy-visible conversation entries.
- `thoughts`: private inner-monologue records.
- `todos`: agent-owned commitments and side-project tasks.
- `tool_calls`: requested or completed tool calls, including result content.
- `memory_entries`: general memory records that are not compacted digests.
- `memory_digests`: compacted memory artifacts with prompt version, time range, and retrieval filters.
- `memory_digest_sources`: ordered source links from a digest back to raw records.
- `runtime_events`: headless observability events for live-session replay or TUI readers.
- `memory_fts`: FTS5 index over message, thought, tool-result, memory-entry, and memory-digest content.

Repository API:

- `create_session`
- `upsert_agent`
- `add_contact`
- `append_message`
- `append_thought`
- `append_todo`
- `append_tool_call`
- `append_memory_entry`
- `append_memory_digest`
- `get_memory_digest`
- `get_memory_digest_sources`
- `expand_memory_digest_sources`
- `append_runtime_event`
- `append_runtime_event_object`
- `list_runtime_events`
- `list_todos`
- `list_messages`
- `list_agent_records`
- `search`

Search supports filters for agent, session, relationship type, conversation type,
source, and date windows. It searches raw records and memory digests through the
same FTS table. Use `expand_memory_digest_sources` to drill from a digest back to
the source records for reanalysis or later re-digestion.

Generated local databases should live under `data/` or `var/` by default, both
of which are ignored by git.

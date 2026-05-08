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
- `memory_digests`: compacted memory artifacts with digest type, generation,
  version, supersession state, system-only/accessibility flags, prompt version,
  time range, and retrieval filters.
- `memory_digest_sources`: ordered source links from a digest back to raw
  records or earlier `memory_digests`.
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
- `seed_memory_digest`
- `seed_identity_and_goal_digests`
- `get_memory_digest`
- `get_memory_digest_sources`
- `expand_memory_digest_sources`
- `expand_memory_digest_source_tree`
- `list_memory_digests`
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
the source records for reanalysis or later re-digestion. Pass `recursive=True`
to include cascading digest sources, or use `expand_memory_digest_source_tree`
when automatic system reanalysis needs the full nested source tree. Pass
`include_digest_records=False` during recursive expansion when only raw leaf
sources are needed.

Digest types are `episodic`, `identity`, `goal_long_term`, and
`goal_short_term`. Backwards-compatible `append_memory_digest` calls create
current, agent-accessible episodic digests at generation 1. Digest-of-digest
inserts automatically advance generation from their source digests unless a
generation is supplied. New digest records can mark older digests as superseded
with `supersedes_digest_ids`; superseded records are preserved for future
reanalysis but can be excluded from agent-facing retrieval with
`list_memory_digests(current_only=True, accessible_only=True)`.

Generated local databases should live under `data/` or `var/` by default, both
of which are ignored by git.

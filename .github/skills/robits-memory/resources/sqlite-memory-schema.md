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
- `memory_entries`: general memory records and future memory digests with source links.
- `memory_fts`: FTS5 index over message, thought, tool-result, and memory-entry content.

Repository API:

- `create_session`
- `upsert_agent`
- `add_contact`
- `append_message`
- `append_thought`
- `append_todo`
- `append_tool_call`
- `append_memory_entry`
- `list_messages`
- `list_agent_records`
- `search`

Search supports filters for agent, session, relationship type, conversation type,
source, and date windows. Keep these filters available when adding prompt
assembly or memory-digest retrieval.

Generated local databases should live under `data/` or `var/` by default, both
of which are ignored by git.

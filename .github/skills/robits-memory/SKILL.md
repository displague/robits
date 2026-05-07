---
name: robits-memory
description: Work on the Robits SQLite memory substrate, including sessions, agents, contacts, messages, thoughts, todos, tool calls, memory entries, FTS search, retrieval filters, memory digests, and tests that use temporary databases.
---

# Robits Memory

## Process

1. Inspect `robits/memory/sqlite.py`, `tests/test_memory_store.py`, and `docs/architecture-vision.md` before changing memory behavior.
2. Keep the memory layer local-first and deterministic. Unit tests must use temporary SQLite databases and must not require model services.
3. Keep raw records durable. Memory digests can summarize context, but they must preserve source links so records can be retrieved or reanalyzed later.
4. Prefer repository APIs over direct SQL in runtime code.
5. Preserve filters for agent, session, relationship type, conversation type, source, and date windows when adding search or prompt assembly behavior.
6. Keep generated database files out of git.

## Validation

- Run `python -m unittest` after memory changes.
- Run the repo-local skill validator after changing this skill.

## References

- Read `resources/sqlite-memory-schema.md` for the current schema and API boundaries.
- Use `assets/search-filter-example.json` as a compact example of memory search filters.

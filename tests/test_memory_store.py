import sqlite3
import tempfile
import unittest
from pathlib import Path

from robits.memory import SQLiteMemoryStore


class SQLiteMemoryStoreTests(unittest.TestCase):
    def build_store(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = SQLiteMemoryStore(Path(temp_dir.name) / "memory.sqlite3")
        self.addCleanup(store.close)
        return store

    def seed_store(self):
        store = self.build_store()
        store.create_session("session-1", title="Planning")
        store.upsert_agent("CEO", "Human", "CEO")
        store.upsert_agent("SE", "SoftwareEngineer", "SE")
        store.upsert_agent("HR", "HR", "HR")
        store.add_contact("SE", "HR", "coworker")
        return store

    def test_schema_contains_core_memory_tables(self):
        store = self.build_store()

        tables = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
            )
        }

        self.assertIn("sessions", tables)
        self.assertIn("agents", tables)
        self.assertIn("contacts", tables)
        self.assertIn("messages", tables)
        self.assertIn("thoughts", tables)
        self.assertIn("todos", tables)
        self.assertIn("tool_calls", tables)
        self.assertIn("memory_entries", tables)
        self.assertIn("memory_digests", tables)
        self.assertIn("memory_digest_sources", tables)
        self.assertIn("runtime_events", tables)
        self.assertIn("memory_fts", tables)

        digest_columns = {
            row["name"]
            for row in store.connection.execute("PRAGMA table_info(memory_digests)")
        }
        self.assertIn("digest_type", digest_columns)
        self.assertIn("generation", digest_columns)
        self.assertIn("version", digest_columns)
        self.assertIn("superseded_by_digest_id", digest_columns)
        self.assertIn("system_only", digest_columns)
        self.assertIn("accessibility", digest_columns)

    def test_append_and_lookup_messages_by_session_and_agent(self):
        store = self.seed_store()

        message_id = store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Please design the durable memory substrate.",
            relationship_type="coworker",
            conversation_type="work",
            source="chat",
        )

        by_session = store.list_messages(session_id="session-1")
        by_agent = store.list_messages(agent_id="SE")

        self.assertEqual(by_session[0]["message_id"], message_id)
        self.assertEqual(by_agent[0]["content"], "Please design the durable memory substrate.")

    def test_search_covers_messages_thoughts_tool_results_and_memory_entries(self):
        store = self.seed_store()
        store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Use sqlite for coworker recall.",
            relationship_type="coworker",
            conversation_type="work",
            source="chat",
        )
        store.append_thought(
            "SE",
            "The sqlite schema needs FTS indexes for private recollection.",
            session_id="session-1",
            relationship_type="coworker",
            conversation_type="work",
            source="thinking",
        )
        store.append_tool_call(
            "call-1",
            "SE",
            "memory.search",
            result_content="Found prior notes about sqlite memory.",
            session_id="session-1",
            relationship_type="coworker",
            conversation_type="tool",
            source="tool_result",
        )
        store.append_memory_entry(
            "memory_note",
            "Note: sqlite records should preserve source links.",
            agent_id="SE",
            session_id="session-1",
            source_table="messages",
            source_id=1,
            relationship_type="coworker",
            conversation_type="work",
            source="digest",
        )

        results = store.search("sqlite", agent_id="SE", session_id="session-1")
        kinds = {result.kind for result in results}

        self.assertIn("message", kinds)
        self.assertIn("thought", kinds)
        self.assertIn("tool_call", kinds)
        self.assertIn("memory_note", kinds)

    def test_search_filters_relationship_conversation_source_and_dates(self):
        store = self.seed_store()
        store.append_thought(
            "SE",
            "Coworker plan for sqlite search.",
            session_id="session-1",
            relationship_type="coworker",
            conversation_type="work",
            source="thinking",
            created_at="2026-05-07T10:00:00+00:00",
        )
        store.append_thought(
            "SE",
            "Family plan for sqlite search.",
            session_id="session-1",
            relationship_type="family",
            conversation_type="home",
            source="thinking",
            created_at="2026-05-08T10:00:00+00:00",
        )

        results = store.search(
            "sqlite",
            relationship_type="coworker",
            conversation_type="work",
            source="thinking",
            start_at="2026-05-07T00:00:00+00:00",
            end_at="2026-05-07T23:59:59+00:00",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Coworker plan for sqlite search.")

    def test_metadata_preserves_valid_falsy_json_values(self):
        store = self.seed_store()

        store.append_tool_call(
            "call-1",
            "SE",
            "memory.empty",
            arguments=[],
            result_content="Empty args are intentional.",
            session_id="session-1",
            metadata=False,
        )

        row = store.connection.execute(
            "SELECT arguments_json, metadata_json FROM tool_calls WHERE tool_call_id = ?",
            ("call-1",),
        ).fetchone()

        self.assertEqual(row["arguments_json"], "[]")
        self.assertEqual(row["metadata_json"], "false")

    def test_tool_call_can_be_updated_with_result_and_reindexed(self):
        store = self.seed_store()

        store.append_tool_call(
            "call-1",
            "SE",
            "memory.search",
            arguments={"query": "sqlite"},
            status="requested",
            session_id="session-1",
        )
        store.append_tool_call(
            "call-1",
            "SE",
            "memory.search",
            arguments={"query": "sqlite"},
            result_content="Updated sqlite result.",
            status="completed",
            session_id="session-1",
        )

        row = store.connection.execute(
            "SELECT status, result_content FROM tool_calls WHERE tool_call_id = ?",
            ("call-1",),
        ).fetchone()
        results = store.search("updated", agent_id="SE")

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["result_content"], "Updated sqlite result.")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record_id, "call-1")

    def test_append_and_list_todos(self):
        store = self.seed_store()

        todo_id = store.append_todo(
            "SE",
            "Add prompt assembly retrieval",
            session_id="session-1",
            content="Use memory search before growing context.",
        )

        todos = store.list_todos(agent_id="SE", status="open")

        self.assertEqual(todos[0]["todo_id"], todo_id)
        self.assertEqual(todos[0]["title"], "Add prompt assembly retrieval")

    def test_agent_record_lookup_includes_messages_thoughts_and_tools(self):
        store = self.seed_store()
        store.append_message("session-1", "CEO", "SE", "Message for SE.")
        store.append_thought("SE", "Private thought.", session_id="session-1")
        store.append_tool_call(
            "call-1",
            "SE",
            "memory.search",
            result_content="Tool result.",
            session_id="session-1",
        )

        records = store.list_agent_records("SE")
        record_types = {record["record_type"] for record in records}
        message = next(record for record in records if record["record_type"] == "message")

        self.assertEqual(record_types, {"message", "thought", "tool_call"})
        self.assertEqual(message["agent_id"], "SE")
        self.assertEqual(message["sender_agent_id"], "CEO")
        self.assertEqual(message["receiver_agent_id"], "SE")

    def test_memory_digest_records_provenance_and_is_searchable(self):
        store = self.seed_store()
        message_id = store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Use sqlite to recall architecture decisions.",
            relationship_type="coworker",
            conversation_type="work",
            source="chat",
            created_at="2026-05-07T10:00:00+00:00",
        )
        thought_id = store.append_thought(
            "SE",
            "The digest should link back to raw records.",
            session_id="session-1",
            relationship_type="coworker",
            conversation_type="work",
            source="thinking",
            created_at="2026-05-07T10:01:00+00:00",
        )

        digest_id = store.append_memory_digest(
            "Digest: sqlite recall needs reversible source links.",
            [
                {
                    "source_table": "messages",
                    "source_id": message_id,
                    "source_created_at": "2026-05-07T10:00:00+00:00",
                },
                {
                    "source_table": "thoughts",
                    "source_id": thought_id,
                    "source_created_at": "2026-05-07T10:01:00+00:00",
                },
            ],
            agent_id="SE",
            session_id="session-1",
            prompt_version="memory-digest-test-v1",
            source_start_at="2026-05-07T10:00:00+00:00",
            source_end_at="2026-05-07T10:01:00+00:00",
            relationship_type="coworker",
            conversation_type="work",
        )

        digest = store.get_memory_digest(digest_id)
        sources = store.get_memory_digest_sources(digest_id)
        results = store.search("reversible", agent_id="SE")

        self.assertEqual(digest["prompt_version"], "memory-digest-test-v1")
        self.assertEqual(digest["source_start_at"], "2026-05-07T10:00:00+00:00")
        self.assertEqual([source.source_table for source in sources], ["messages", "thoughts"])
        self.assertEqual(results[0].kind, "memory_digest")
        self.assertEqual(results[0].record_id, str(digest_id))

    def test_memory_digest_sources_can_expand_to_raw_records(self):
        store = self.seed_store()
        message_id = store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Raw source message.",
        )
        tool_call_id = store.append_tool_call(
            "call-1",
            "SE",
            "memory.search",
            result_content="Raw tool result.",
            session_id="session-1",
        )
        digest_id = store.append_memory_digest(
            "Digest with two source kinds.",
            [
                {"source_table": "messages", "source_id": message_id},
                {"source_table": "tool_calls", "source_id": tool_call_id},
            ],
            agent_id="SE",
            session_id="session-1",
        )

        expanded = store.expand_memory_digest_sources(digest_id)

        self.assertEqual([row["source_table"] for row in expanded], ["messages", "tool_calls"])
        self.assertEqual(expanded[0]["content"], "Raw source message.")
        self.assertEqual(expanded[1]["result_content"], "Raw tool result.")

    def test_memory_digest_can_use_digest_sources_and_expand_recursively(self):
        store = self.seed_store()
        message_id = store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Raw cascade source message.",
        )
        first_digest_id = store.append_memory_digest(
            "First generation digest.",
            [{"source_table": "messages", "source_id": message_id}],
            agent_id="SE",
            session_id="session-1",
        )
        second_digest_id = store.append_memory_digest(
            "Second generation digest.",
            [{"source_table": "memory_digests", "source_id": first_digest_id}],
            agent_id="SE",
            session_id="session-1",
        )

        second_digest = store.get_memory_digest(second_digest_id)
        direct_sources = store.expand_memory_digest_sources(second_digest_id)
        recursive_sources = store.expand_memory_digest_sources(second_digest_id, recursive=True)
        tree = store.expand_memory_digest_source_tree(second_digest_id)

        self.assertEqual(second_digest["generation"], 2)
        self.assertEqual(direct_sources[0]["source_table"], "memory_digests")
        self.assertEqual(recursive_sources[0].source_table, "memory_digests")
        self.assertEqual(recursive_sources[1].source_table, "messages")
        self.assertEqual(recursive_sources[1].record["content"], "Raw cascade source message.")
        self.assertEqual(tree["sources"][0]["sources"][0]["record"]["content"], "Raw cascade source message.")

    def test_memory_digest_current_accessibility_excludes_superseded_and_system_only(self):
        store = self.seed_store()
        message_id = store.append_message("session-1", "CEO", "SE", "Source.")
        original_digest_id = store.append_memory_digest(
            "Original accessible digest.",
            [{"source_table": "messages", "source_id": message_id}],
            agent_id="SE",
            session_id="session-1",
        )
        replacement_digest_id = store.append_memory_digest(
            "Replacement accessible digest.",
            [{"source_table": "memory_digests", "source_id": original_digest_id}],
            agent_id="SE",
            session_id="session-1",
            supersedes_digest_ids=[original_digest_id],
        )
        system_digest_id = store.append_memory_digest(
            "System-only reanalysis digest.",
            [{"source_table": "memory_digests", "source_id": replacement_digest_id}],
            agent_id="SE",
            session_id="session-1",
            system_only=True,
        )

        all_digests = store.list_memory_digests(agent_id="SE")
        current_accessible = store.list_memory_digests(
            agent_id="SE",
            current_only=True,
            accessible_only=True,
        )
        original_digest = store.get_memory_digest(original_digest_id)
        system_digest = store.get_memory_digest(system_digest_id)

        self.assertEqual(
            {digest["digest_id"] for digest in all_digests},
            {original_digest_id, replacement_digest_id, system_digest_id},
        )
        self.assertEqual([digest["digest_id"] for digest in current_accessible], [replacement_digest_id])
        self.assertEqual(original_digest["superseded_by_digest_id"], replacement_digest_id)
        self.assertEqual(system_digest["system_only"], 1)
        self.assertEqual(system_digest["accessibility"], "system")

    def test_seed_identity_and_goal_digest_records(self):
        store = self.seed_store()

        seeded = store.seed_identity_and_goal_digests(
            "SE",
            "Identity: SE owns reliable runtime changes.",
            "Long-term goal: preserve agent lives.",
            "Short-term goal: keep memory compact and reanalyzable.",
            session_id="session-1",
        )

        identity_digest = store.get_memory_digest(seeded["identity"])
        long_goal_digest = store.get_memory_digest(seeded["goal_long_term"])
        short_goal_digest = store.get_memory_digest(seeded["goal_short_term"])
        identity_sources = store.expand_memory_digest_sources(seeded["identity"])
        short_goal_sources = store.expand_memory_digest_sources(seeded["goal_short_term"])

        self.assertEqual(identity_digest["digest_type"], "identity")
        self.assertEqual(long_goal_digest["digest_type"], "goal_long_term")
        self.assertEqual(short_goal_digest["digest_type"], "goal_short_term")
        self.assertEqual(identity_digest["generation"], 0)
        self.assertEqual(short_goal_digest["generation"], 0)
        self.assertEqual(identity_sources[0]["kind"], "identity_digest_seed")
        self.assertEqual(short_goal_sources[0]["kind"], "goal_short_term_digest_seed")

    def test_recursive_digest_expansion_can_return_raw_sources_without_digest_nodes(self):
        store = self.seed_store()
        thought_id = store.append_thought(
            "SE",
            "Raw thought remains available for automatic reanalysis.",
            session_id="session-1",
        )
        first_digest_id = store.append_memory_digest(
            "Digest of raw thought.",
            [{"source_table": "thoughts", "source_id": thought_id}],
            agent_id="SE",
            session_id="session-1",
        )
        second_digest_id = store.append_memory_digest(
            "Digest of digest.",
            [{"source_table": "memory_digests", "source_id": first_digest_id}],
            agent_id="SE",
            session_id="session-1",
        )

        raw_sources = store.expand_memory_digest_sources(
            second_digest_id,
            recursive=True,
            include_digest_records=False,
        )

        self.assertEqual(len(raw_sources), 1)
        self.assertEqual(raw_sources[0].source_table, "thoughts")
        self.assertEqual(
            raw_sources[0].record["content"],
            "Raw thought remains available for automatic reanalysis.",
        )

    def test_memory_digest_requires_sources(self):
        store = self.seed_store()

        with self.assertRaisesRegex(ValueError, "requires at least one source"):
            store.append_memory_digest("No source links.", [], agent_id="SE")

    def test_memory_digest_validates_sources_before_insert(self):
        store = self.seed_store()

        with self.assertRaisesRegex(ValueError, "source_table and source_id"):
            store.append_memory_digest(
                "Bad source link.",
                [{"source_table": "messages"}],
                agent_id="SE",
            )

        count = store.connection.execute(
            "SELECT COUNT(*) AS count FROM memory_digests"
        ).fetchone()["count"]

        self.assertEqual(count, 0)

    def test_memory_digest_source_metadata_must_be_object(self):
        store = self.seed_store()
        message_id = store.append_message("session-1", "CEO", "SE", "Source.")

        with self.assertRaisesRegex(ValueError, "metadata must be an object"):
            store.append_memory_digest(
                "Bad source metadata.",
                [
                    {
                        "source_table": "messages",
                        "source_id": message_id,
                        "metadata": False,
                    }
                ],
                agent_id="SE",
            )

    def test_memory_digest_rolls_back_partial_insert_on_source_error(self):
        store = self.seed_store()
        message_id = store.append_message("session-1", "CEO", "SE", "Source.")

        with self.assertRaises(sqlite3.IntegrityError):
            store.append_memory_digest(
                "Duplicate source links should roll back.",
                [
                    {"source_table": "messages", "source_id": message_id},
                    {"source_table": "messages", "source_id": message_id},
                ],
                agent_id="SE",
            )

        digest_count = store.connection.execute(
            "SELECT COUNT(*) AS count FROM memory_digests"
        ).fetchone()["count"]
        source_count = store.connection.execute(
            "SELECT COUNT(*) AS count FROM memory_digest_sources"
        ).fetchone()["count"]
        fts_count = store.connection.execute(
            "SELECT COUNT(*) AS count FROM memory_fts WHERE kind = 'memory_digest'"
        ).fetchone()["count"]

        self.assertEqual(digest_count, 0)
        self.assertEqual(source_count, 0)
        self.assertEqual(fts_count, 0)

    def test_legacy_memory_entry_digest_kind_is_rejected(self):
        store = self.seed_store()

        with self.assertRaisesRegex(ValueError, "Use append_memory_digest"):
            store.append_memory_entry(
                "memory_digest",
                "Ambiguous legacy digest.",
                agent_id="SE",
            )

    def test_runtime_events_can_be_persisted_and_filtered(self):
        store = self.seed_store()

        event_id = store.append_runtime_event(
            "session-1",
            "thought.recorded",
            payload={"agent": "SE", "content": "Private note."},
            visibility="private",
            sequence=1,
            created_at="2026-05-07T10:00:00+00:00",
        )

        events = store.list_runtime_events(
            session_id="session-1",
            event_type="thought.recorded",
            visibility="private",
        )

        self.assertEqual(events[0]["event_id"], event_id)
        self.assertEqual(events[0]["payload_json"], '{"agent": "SE", "content": "Private note."}')


if __name__ == "__main__":
    unittest.main()

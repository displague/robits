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

    def test_existing_digest_table_migrates_before_current_index_creation(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_path = Path(temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            """
            CREATE TABLE memory_digests (
                digest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                session_id TEXT,
                content TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                source_start_at TEXT,
                source_end_at TEXT,
                relationship_type TEXT,
                conversation_type TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        connection.close()

        store = SQLiteMemoryStore(db_path)
        self.addCleanup(store.close)

        digest_columns = {
            row["name"]
            for row in store.connection.execute("PRAGMA table_info(memory_digests)")
        }
        indexes = {
            row["name"]
            for row in store.connection.execute("PRAGMA index_list(memory_digests)")
        }

        self.assertIn("digest_type", digest_columns)
        self.assertIn("idx_memory_digests_current", indexes)

    def test_append_and_lookup_messages_by_session_and_agent(self):
        store = self.seed_store()

        message_id = store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Please design the durable memory substrate.",
            relationship_type="coworker",
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
            source="chat",
        )
        store.append_thought(
            "SE",
            "The sqlite schema needs FTS indexes for private recollection.",
            session_id="session-1",
            relationship_type="coworker",
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

    def test_search_filters_relationship_source_and_dates(self):
        store = self.seed_store()
        from robits.memory.sqlite import CHANNEL_WORK_PEER, SOCIAL_PROFESSIONAL

        work_channel = store.get_or_create_channel(
            CHANNEL_WORK_PEER, participants=["SE"], social_distance=SOCIAL_PROFESSIONAL
        )
        store.append_thought(
            "SE",
            "Coworker plan for sqlite search.",
            session_id="session-1",
            relationship_type="coworker",
            channel_id=work_channel,
            source="thinking",
            created_at="2026-05-07T10:00:00+00:00",
        )
        store.append_thought(
            "SE",
            "Family plan for sqlite search.",
            session_id="session-1",
            relationship_type="family",
            source="thinking",
            created_at="2026-05-08T10:00:00+00:00",
        )

        results = store.search(
            "sqlite",
            relationship_type="coworker",
            conversation_type=CHANNEL_WORK_PEER,
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
            source="chat",
            created_at="2026-05-07T10:00:00+00:00",
        )
        thought_id = store.append_thought(
            "SE",
            "The digest should link back to raw records.",
            session_id="session-1",
            relationship_type="coworker",
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

    def test_search_cascade_surfaces_parent_digest_for_inner_hits(self):
        store = self.seed_store()
        message_id = store.append_message(
            "session-1",
            "SE",
            "HR",
            "cascade inner content",
            created_at="2026-05-07T10:00:00+00:00",
        )
        digest_id = store.append_memory_digest(
            "Digest summarising the inner message.",
            [{"source_table": "messages", "source_id": message_id}],
            agent_id="SE",
            session_id="session-1",
        )

        # Direct FTS on inner content surfaces the message…
        direct = store.search("cascade inner content", agent_id="SE")
        # …but cascade should surface the parent digest at the outer level.
        cascade = store.search_cascade("cascade inner content", agent_id="SE")

        direct_kinds = {r.kind for r in direct}
        cascade_kinds = {r.kind for r in cascade}
        cascade_record_ids = {r.record_id for r in cascade}

        self.assertIn("message", direct_kinds)
        # The outer message hit is preserved…
        self.assertIn("message", cascade_kinds)
        # …and the parent digest is also surfaced.
        self.assertIn("memory_digest", cascade_kinds)
        self.assertIn(str(digest_id), cascade_record_ids)

    def test_search_cascade_deduplicates_digest_already_in_outer_hits(self):
        store = self.seed_store()
        message_id = store.append_message(
            "session-1",
            "SE",
            "HR",
            "unique term xyzzy",
            created_at="2026-05-07T10:00:00+00:00",
        )
        digest_id = store.append_memory_digest(
            "Digest with unique term xyzzy in content.",
            [{"source_table": "messages", "source_id": message_id}],
            agent_id="SE",
            session_id="session-1",
        )

        cascade = store.search_cascade("xyzzy", agent_id="SE")

        digest_hits = [r for r in cascade if r.kind == "memory_digest"]
        self.assertEqual(len(digest_hits), 1, "Digest should appear exactly once.")
        self.assertEqual(digest_hits[0].record_id, str(digest_id))

    def test_expand_memory_digest_sources_respects_max_depth(self):
        store = self.seed_store()
        message_id = store.append_message("session-1", "CEO", "SE", "Leaf message.")
        d1 = store.append_memory_digest(
            "Depth 1 digest.",
            [{"source_table": "messages", "source_id": message_id}],
            agent_id="SE",
            session_id="session-1",
        )
        d2 = store.append_memory_digest(
            "Depth 2 digest.",
            [{"source_table": "memory_digests", "source_id": d1}],
            agent_id="SE",
            session_id="session-1",
        )

        # max_depth=0 means don't recurse into child digests.
        depth0 = store.expand_memory_digest_sources(d2, recursive=True, max_depth=0)
        # max_depth=1 allows one level of recursion into d1 and its raw sources.
        depth1 = store.expand_memory_digest_sources(d2, recursive=True, max_depth=1)

        depth0_tables = [r.source_table for r in depth0]
        depth1_tables = [r.source_table for r in depth1]

        # At depth 0 we see the immediate source (d1 digest record).
        self.assertIn("memory_digests", depth0_tables)
        self.assertNotIn("messages", depth0_tables)

        # At depth 1 we also see the message inside d1.
        self.assertIn("memory_digests", depth1_tables)
        self.assertIn("messages", depth1_tables)

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


    def test_list_recent_message_ids_returns_chronological_order(self):
        store = self.build_store()
        store.create_session("s-order")
        store.upsert_agent("A", "Role", "A")
        store.upsert_agent("B", "Role", "B")
        id1 = store.append_message("s-order", "A", "B", "first")
        id2 = store.append_message("s-order", "B", "A", "second")
        id3 = store.append_message("s-order", "A", "B", "third")
        ids = store.list_recent_message_ids("s-order", 3)
        self.assertEqual(ids, [id1, id2, id3])

    def test_list_recent_message_ids_limits_to_last_n(self):
        store = self.build_store()
        store.create_session("s-limit")
        store.upsert_agent("A", "Role", "A")
        store.upsert_agent("B", "Role", "B")
        store.append_message("s-limit", "A", "B", "first")
        id2 = store.append_message("s-limit", "B", "A", "second")
        id3 = store.append_message("s-limit", "A", "B", "third")
        ids = store.list_recent_message_ids("s-limit", 2)
        self.assertEqual(ids, [id2, id3])

    def test_end_session_sets_ended_at(self):
        store = self.build_store()
        store.create_session("s-end")
        row = store.connection.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 's-end'"
        ).fetchone()
        self.assertIsNone(row["ended_at"])
        store.end_session("s-end")
        row = store.connection.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 's-end'"
        ).fetchone()
        self.assertIsNotNone(row["ended_at"])

    def test_end_session_accepts_explicit_timestamp(self):
        store = self.build_store()
        store.create_session("s-ts")
        store.end_session("s-ts", ended_at="2024-01-01T00:00:00")
        row = store.connection.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 's-ts'"
        ).fetchone()
        self.assertEqual(row["ended_at"], "2024-01-01T00:00:00")


class ChannelTests(unittest.TestCase):
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
        return store

    def test_schema_contains_channels_table(self):
        store = self.build_store()
        tables = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')"
            )
        }
        self.assertIn("channels", tables)

    def test_messages_and_thoughts_have_channel_id_column(self):
        store = self.build_store()
        for table in ("messages", "thoughts"):
            cols = {
                row["name"]
                for row in store.connection.execute(f"PRAGMA table_info({table})")
            }
            self.assertIn("channel_id", cols, f"{table} missing channel_id")

    def test_get_or_create_channel_returns_stable_id(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL

        id1 = store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
        id2 = store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
        self.assertEqual(id1, id2)
        self.assertIsInstance(id1, int)

    def test_get_or_create_channel_distinguishes_by_type_and_participants(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_AGENT_DM, CHANNEL_ORG_CHAT

        org_id = store.get_or_create_channel(CHANNEL_ORG_CHAT)
        dm_id = store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", "SE"])
        self.assertNotEqual(org_id, dm_id)

    def test_get_or_create_channel_normalizes_duplicate_participants(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_AGENT_DM

        id1 = store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["SE", "CEO", "SE"])
        id2 = store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", "SE"])
        row = store.connection.execute(
            "SELECT participants_json FROM channels WHERE channel_id = ?", (id1,)
        ).fetchone()

        self.assertEqual(id1, id2)
        self.assertEqual(row["participants_json"], '["CEO", "SE"]')

    def test_get_or_create_channel_rejects_non_string_participants(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_AGENT_DM

        with self.assertRaises(TypeError):
            store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", 3])

    def test_get_or_create_channel_rejects_out_of_range_social_distance(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_ORG_CHAT

        with self.assertRaises(ValueError):
            store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=6.0)
        with self.assertRaises(ValueError):
            store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=-1.0)

    def test_list_channels_filters_by_type(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_AGENT_DM, CHANNEL_ORG_CHAT

        store.get_or_create_channel(CHANNEL_ORG_CHAT)
        store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", "SE"])
        store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", "HR"])

        org_channels = store.list_channels(channel_type=CHANNEL_ORG_CHAT)
        dm_channels = store.list_channels(channel_type=CHANNEL_AGENT_DM)

        self.assertEqual(len(org_channels), 1)
        self.assertEqual(len(dm_channels), 2)

    def test_list_channels_filters_by_participant(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_AGENT_DM

        store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", "SE"])
        store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["CEO", "HR"])
        store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["SE", "HR"])

        se_channels = store.list_channels(participant="SE")
        self.assertEqual(len(se_channels), 2)

    def test_list_channels_participant_filter_handles_like_metacharacters(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_AGENT_DM

        exact_id = store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["agent_%"])
        store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["agent_12"])

        channels = store.list_channels(participant="agent_%")

        self.assertEqual(len(channels), 1)
        self.assertEqual(channels[0]["channel_id"], exact_id)

    def test_channel_id_is_stored_on_messages(self):
        store = self.seed_store()
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL

        channel_id = store.get_or_create_channel(
            CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL
        )
        msg_id = store.append_message(
            "session-1", "CEO", "SE", "hello via channel", channel_id=channel_id
        )
        row = store.connection.execute(
            "SELECT channel_id FROM messages WHERE message_id = ?", (msg_id,)
        ).fetchone()
        self.assertEqual(row["channel_id"], channel_id)

    def test_channel_id_is_stored_on_thoughts(self):
        store = self.seed_store()
        from robits.memory.sqlite import CHANNEL_AGENT_THOUGHT

        channel_id = store.get_or_create_channel(CHANNEL_AGENT_THOUGHT, participants=["SE"])
        thought_id = store.append_thought(
            "SE", "private thought in channel", session_id="session-1", channel_id=channel_id
        )
        row = store.connection.execute(
            "SELECT channel_id FROM thoughts WHERE thought_id = ?", (thought_id,)
        ).fetchone()
        self.assertEqual(row["channel_id"], channel_id)

    def test_channel_social_distance_is_persisted(self):
        store = self.build_store()
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL

        channel_id = store.get_or_create_channel(
            CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL
        )
        row = store.connection.execute(
            "SELECT social_distance FROM channels WHERE channel_id = ?", (channel_id,)
        ).fetchone()
        self.assertEqual(row["social_distance"], SOCIAL_PROFESSIONAL)

    def test_legacy_messages_without_channel_id_remain_accessible(self):
        store = self.seed_store()
        msg_id = store.append_message("session-1", "CEO", "SE", "no channel message")
        row = store.connection.execute(
            "SELECT channel_id FROM messages WHERE message_id = ?", (msg_id,)
        ).fetchone()
        self.assertIsNone(row["channel_id"])

    def test_existing_db_gains_channel_id_columns_on_open(self):
        import sqlite3 as _sqlite3

        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp_dir.cleanup)
        db_path = Path(temp_dir.name) / "legacy.sqlite3"

        legacy = _sqlite3.connect(db_path)
        legacy.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, title TEXT,
                started_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE agents (agent_id TEXT PRIMARY KEY, role TEXT NOT NULL,
                display_name TEXT, lifecycle_state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE messages (message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, sender_agent_id TEXT NOT NULL,
                receiver_agent_id TEXT, content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'message', visibility TEXT NOT NULL DEFAULT 'public',
                relationship_type TEXT, conversation_type TEXT, source TEXT,
                created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE thoughts (thought_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, agent_id TEXT NOT NULL, content TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private', relationship_type TEXT,
                conversation_type TEXT, source TEXT, created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}');
            """
        )
        legacy.close()

        store = SQLiteMemoryStore(db_path)
        self.addCleanup(store.close)

        for table in ("messages", "thoughts"):
            cols = {
                row["name"]
                for row in store.connection.execute(f"PRAGMA table_info({table})")
            }
            self.assertIn("channel_id", cols, f"migration failed for {table}")


if __name__ == "__main__":
    unittest.main()

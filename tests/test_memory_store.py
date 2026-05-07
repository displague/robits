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
        self.assertIn("memory_fts", tables)

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
            "memory_digest",
            "Digest: sqlite records should preserve source links.",
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
        self.assertIn("memory_digest", kinds)

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

        self.assertEqual(record_types, {"message", "thought", "tool_call"})


if __name__ == "__main__":
    unittest.main()

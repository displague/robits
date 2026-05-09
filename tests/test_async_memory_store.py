import asyncio
import tempfile
import unittest
from pathlib import Path

from robits.memory import AsyncSQLiteMemoryStore, SQLiteMemoryStore


class AsyncSQLiteMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def build_store(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = await AsyncSQLiteMemoryStore.open(Path(temp_dir.name) / "memory.sqlite3")
        self.addAsyncCleanup(store.close)
        return store

    async def seed_store(self):
        store = await self.build_store()
        await store.create_session("session-1", title="Planning")
        await store.upsert_agent("CEO", "Human", "CEO")
        await store.upsert_agent("SE", "SoftwareEngineer", "SE")
        await store.upsert_agent("HR", "HR", "HR")
        return store

    async def test_async_store_bootstraps_schema_and_writes_messages(self):
        store = await self.seed_store()

        message_id = await store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Please design async storage.",
            relationship_type="coworker",
            conversation_type="org_chat",
            source="chat",
        )

        messages = await store.list_messages(session_id="session-1", agent_id="SE")

        self.assertEqual(messages[0]["message_id"], message_id)
        self.assertEqual(messages[0]["content"], "Please design async storage.")
        self.assertEqual(messages[0]["conversation_type"], "org_chat")

    async def test_concurrent_async_appends_are_serialized_and_visible(self):
        store = await self.seed_store()

        async def append(index):
            return await store.append_message(
                "session-1",
                "CEO" if index % 2 == 0 else "HR",
                "SE",
                f"Concurrent message {index}.",
                conversation_type="org_chat",
                source="test",
            )

        ids = await asyncio.gather(*(append(index) for index in range(20)))
        messages = await store.list_messages(
            session_id="session-1",
            conversation_type="org_chat",
            limit=25,
        )

        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(len(messages), 20)
        self.assertEqual(
            {message["content"] for message in messages},
            {f"Concurrent message {index}." for index in range(20)},
        )

    async def test_runtime_events_can_be_read_while_writes_continue(self):
        store = await self.seed_store()

        async def append_events():
            for index in range(10):
                await store.append_runtime_event(
                    "session-1",
                    "message.created",
                    payload={"index": index},
                    sequence=index,
                )

        async def read_events():
            observed_counts = []
            for _ in range(5):
                rows = await store.list_runtime_events(session_id="session-1")
                observed_counts.append(len(rows))
                await asyncio.sleep(0)
            return observed_counts

        _, observed_counts = await asyncio.gather(append_events(), read_events())
        events = await store.list_runtime_events(session_id="session-1", limit=20)

        self.assertEqual(len(events), 10)
        self.assertEqual([event["sequence"] for event in events], list(range(10)))
        self.assertGreaterEqual(max(observed_counts), 0)

    async def test_async_and_sync_stores_can_share_file_backed_database(self):
        store = await self.seed_store()
        await store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Async write visible to sync reader.",
            conversation_type="org_chat",
        )
        await store.close()

        sync_store = SQLiteMemoryStore(store.path)
        self.addCleanup(sync_store.close)
        messages = sync_store.list_messages(session_id="session-1")

        self.assertEqual(messages[0]["content"], "Async write visible to sync reader.")

    async def test_memory_fts_is_updated_for_async_messages_and_thoughts(self):
        store = await self.seed_store()
        await store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Async sqlite message should be searchable.",
            conversation_type="org_chat",
        )
        await store.append_thought(
            "SE",
            "Async sqlite thought should also be searchable.",
            session_id="session-1",
            conversation_type="agent_thought",
        )
        await store.close()

        sync_store = SQLiteMemoryStore(store.path)
        self.addCleanup(sync_store.close)
        results = sync_store.search("searchable", session_id="session-1")
        kinds = {result.kind for result in results}

        self.assertIn("message", kinds)
        self.assertIn("thought", kinds)


if __name__ == "__main__":
    unittest.main()

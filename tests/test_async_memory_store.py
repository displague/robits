import asyncio
import inspect
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
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL
        store = await self.seed_store()
        ch = await store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)

        message_id = await store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Please design async storage.",
            relationship_type="coworker",
            channel_id=ch,
            source="chat",
        )

        messages = await store.list_messages(session_id="session-1", agent_id="SE")

        self.assertEqual(messages[0]["message_id"], message_id)
        self.assertEqual(messages[0]["content"], "Please design async storage.")
        self.assertEqual(messages[0]["channel_id"], ch)

    async def test_concurrent_async_appends_are_serialized_and_visible(self):
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL
        store = await self.seed_store()
        ch = await store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)

        async def append(index):
            return await store.append_message(
                "session-1",
                "CEO" if index % 2 == 0 else "HR",
                "SE",
                f"Concurrent message {index}.",
                channel_id=ch,
                source="test",
            )

        ids = await asyncio.gather(*(append(index) for index in range(20)))
        messages = await store.list_messages(
            session_id="session-1",
            channel_id=ch,
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
        # Reads see committed state only: counts must be non-decreasing
        self.assertEqual(observed_counts, sorted(observed_counts))
        self.assertLessEqual(max(observed_counts), 10)

    async def test_async_and_sync_stores_can_share_file_backed_database(self):
        store = await self.seed_store()
        await store.append_message(
            "session-1",
            "CEO",
            "SE",
            "Async write visible to sync reader.",
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
        )
        await store.append_thought(
            "SE",
            "Async sqlite thought should also be searchable.",
            session_id="session-1",
        )
        await store.close()

        sync_store = SQLiteMemoryStore(store.path)
        self.addCleanup(sync_store.close)
        results = sync_store.search("searchable", session_id="session-1")
        kinds = {result.kind for result in results}

        self.assertIn("message", kinds)
        self.assertIn("thought", kinds)

    async def test_async_store_exposes_sync_store_public_api_methods(self):
        sync_public = {
            name
            for name, member in inspect.getmembers(SQLiteMemoryStore, inspect.isfunction)
            if not name.startswith("_")
        }
        async_public = {
            name
            for name, member in inspect.getmembers(AsyncSQLiteMemoryStore, inspect.iscoroutinefunction)
            if not name.startswith("_")
        }
        self.assertTrue(sync_public.issubset(async_public))

    async def test_async_channel_create_and_list(self):
        from robits.memory.sqlite import CHANNEL_AGENT_DM, SOCIAL_FRIENDLY
        store = await self.seed_store()

        ch_id = await store.get_or_create_channel(
            CHANNEL_AGENT_DM,
            participants=["SE", "HR"],
            social_distance=SOCIAL_FRIENDLY,
        )
        self.assertIsInstance(ch_id, int)

        ch_id2 = await store.get_or_create_channel(
            CHANNEL_AGENT_DM,
            participants=["SE", "HR"],
            social_distance=SOCIAL_FRIENDLY,
        )
        self.assertEqual(ch_id, ch_id2, "idempotent: same args yield same id")

        channels = await store.list_channels(channel_type=CHANNEL_AGENT_DM)
        self.assertEqual(len(channels), 1)
        self.assertEqual(channels[0]["channel_id"], ch_id)
        self.assertAlmostEqual(channels[0]["social_distance"], SOCIAL_FRIENDLY)

        by_participant = await store.list_channels(participant="SE")
        self.assertEqual(len(by_participant), 1)
        self.assertEqual(by_participant[0]["channel_id"], ch_id)

    async def test_sync_parity_methods_delegate_through_async_store(self):
        store = await self.seed_store()
        await store.add_contact("SE", "HR", "coworker")

        todo_id = await store.append_todo(
            "SE",
            "Capture drop-in parity tasks.",
            session_id="session-1",
        )

        todos = await store.list_todos(agent_id="SE", session_id="session-1")
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0]["todo_id"], todo_id)
        self.assertEqual(todos[0]["title"], "Capture drop-in parity tasks.")

    async def test_channel_methods_delegate_through_async_store(self):
        from robits.memory.sqlite import CHANNEL_AGENT_DM, SOCIAL_PROFESSIONAL

        store = await self.seed_store()
        channel_id = await store.get_or_create_channel(
            CHANNEL_AGENT_DM,
            participants=["SE", "CEO", "SE"],
            social_distance=SOCIAL_PROFESSIONAL,
        )
        same_channel_id = await store.get_or_create_channel(
            CHANNEL_AGENT_DM,
            participants=["CEO", "SE"],
            social_distance=SOCIAL_PROFESSIONAL,
        )
        await store.get_or_create_channel(CHANNEL_AGENT_DM, participants=["agent_%"])

        se_channels = await store.list_channels(
            channel_type=CHANNEL_AGENT_DM,
            participant="SE",
        )
        wildcard_channels = await store.list_channels(participant="agent_%")

        self.assertEqual(channel_id, same_channel_id)
        self.assertEqual(len(se_channels), 1)
        self.assertEqual(se_channels[0]["channel_id"], channel_id)
        self.assertEqual(len(wildcard_channels), 1)


if __name__ == "__main__":
    unittest.main()

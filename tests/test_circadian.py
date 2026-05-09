"""Tests for circadian phase foundation (issue #73)."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import main
from robits.memory import SQLiteMemoryStore, compute_phase_shift


def _make_store():
    """Return a file-backed SQLiteMemoryStore with a temporary directory."""
    temp_dir = tempfile.TemporaryDirectory()
    store = SQLiteMemoryStore(Path(temp_dir.name) / "memory.sqlite3")
    return store, temp_dir


class CircadianSchemaTests(unittest.TestCase):
    def setUp(self):
        self.store, self.temp_dir = _make_store()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _agent_columns(self):
        return {
            row["name"]
            for row in self.store.connection.execute("PRAGMA table_info(agents)").fetchall()
        }

    def _message_columns(self):
        return {
            row["name"]
            for row in self.store.connection.execute("PRAGMA table_info(messages)").fetchall()
        }

    def test_agents_have_circadian_phase_column(self):
        self.assertIn("circadian_phase", self._agent_columns())

    def test_messages_have_sender_phase_column(self):
        self.assertIn("sender_phase", self._message_columns())


class AgentPhaseTests(unittest.TestCase):
    def setUp(self):
        self.store, self.temp_dir = _make_store()
        self.store.create_session("s1")
        self.store.upsert_agent("alice", "Human", "Alice")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_get_set_agent_phase_round_trip(self):
        self.store.set_agent_phase("alice", 0.75)
        self.assertAlmostEqual(self.store.get_agent_phase("alice"), 0.75)

    def test_set_agent_phase_clamps_above_one(self):
        self.store.set_agent_phase("alice", 1.5)
        self.assertAlmostEqual(self.store.get_agent_phase("alice"), 1.0)

    def test_set_agent_phase_clamps_below_zero(self):
        self.store.set_agent_phase("alice", -0.3)
        self.assertAlmostEqual(self.store.get_agent_phase("alice"), 0.0)

    def test_get_agent_phase_returns_none_for_missing_agent(self):
        self.assertIsNone(self.store.get_agent_phase("does_not_exist"))

    def test_new_agent_has_default_phase_of_0_5(self):
        phase = self.store.get_agent_phase("alice")
        self.assertAlmostEqual(phase, 0.5)


class ComputePhaseShiftTests(unittest.TestCase):
    def test_compute_phase_shift_moves_toward_event(self):
        # Self contact (distance=0) should fully shift to event_phase.
        result = compute_phase_shift(0.0, 1.0, 0.0)
        self.assertAlmostEqual(result, 1.0)

    def test_compute_phase_shift_self_contact_has_full_weight(self):
        # weight = 1/(0+1) = 1.0 => result = current + 1.0*(event - current) = event
        result = compute_phase_shift(0.2, 0.8, 0.0)
        self.assertAlmostEqual(result, 0.8)

    def test_compute_phase_shift_hostile_has_minimal_weight(self):
        # social_distance=5 => weight = 1/6
        current, event = 0.0, 0.6
        expected = 0.0 + (1.0 / 6.0) * (0.6 - 0.0)
        result = compute_phase_shift(current, event, 5.0)
        self.assertAlmostEqual(result, expected, places=10)

    def test_compute_phase_shift_clamps_to_unit_interval_high(self):
        result = compute_phase_shift(0.99, 1.0, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_compute_phase_shift_clamps_to_unit_interval_low(self):
        result = compute_phase_shift(0.01, 0.0, 0.0)
        self.assertGreaterEqual(result, 0.0)

    def test_compute_phase_shift_partial_weight(self):
        # social_distance=1 => weight = 0.5
        result = compute_phase_shift(0.0, 1.0, 1.0)
        self.assertAlmostEqual(result, 0.5)


class PhaseMigrationTests(unittest.TestCase):
    def test_phase_migration_on_legacy_db(self):
        """A legacy DB without circadian_phase column gets it added on open."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_path = Path(temp_dir.name) / "legacy.sqlite3"

        # Build a database that looks like the old schema (no circadian_phase).
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE agents (
                agent_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                display_name TEXT,
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        conn.commit()
        conn.close()

        # Opening the store should run the migration.
        store = SQLiteMemoryStore(db_path)
        self.addCleanup(store.close)

        columns = {
            row["name"]
            for row in store.connection.execute("PRAGMA table_info(agents)").fetchall()
        }
        self.assertIn("circadian_phase", columns)

        # Verify the column has the correct default (NULL for existing rows,
        # 0.5 for new inserts via DEFAULT).
        store.upsert_agent("bot", "Bot", "Bot")
        phase = store.get_agent_phase("bot")
        self.assertAlmostEqual(phase, 0.5)


class RecordTurnSenderPhaseTests(unittest.TestCase):
    """Integration: record_turn stamps sender_phase on messages."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "memory.sqlite3"
        self.store = SQLiteMemoryStore(db_path)
        self.original_store = main.memory_store
        self.original_digest_interval = main.memory_digest_interval
        self.original_org_digest_interval = main.org_digest_interval
        main.memory_store = self.store
        main.memory_digest_interval = 0
        main.org_digest_interval = 0

    def tearDown(self):
        main.memory_store = self.original_store
        main.memory_digest_interval = self.original_digest_interval
        main.org_digest_interval = self.original_org_digest_interval
        self.store.close()
        self.temp_dir.cleanup()

    def _make_session(self):
        participants = {
            "A": SimpleNamespace(
                name="A",
                lifecycle_state="active",
                interact=lambda *a, **kw: "hello",
                update_group_conversations=lambda m: None,
                runtime_tool_results=[],
            ),
            "B": SimpleNamespace(
                name="B",
                lifecycle_state="active",
                interact=lambda *a, **kw: "world",
                update_group_conversations=lambda m: None,
                runtime_tool_results=[],
            ),
        }
        system = SimpleNamespace(interact=lambda *a, **kw: None)
        scheduler = main.RoundRobinScheduler(list(participants))
        return main.Session(
            participants=participants,
            system=system,
            scheduler=scheduler,
        )

    def test_record_turn_stamps_sender_phase_on_messages(self):
        session = self._make_session()
        # Set known phases for agents.
        self.store.set_agent_phase("A", 0.3)
        self.store.set_agent_phase("B", 0.7)

        session.record_turn("A", "B", "hello prompt", "world response")

        rows = self.store.connection.execute(
            "SELECT sender_agent_id, sender_phase FROM messages WHERE session_id = ?",
            (session.run_id,),
        ).fetchall()
        phase_by_sender = {row["sender_agent_id"]: row["sender_phase"] for row in rows}

        # Prompt: sender=A, sender_phase should be A's phase (0.3)
        self.assertAlmostEqual(phase_by_sender["A"], 0.3)
        # Response: sender=B, sender_phase should be B's phase (0.7)
        self.assertAlmostEqual(phase_by_sender["B"], 0.7)

    def test_record_turn_shifts_receiver_phase_when_channel_set(self):
        session = self._make_session()
        # Set known phases.
        self.store.set_agent_phase("A", 0.2)
        self.store.set_agent_phase("B", 0.8)

        from robits.memory.sqlite import CHANNEL_AGENT_DM
        ch_id = self.store.get_or_create_channel(
            CHANNEL_AGENT_DM, participants=["A", "B"], social_distance=1.0
        )
        session._org_chat_channel_id = ch_id
        session.record_turn("A", "B", "hello", "world")

        # Receiver is B; sender is A with phase 0.2. social_distance=1 => weight=0.5
        # shifted = 0.8 + 0.5*(0.2 - 0.8) = 0.8 - 0.3 = 0.5
        shifted = self.store.get_agent_phase("B")
        self.assertAlmostEqual(shifted, 0.5)


if __name__ == "__main__":
    unittest.main()

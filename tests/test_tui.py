import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from robits.ui.app import RobitsDbReader, RobitsTuiApp


class TestRobitsTui(unittest.TestCase):
    def setUp(self):
        # Create a temporary database for testing DB reader methods
        self.db_fd, self.db_path = tempfile.mkstemp()
        os.close(self.db_fd)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create minimal schemas needed for tests
        self.cursor.executescript("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL
            );
            CREATE TABLE channels (
                channel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_type TEXT NOT NULL,
                participants_json TEXT NOT NULL DEFAULT '[]',
                visibility TEXT NOT NULL DEFAULT 'public'
            );
            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender_agent_id TEXT NOT NULL,
                receiver_agent_id TEXT,
                content TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'public',
                channel_id INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE thoughts (
                thought_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                channel_id INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE TABLE runtime_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sequence INTEGER,
                event_type TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'public',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE agents (
                agent_id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                display_name TEXT,
                username TEXT,
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
        """)
        self.conn.commit()
        self.reader = RobitsDbReader(self.db_path)

    def tearDown(self):
        self.conn.close()
        try:
            Path(self.db_path).unlink()
        except OSError:
            pass

    def test_list_recent_sessions(self):
        # Insert test sessions
        self.cursor.execute("INSERT INTO sessions VALUES (?, ?)", ("run-1", "2026-05-20T00:00:00Z"))
        self.cursor.execute("INSERT INTO sessions VALUES (?, ?)", ("run-2", "2026-05-20T01:00:00Z"))
        self.conn.commit()

        sessions = self.reader.list_recent_sessions()
        self.assertEqual(len(sessions), 2)
        # Should be ordered by started_at DESC
        self.assertEqual(sessions[0]["session_id"], "run-2")
        self.assertEqual(sessions[1]["session_id"], "run-1")

    def test_get_token_totals(self):
        # Insert token usage events for a session
        self.cursor.execute("INSERT INTO sessions VALUES (?, ?)", ("run-1", "2026-05-20T00:00:00Z"))
        payload_1 = json.dumps({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        payload_2 = json.dumps({"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})

        self.cursor.execute(
            "INSERT INTO runtime_events (session_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("run-1", "token_usage", payload_1, "2026-05-20T00:01:00Z")
        )
        self.cursor.execute(
            "INSERT INTO runtime_events (session_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("run-1", "token_usage", payload_2, "2026-05-20T00:02:00Z")
        )
        self.conn.commit()

        prompt, completion, total = self.reader.get_token_totals("run-1")
        self.assertEqual(prompt, 300)
        self.assertEqual(completion, 150)
        self.assertEqual(total, 450)

    def test_filter_channels_by_policy(self):
        # Test client-side channel filtering based on policy
        app = RobitsTuiApp(self.db_path, policy="full")
        channels = [
            {"channel_id": 1, "channel_type": "org_chat"},
            {"channel_id": 2, "channel_type": "agent_dm"},
            {"channel_id": 3, "channel_type": "agent_thought"},
        ]

        # full policy allows everything
        filtered = app.filter_channels_by_policy(channels)
        self.assertEqual(len(filtered), 3)

        # restricted policy hides thoughts
        app.policy = "restricted"
        filtered = app.filter_channels_by_policy(channels)
        self.assertEqual(len(filtered), 2)
        self.assertNotIn("agent_thought", [ch["channel_type"] for ch in filtered])

        # public-only policy only allows org_chat
        app.policy = "public-only"
        filtered = app.filter_channels_by_policy(channels)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["channel_type"], "org_chat")

    def test_format_channel_name(self):
        app = RobitsTuiApp(self.db_path)
        
        ch_org = {"channel_type": "org_chat", "participants_json": "[]", "channel_id": 1}
        ch_thought = {"channel_type": "agent_thought", "participants_json": '["CEO"]', "channel_id": 2}
        ch_dm = {"channel_type": "agent_dm", "participants_json": '["CEO", "Dev"]', "channel_id": 3}

        self.assertEqual(app.format_channel_name(ch_org), "# org-chat")
        self.assertEqual(app.format_channel_name(ch_thought), "💭 thoughts (CEO)")
        self.assertEqual(app.format_channel_name(ch_dm), "✉️ dm (CEO <-> Dev)")

    def test_get_agents(self):
        self.cursor.execute(
            "INSERT INTO agents (agent_id, role, display_name, username, created_at) VALUES (?, ?, ?, ?, ?)",
            ("SE-1", "SoftwareEngineer", "Jane Dev", "janedev", "2026-05-20T00:00:00Z")
        )
        self.conn.commit()

        agents = self.reader.get_agents()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_id"], "SE-1")
        self.assertEqual(agents[0]["role"], "SoftwareEngineer")

    def test_interactive_flag_assignment(self):
        app_passive = RobitsTuiApp(self.db_path, interactive=False)
        self.assertFalse(app_passive.interactive)

        app_active = RobitsTuiApp(self.db_path, interactive=True)
        self.assertTrue(app_active.interactive)


if __name__ == "__main__":
    unittest.main()

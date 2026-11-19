"""Durable private subtask records (issue #132): restart, reattach, interrupted work, expiry.

Temporary SQLite files and fake models only; no model service is needed.
"""
import json
import os
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robits.core.config import _config as config
from robits.core.tools import ToolRegistry
from robits.runtime.mcp import owner
from robits.runtime.subtasks import SubtaskManager, subtask_control
from robits.runtime.task_store import TaskStore
from tests.test_mcp_subtasks import message, role


def fake_client(*replies):
    replies, requests = list(replies), []

    def create(**kwargs):
        requests.append(kwargs)
        return replies.pop(0)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), requests


class DurableSubtaskTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "subtasks.db")
        self.registry = ToolRegistry()
        with open("tools.yaml", encoding="utf-8") as stream:
            for definition in yaml.safe_load(stream):
                self.registry.register_definition(definition)
        self.parent = role()
        self.now = [1000.0]
        self.boot()

    def boot(self, parent=None):
        """A fresh process: new store connection, new manager, same file."""
        self.parent = parent or self.parent
        if getattr(self, "store", None):
            self.store.close()   # (closing twice is harmless)
        self.store = TaskStore(self.path, retention_s=3600, clock=lambda: self.now[0])
        self.manager = SubtaskManager(self.store)
        self.addCleanup(self.store.close)
        scope = patch.multiple(config, tool_registry=self.registry, subtask_manager=self.manager,
                               active_tool_caller=self.parent, active_tool_caller_name="SE", memory_store=None,
                               mcp_config_path="", mcp_manager=None, subtask_max_depth=2, subtask_max_tasks=32)
        scope.start()
        self.addCleanup(scope.stop)

    def spawn(self, **kwargs):
        out = self.registry.execute("agent.spawn", {"task": "focused work", "defer": True, **kwargs}, {}, self.parent)
        return json.loads(out)["task_id"]

    def test_queued_task_survives_a_restart_and_runs(self):
        task_id = self.spawn()
        self.boot()
        client, requests = fake_client(message("Done after restart."))
        with patch.object(config, "client", client):
            done = json.loads(subtask_control({}, task_id, "run"))
        self.assertEqual((done["status"], done["result"]), ("completed", "Done after restart."))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["messages"][-1], {"role": "user", "content": "focused work"})

    def test_input_request_and_private_history_survive_and_resume(self):
        task_id = self.spawn()
        client, _ = fake_client(message(tool="agent__elicit", arguments={"message": "Which city?"}))
        with patch.object(config, "client", client):
            self.assertEqual(json.loads(subtask_control({}, task_id, "run"))["status"], "input_required")
        self.boot()
        pending = json.loads(subtask_control({}, task_id, "inspect"))
        self.assertEqual((pending["status"], pending["input_request"]), ("input_required", "Which city?"))
        subtask_control({}, task_id, "respond", "Paris")
        self.boot()
        client, requests = fake_client(message("Paris forecast."))
        with patch.object(config, "client", client):
            done = json.loads(subtask_control({}, task_id, "run"))
        self.assertEqual(done["result"], "Paris forecast.")
        self.assertEqual(requests[0]["messages"][-1], {"role": "user", "content": "Paris"})
        self.assertEqual(requests[0]["messages"][-2]["tool_call_id"], "call-1")

    def test_work_that_was_running_is_interrupted_and_never_repeated(self):
        task_id = self.spawn()
        record = self.manager.tasks[task_id]
        record["status"] = "running"
        self.manager.event(task_id, "running", {})   # the process stops here
        self.boot()
        client, requests = fake_client(message("must not be asked"))
        with patch.object(config, "client", client):
            view = json.loads(subtask_control({}, task_id, "run"))
            again = json.loads(subtask_control({}, task_id, "inspect"))
        self.assertEqual(view["status"], "interrupted")
        self.assertIn("not repeated", view["result"].lower())
        self.assertEqual(requests, [])
        self.assertIn("interrupted", [e["kind"] for e in again["events"]])
        self.boot()   # a second restart keeps it, without a second interrupted event
        kinds = [e["kind"] for e in json.loads(subtask_control({}, task_id, "inspect"))["events"]]
        self.assertEqual(kinds.count("interrupted"), 1)

    def test_an_interrupted_task_can_be_cancelled_and_stays_cancelled(self):
        task_id = self.spawn()
        self.manager.tasks[task_id]["status"] = "running"
        self.manager.event(task_id, "running", {})
        self.boot()
        self.assertEqual(json.loads(subtask_control({}, task_id, "inspect"))["status"], "interrupted")
        cancelled = json.loads(subtask_control({}, task_id, "cancel"))
        self.assertEqual(cancelled["status"], "cancelled")
        kinds = [e["kind"] for e in json.loads(subtask_control({}, task_id, "inspect"))["events"]]
        self.assertEqual(kinds.count("cancelled"), 1)
        self.boot()
        again = json.loads(subtask_control({}, task_id, "inspect"))
        self.assertEqual(again["status"], "cancelled")
        self.assertEqual([e["kind"] for e in again["events"]].count("interrupted"), 1)
        self.assertEqual(json.loads(subtask_control({}, task_id, "cancel"))["status"], "cancelled")

    def test_an_interrupted_parent_cancels_its_queued_children(self):
        parent_id = self.spawn()
        child = self.manager.tasks[parent_id]["role"]
        child.runtime_task_depth = 1
        with patch.multiple(config, active_tool_caller=child):
            child_id = json.loads(self.registry.execute(
                "agent.spawn", {"task": "nested", "defer": True}, {}, child))["task_id"]
        self.manager.tasks[parent_id]["status"] = "running"
        self.manager.event(parent_id, "running", {})
        self.boot()
        subtask_control({}, parent_id, "cancel")
        self.assertEqual(self.manager.tasks[parent_id]["status"], "cancelled")
        self.assertEqual(self.manager.tasks[child_id]["status"], "cancelled")

    def test_terminal_result_is_stable_across_restarts(self):
        task_id = self.spawn()
        client, _ = fake_client(message("Final."))
        with patch.object(config, "client", client):
            subtask_control({}, task_id, "run")
        self.boot()
        self.assertEqual(json.loads(subtask_control({}, task_id, "inspect"))["result"], "Final.")
        self.assertEqual(json.loads(subtask_control({}, task_id, "cancel"))["status"], "completed")

    def test_another_caller_cannot_reattach(self):
        task_id = self.spawn()
        self.boot(parent=role(name="Other", session="other-session"))
        self.assertTrue(subtask_control({}, task_id, "inspect").startswith("Error: Unknown subtask"))

    def test_recovery_cuts_tools_to_the_callers_current_grants(self):
        task_id = self.spawn(tools=["agent.think", "builtin.web_search"])
        self.assertIn("builtin.web_search", self.manager.tasks[task_id]["role"].allowed_tools)
        narrowed = role()
        narrowed.allowed_tools = {"agent.*"}
        self.boot(parent=narrowed)
        subtask_control({}, task_id, "inspect")
        recovered = self.manager.tasks[task_id]["role"].allowed_tools
        self.assertNotIn("builtin.web_search", recovered)
        self.assertIn("agent.think", recovered)

    def test_records_expire_and_nothing_secret_is_stored(self):
        task_id = self.spawn()
        mode = stat.S_IMODE(os.stat(self.path).st_mode)
        self.assertEqual(mode & 0o077, 0)
        with open(self.path, "rb") as stream:
            raw = stream.read()
        for live in (b"employee_dict", b"runtime_event_stream", b"api_key", b"Authorization"):
            self.assertNotIn(live, raw)
        self.now[0] += 3601
        self.boot()
        self.assertTrue(subtask_control({}, task_id, "inspect").startswith("Error: Unknown subtask"))

    def test_an_existing_loose_file_is_tightened_and_its_rows_kept(self):
        task_id = self.spawn()
        self.store.close()
        os.chmod(self.path, 0o644)
        self.boot()
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode) & 0o077, 0)
        rows = self.store.load_root(owner(self.parent))
        self.assertEqual([r["task_id"] for r in rows], [task_id])
        self.assertEqual(rows[0]["messages"][-1]["content"], "focused work")
        # An empty file made earlier with a looser mode is tightened too.
        fresh = os.path.join(self.dir.name, "empty.db")
        open(fresh, "w").close()
        os.chmod(fresh, 0o666)
        TaskStore(fresh).close()
        self.assertEqual(stat.S_IMODE(os.stat(fresh).st_mode), 0o600)

    def test_without_a_store_nothing_changes(self):
        manager = SubtaskManager()
        with patch.object(config, "subtask_manager", manager):
            task_id = json.loads(self.registry.execute(
                "agent.spawn", {"task": "x", "defer": True}, {}, self.parent))["task_id"]
        self.assertIn(task_id, manager.tasks)
        self.assertIsNone(manager.store)


if __name__ == "__main__":
    unittest.main()

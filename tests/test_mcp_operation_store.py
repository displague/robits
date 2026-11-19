"""Remote MCP operations survive a restart (issue #132): temporary SQLite and fake servers only."""
import json
import os
import tempfile
import unittest

from robits.runtime.mcp import MCPManager
from robits.runtime.task_store import TaskStore
from tests.test_mcp_subtasks import FakeTransport, role


class AuthTransport(FakeTransport):
    """A server with per-caller identities, as OAuth servers have."""
    auth = object()
    service_tools = None
    ident = "identity-a"

    def identity_for(self, caller):
        return self.ident

    def request(self, method, params, notify, identity=None):
        return super().request(method, params, notify)


class DurableMCPTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "tasks.db")
        self.now = 1000.0
        self.role = role()
        self.store = None
        self.boot()

    def boot(self, factory=FakeTransport, servers=None):
        """A fresh process: new store connection and manager over the same file."""
        if self.store:
            self.store.close()
        self.store = TaskStore(self.path, retention_s=3600, clock=lambda: self.now)
        self.addCleanup(self.store.close)
        self.manager = MCPManager(servers or {"example": {"url": "http://localhost/mcp"}}, factory, store=self.store)
        self.transport = self.manager.servers["example"]

    def start_task(self, **extra):
        self.transport.results = [{"resultType": "task", "taskId": "t", "status": "working", **extra}]
        return self.manager.call("example", "slow", {"q": 1}, self.role)["operation_id"]

    def test_task_handle_survives_a_restart_and_is_polled_never_recreated(self):
        handle = self.start_task()
        self.boot()
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed",
                                   "result": {"content": ["done"]}}]
        result = self.manager.advance(handle, self.role)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([m for m, _ in self.transport.calls], ["tasks/get"])
        self.assertEqual(self.transport.calls[0][1]["taskId"], "t")

    def test_cancel_of_an_input_continuation_is_remembered(self):
        self.transport.results = [{"resultType": "input_required", "requestState": "s", "inputRequests": {}}]
        handle = self.manager.call("example", "pay", {}, self.role)["operation_id"]
        self.assertEqual(self.manager.advance(handle, self.role, cancel=True)["status"], "cancelled")
        self.boot()
        self.assertEqual(self.manager.advance(handle, self.role)["status"], "cancelled")

    def test_another_caller_cannot_reattach(self):
        handle = self.start_task()
        self.boot()
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, role(name="Other", session="other"))
        self.assertEqual(self.transport.calls, [])

    def test_a_different_identity_at_the_server_cannot_reattach(self):
        self.boot(AuthTransport)
        handle = self.start_task()
        self.boot(AuthTransport)
        self.transport.ident = "identity-b"
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, self.role)
        self.assertEqual(self.transport.calls, [])
        self.transport.ident = "identity-a"
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed", "result": {}}]
        self.assertEqual(self.manager.advance(handle, self.role)["status"], "completed")

    def test_a_server_removed_from_the_configuration_is_refused(self):
        handle = self.start_task()
        self.boot(servers={"example": {"url": "http://localhost/mcp"}})
        del self.manager.servers["example"]
        with self.assertRaisesRegex(ValueError, "no longer configured"):
            self.manager.advance(handle, self.role)

    def raw(self):
        with open(self.path, "rb") as stream:
            return stream.read()

    def test_an_input_required_continuation_is_never_saved(self):
        self.transport.results = [{"resultType": "input_required", "requestState": "STATE-OPAQUE-77",
                                   "inputRequests": {"q": {"method": "elicitation/create",
                                                           "params": {"message": "Enter code QUESTION-SECRET-5"}}}}]
        handle = self.manager.call("example", "pay", {"note": "free text ARGUMENT-SECRET-9", "amount": 3},
                                   self.role)["operation_id"]
        for secret in (b"STATE-OPAQUE-77", b"QUESTION-SECRET-5", b"ARGUMENT-SECRET-9"):
            self.assertNotIn(secret, self.raw())
        self.assertIn(handle, self.manager.pending)                  # it still works in this process
        self.boot()
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, self.role)                  # and fails closed after a restart
        self.assertEqual(self.transport.calls, [])

    def test_a_task_is_saved_as_scalar_fields_only_and_resumes_by_polling(self):
        self.transport.results = [{"resultType": "task", "taskId": "t", "status": "input_required",
                                   "statusMessage": "STATUS-SECRET-1", "pollIntervalMs": 60000, "debug": "EXTRA-FIELD",
                                   "inputRequests": {"q": {"method": "elicitation/create",
                                                           "params": {"message": "Enter QUESTION-SECRET-2"}}}}]
        handle = self.manager.call("example", "slow", {"note": "ARGUMENT-SECRET-3"}, self.role)["operation_id"]
        for secret in (b"STATUS-SECRET-1", b"QUESTION-SECRET-2", b"ARGUMENT-SECRET-3", b"EXTRA-FIELD"):
            self.assertNotIn(secret, self.raw())
        self.boot()
        # Answering before the questions were fetched again is refused with a clear message.
        with self.assertRaisesRegex(ValueError, "ask for its status first"):
            self.manager.advance(handle, self.role, {"q": {"action": "decline"}})
        self.assertEqual(self.transport.calls, [])
        # The first ask after a restart polls at once, even inside the saved poll interval.
        question = {"q": {"method": "elicitation/create", "params": {"message": "Proceed?"}}}
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "input_required",
                                   "inputRequests": question}]
        self.manager.advance(handle, self.role)
        self.assertEqual([m for m, _ in self.transport.calls], ["tasks/get"])
        self.transport.results = [{"resultType": "complete"}]
        ack = self.manager.advance(handle, self.role, {"q": {"action": "decline"}})
        self.assertEqual(ack["acknowledged"], "tasks/update")

    def test_the_round_budget_is_preserved(self):
        handle = self.start_task(pollIntervalMs=60000)
        record = self.manager.pending[handle]
        record["rounds"] = 31
        self.manager._persist(handle, record, self.role)
        self.boot()
        self.assertEqual(self.manager._recover(handle, self.role)["rounds"], 31)

    def test_a_finished_operation_keeps_its_status_but_not_its_text(self):
        handle = self.start_task()
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed",
                                   "statusMessage": "STATUS-SECRET-4",
                                   "result": {"content": [{"type": "text", "text": "token=RESULT-SECRET-6"}]}}]
        first = self.manager.advance(handle, self.role)
        self.assertIn("RESULT-SECRET-6", json.dumps(first))          # this process still returns it
        for secret in (b"RESULT-SECRET-6", b"STATUS-SECRET-4"):
            self.assertNotIn(secret, self.raw())
        self.boot()
        again = self.manager.advance(handle, self.role)
        self.assertEqual((again["status"], again["taskId"]), ("completed", "t"))
        self.assertNotIn("result", again)
        self.assertIn("not kept across a restart", again["note"])
        self.assertEqual(self.transport.calls, [])
        self.now += 3601
        self.boot()
        with self.assertRaises(ValueError):
            self.manager.advance(handle, self.role)

    def test_a_finished_operation_is_reattached_under_the_same_checks(self):
        self.boot(AuthTransport)
        handle = self.start_task()
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed",
                                   "result": {"content": ["kept"]}}]
        self.manager.advance(handle, self.role)
        self.boot(AuthTransport)
        self.transport.ident = "identity-b"
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, self.role)
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, role(name="Other", session="other"))
        self.transport.ident = "identity-a"
        self.assertEqual(self.manager.advance(handle, self.role)["status"], "completed")
        del self.manager.servers["example"]
        with self.assertRaisesRegex(ValueError, "no longer configured"):
            self.manager.advance(handle, self.role)

    def test_only_the_server_name_is_stored_no_url_or_credentials(self):
        self.start_task()
        with open(self.path, "rb") as stream:
            raw = stream.read()
        self.assertNotIn(b"http://localhost/mcp", raw)
        self.assertIn(b"example", raw)

    def test_without_a_store_a_restart_forgets_operations_as_before(self):
        handle = self.start_task()
        manager = MCPManager({"example": {"url": "http://localhost/mcp"}}, FakeTransport)
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            manager.advance(handle, self.role)


if __name__ == "__main__":
    unittest.main()

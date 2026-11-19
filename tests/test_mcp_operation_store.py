"""Remote MCP operations survive a restart (issue #132): temporary SQLite and fake servers only."""
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

    def test_input_required_continuation_keeps_its_opaque_state(self):
        self.transport.results = [{"resultType": "input_required", "requestState": "opaque-state",
                                   "inputRequests": {"q": {"method": "elicitation/create",
                                                           "params": {"message": "Proceed?"}}}}]
        handle = self.manager.call("example", "pay", {"amount": 3}, self.role)["operation_id"]
        self.boot()
        self.transport.results = [{"resultType": "complete", "content": []}]
        self.manager.advance(handle, self.role, {"q": {"action": "decline"}})
        method, params = self.transport.calls[0]
        self.assertEqual(method, "tools/call")      # the continuation of that call, carrying its state
        self.assertEqual(params["requestState"], "opaque-state")
        self.assertEqual(params["arguments"], {"amount": 3})
        self.assertEqual(params["inputResponses"], {"q": {"action": "decline"}})

    def test_the_poll_interval_and_round_budget_are_preserved(self):
        handle = self.start_task(pollIntervalMs=60000)
        self.boot()
        early = self.manager.advance(handle, self.role)
        self.assertEqual(early["operation_id"], handle)
        self.assertEqual(self.transport.calls, [])        # not due yet: no request
        record = self.manager.pending[handle]
        self.assertEqual(record["rounds"], 0)
        self.manager.pending[handle]["rounds"] = 31
        self.manager._persist(handle, record, self.role)
        self.boot()
        self.assertEqual(self.manager._recover(handle, self.role)["rounds"], 31)

    def test_a_finished_operation_keeps_its_outcome_until_it_expires(self):
        handle = self.start_task()
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed",
                                   "result": {"content": ["final"]}}]
        first = self.manager.advance(handle, self.role)
        self.boot()
        self.assertEqual(self.manager.advance(handle, self.role), first)
        self.assertEqual(self.transport.calls, [])
        self.now += 3601
        self.boot()
        with self.assertRaises(ValueError):
            self.manager.advance(handle, self.role)

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

    def finish_one(self, **kwargs):
        handle = self.start_task()
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed", **kwargs}]
        self.manager.advance(handle, self.role)
        return handle

    def test_a_finished_operation_is_reattached_under_the_same_checks(self):
        self.boot(AuthTransport)
        handle = self.finish_one(result={"content": ["kept"]})
        self.boot(AuthTransport)
        self.transport.ident = "identity-b"
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, self.role)
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, role(name="Other", session="other"))
        self.transport.ident = "identity-a"
        self.assertEqual(self.manager.advance(handle, self.role)["result"], {"content": ["kept"]})
        del self.manager.servers["example"]
        with self.assertRaisesRegex(ValueError, "no longer configured"):
            self.manager.advance(handle, self.role)

    def test_a_task_is_saved_without_its_call_arguments_and_still_resumes(self):
        self.transport.results = [{"resultType": "task", "taskId": "t", "status": "working",
                                   "debug": "SERVER-EXTRA-FIELD"}]
        handle = self.manager.call("example", "slow", {"note": "PRIVATE-ARGUMENT"}, self.role)["operation_id"]
        with open(self.path, "rb") as stream:
            raw = stream.read()
        self.assertNotIn(b"PRIVATE-ARGUMENT", raw)       # a task is resumed by its ID, not its arguments
        self.assertNotIn(b"SERVER-EXTRA-FIELD", raw)     # only the resume fields of the answer are kept
        self.boot()
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "completed", "result": {}}]
        self.assertEqual(self.manager.advance(handle, self.role)["status"], "completed")

    def test_a_continuation_that_needs_a_secret_argument_is_not_saved(self):
        self.transport.results = [{"resultType": "input_required", "requestState": "s", "inputRequests": {}}]
        handle = self.manager.call("example", "pay", {"api_key": "SECRET-VALUE-9", "amount": 3}, self.role)["operation_id"]
        with open(self.path, "rb") as stream:
            self.assertNotIn(b"SECRET-VALUE-9", stream.read())
        self.assertIn(handle, self.manager.pending)      # still works in this process
        self.boot()
        with self.assertRaisesRegex(ValueError, "Unknown MCP operation"):
            self.manager.advance(handle, self.role)      # fails closed after a restart

    def test_configured_redact_keys_also_keep_a_continuation_out_of_the_store(self):
        servers = {"example": {"url": "http://localhost/mcp", "redact_keys": ["Account"]}}
        self.boot(servers=servers)
        self.transport.results = [{"resultType": "input_required", "requestState": "s", "inputRequests": {}}]
        self.manager.call("example", "pay", {"account": "ACCT-4455", "amount": 3}, self.role)
        with open(self.path, "rb") as stream:
            self.assertNotIn(b"ACCT-4455", stream.read())

    def test_a_finished_result_is_redacted_and_bounded(self):
        servers = {"example": {"url": "http://localhost/mcp", "redact_keys": ["balance"]}}
        self.boot(servers=servers)
        handle = self.finish_one(result={"content": ["ok"], "balance": 120, "session_token": "TOKEN-VALUE-7",
                                         "nested": [{"password": "PW-VALUE-3", "city": "Paris"}]})
        with open(self.path, "rb") as stream:
            raw = stream.read()
        for secret in (b"TOKEN-VALUE-7", b"PW-VALUE-3", b"120"):
            self.assertNotIn(secret, raw)
        self.boot(servers=servers)
        kept = self.manager.advance(handle, self.role)["result"]
        self.assertEqual(kept["balance"], "[redacted]")
        self.assertEqual(kept["session_token"], "[redacted]")
        self.assertEqual(kept["nested"], [{"password": "[redacted]", "city": "Paris"}])
        self.assertEqual(kept["content"], ["ok"])
        big = self.finish_one(result={"content": ["x" * 20000]})
        self.boot(servers=servers)
        self.assertIn("not kept", self.manager.advance(big, self.role)["note"])

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

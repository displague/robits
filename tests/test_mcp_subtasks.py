"""Protocol and delegation tests; no model server or external service required."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.providers import ChatCompletionsProvider
from robits.core.tools import ToolRegistry, role_can_use_tool
from robits.runtime.mcp import MCPManager, VERSION, HttpTransport, StdioTransport, header_paths, header_value
from robits.runtime.subtasks import SubtaskManager, subtask_control


class FakeTransport:
    def __init__(self, url):
        self.url, self.calls, self.results = url, [], []

    def request(self, method, params, notify):
        self.calls.append((method, params))
        notify({"method": "notifications/progress", "params": {"progress": 1}})
        return self.results.pop(0)


def role(name="SE", session="session"):
    return SimpleNamespace(name=name, runtime_role_name=name, runtime_session_id=session,
        capabilities={"mcp"}, allowed_tools={"agent.*", "mcp.*", "builtin.web_search"},
        employee_dict={}, max_tokens=200, temperature=0.1,
        runtime_event_stream=main.RuntimeEventStream(), runtime_tool_results=[])


def message(content=None, tool=None, arguments=None):
    calls = [] if tool is None else [SimpleNamespace(id="call-1", type="function",
        function=SimpleNamespace(name=tool, arguments=json.dumps(arguments or {})))]
    result = SimpleNamespace(content=content, tool_calls=calls)
    result.model_dump = lambda: {"role": "assistant", "content": content,
        "tool_calls": [{"id": c.id, "type": c.type, "function": {
            "name": c.function.name, "arguments": c.function.arguments}} for c in calls]}
    return SimpleNamespace(choices=[SimpleNamespace(message=result)])


class MCPTests(unittest.TestCase):
    def setUp(self):
        self.role = role()
        self.manager = MCPManager({"example": {"url": "http://localhost/mcp"}}, FakeTransport)
        self.transport = self.manager.servers["example"]
        self.registry = ToolRegistry()
        self.scope = patch.multiple(config, mcp_manager=self.manager, memory_store=None,
                                    mcp_config_path="", active_tool_caller=self.role)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def discover(self):
        schema = {"type": "object", "properties": {"city": {"type": "string"}},
                  "required": ["city"], "additionalProperties": False}
        self.transport.results = [{"tools": [{"name": "weather-lookup", "inputSchema": schema}]}]
        tools = self.registry.as_chat_completion_tools(self.role)
        self.assertEqual(tools[0]["function"]["parameters"], schema)
        self.assertLessEqual(len(tools[0]["function"]["name"]), 64)
        return tools[0]["function"]["name"]

    def test_native_schema_call_result_and_permissions(self):
        name = self.discover()
        self.transport.results = [{"content": [{"type": "text", "text": "sunny"}],
                                   "structuredContent": {"forecast": "sunny"}, "isError": False}]
        result = json.loads(self.registry.execute(name, {"city": "Paris"}, {}, caller=self.role))
        self.assertEqual(result["structuredContent"]["forecast"], "sunny")
        method, params = self.transport.calls[-1]
        self.assertEqual(method, "tools/call")
        self.assertEqual(params["name"], "weather-lookup")
        self.assertEqual(params["_meta"]["io.modelcontextprotocol/protocolVersion"], VERSION)
        denied = role("Other")
        denied.allowed_tools = set()
        self.assertTrue(self.registry.execute(name, {"city": "Paris"}, {}, denied).startswith("Error:"))
        self.assertTrue(self.registry.execute(name, {"city": 42}, {}, self.role).startswith("Error:"))
        self.assertEqual(len(self.transport.calls), 2)

    def test_native_provider_round_trip(self):
        name = self.discover()
        self.transport.results = [{"resultType": "complete", "content": [{"type": "text", "text": "sunny"}]}]
        results = [message(tool=name, arguments={"city": "Paris"}), message("Sunny.")]
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return results.pop(0)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(ChatCompletionsProvider(client, self.registry).generate(
            self.role, "fake", "CEO", [{"role": "user", "content": "weather?"}]), "Sunny.")
        self.assertEqual(calls[1]["messages"][-1]["tool_call_id"], "call-1")

    def test_mrtr_echo_scoping_and_form_schema(self):
        request = {"question": {"method": "elicitation/create", "params": {
            "message": "Which city?", "requestedSchema": {"type": "object", "properties": {
                "city": {"type": "string"}}, "required": ["city"]}}}}
        self.transport.results = [{"resultType": "input_required", "requestState": " opaque\nstate ",
                                   "inputRequests": request}]
        pending = self.manager.call("example", "weather", {}, self.role)
        self.assertNotIn("requestState", pending)
        handle = pending["operation_id"]
        for stranger in [role("Other"), role(session="other-session")]:
            with self.assertRaises(ValueError):
                self.manager.advance(handle, stranger, {})
        with self.assertRaises(ValueError):
            self.manager.advance(handle, self.role, {})
        from jsonschema import ValidationError
        with self.assertRaises(ValidationError):
            self.manager.advance(handle, self.role, {"question": {"action": "accept", "content": {"city": 2}}})
        self.transport.results = [{"resultType": "complete", "content": []}]
        self.manager.advance(handle, self.role, {"question": {"action": "accept", "content": {"city": "Paris"}}})
        self.assertEqual(self.transport.calls[-1][1]["requestState"], " opaque\nstate ")
        self.assertNotIn(handle, self.manager.pending)

    def test_task_poll_update_and_final_result(self):
        self.transport.results = [
            {"resultType": "task", "taskId": "t", "status": "working"},
            {"resultType": "complete", "taskId": "t", "status": "input_required", "inputRequests": {
                "q": {"method": "elicitation/create", "params": {"message": "Proceed?"}}}},
            {"resultType": "complete", "taskId": "t", "status": "completed", "result": {"content": []}}]
        handle = self.manager.call("example", "slow", {}, self.role)["operation_id"]
        self.manager.advance(handle, self.role)
        self.assertEqual(self.transport.calls[-1][0], "tasks/get")
        result = self.manager.advance(handle, self.role, {"q": {"action": "decline"}})
        self.assertEqual(self.transport.calls[-1][0], "tasks/update")
        self.assertEqual(result["status"], "completed")
        self.assertNotIn(handle, self.manager.pending)

    def test_pagination_and_repeated_cursor_rejected(self):
        self.transport.results = [{"tools": [], "nextCursor": "a"}, {"tools": [], "nextCursor": "a"}]
        with self.assertRaisesRegex(ValueError, "cursor repeated"):
            self.manager.register(self.registry, self.role)

    def test_remote_arguments_do_not_collide_with_runtime_parameter_names(self):
        schema = {"type": "object", "properties": {"employee_dict": {"type": "string"}},
                  "patternProperties": {"^extra_": {"type": "integer"}}, "additionalProperties": False}
        self.transport.results = [{"tools": [{"name": "raw", "inputSchema": schema}]}]
        name = self.registry.as_chat_completion_tools(self.role)[0]["function"]["name"]
        self.transport.results = [{"content": []}]
        result = self.registry.execute(name, {"employee_dict": "data", "extra_number": 1}, {}, self.role)
        self.assertFalse(result.startswith("Error:"), result)
        self.assertEqual(self.transport.calls[-1][1]["arguments"]["employee_dict"], "data")

    def test_cancel_remains_available_after_continuation_budget(self):
        self.transport.results = [{"resultType": "task", "taskId": "t", "status": "working"}]
        handle = self.manager.call("example", "slow", {}, self.role)["operation_id"]
        self.manager.pending[handle]["rounds"] = 32
        self.transport.results = [{"resultType": "complete", "taskId": "t", "status": "cancelled"}]
        self.assertEqual(self.manager.advance(handle, self.role, cancel=True)["status"], "cancelled")

    def test_task_acknowledgements_keep_handle_until_terminal_poll(self):
        self.transport.results = [
            {"resultType": "task", "taskId": "t", "status": "working"},
            {"resultType": "complete"},
            {"resultType": "complete", "taskId": "t", "status": "input_required", "inputRequests": {
                "q": {"method": "elicitation/create", "params": {"message": "Which option?"}}}},
            {"resultType": "complete"},
            {"resultType": "complete", "taskId": "t", "status": "completed", "result": {"content": []}},
        ]
        handle = self.manager.call("example", "slow", {}, self.role)["operation_id"]
        ack = self.manager.advance(handle, self.role, cancel=True)
        self.assertEqual(ack["acknowledged"], "tasks/cancel")
        self.assertEqual(ack["status"], "working")
        self.assertIn(handle, self.manager.pending)
        self.manager.advance(handle, self.role)
        ack = self.manager.advance(handle, self.role, {"q": {"action": "decline"}})
        self.assertEqual(ack["acknowledged"], "tasks/update")
        self.assertIn(handle, self.manager.pending)
        result = self.manager.advance(handle, self.role)
        self.assertEqual(result["status"], "completed")
        self.assertNotIn(handle, self.manager.pending)


class SubtaskTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.registry, self.manager, self.parent = ToolRegistry(), SubtaskManager(), role()
        with open("tools.yaml", encoding="utf-8") as stream:
            for definition in yaml.safe_load(stream):
                self.registry.register_definition(definition)
        self.scope = patch.multiple(config, tool_registry=self.registry, subtask_manager=self.manager,
            active_tool_caller=self.parent, active_tool_caller_name="SE", memory_store=None,
            mcp_config_path="", mcp_manager=None, subtask_max_depth=2, subtask_max_tasks=32)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def spawn(self, **kwargs):
        return self.registry.execute("agent.spawn", {"task": "focused work", "defer": True, **kwargs}, {}, self.parent)

    def test_private_work_stays_out_of_parent_conversation_and_console(self):
        from contextlib import redirect_stdout
        from io import StringIO
        parent = main.Role("Rowan", "You are Rowan, a warm and practical colleague.", {})
        parent.allowed_tools = {"agent.*"}
        parent.runtime_session_id = "persona-session"
        parent.runtime_event_stream = main.RuntimeEventStream()
        secret = "PRIVATE_WORKING_NOTE_7e1c"
        responses = [
            message(tool="agent__focus", arguments={"task": f"Check the share privately: {secret}"}),
            message(tool="agent__think", arguments={"content": secret}),
            message("The share is $47.23."),
            message("It's $47.23 each. I'll bring the thermos."),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: responses.pop(0))))
        output = StringIO()
        with patch.multiple(config, client=client, model_provider=ChatCompletionsProvider(client, self.registry)), redirect_stdout(output):
            reply = parent.interact("CEO", "What do we each owe for the order?")
        self.assertEqual(reply, "It's $47.23 each. I'll bring the thermos.")
        self.assertEqual(output.getvalue(), "")
        history = parent.conversation_history[parent.name]
        self.assertEqual([item["role"] for item in history], ["system", "user", "assistant"])
        self.assertNotIn(secret, json.dumps(history))
        events = parent.runtime_event_stream.events()
        internal = [e for e in events if e.event_type.startswith(("tool_call.", "subtask."))]
        self.assertTrue(internal)
        self.assertTrue(all(e.visibility == "private" for e in internal))
        task = next(iter(self.manager.tasks.values()))
        self.assertEqual(task["result"], "The share is $47.23.")
        self.assertNotIn("warm and practical colleague", json.dumps(task["messages"]))

    def test_surface_private_note_events_do_not_enter_public_observer_stream(self):
        from robits.core.providers import _execute_tool_call
        _execute_tool_call(self.registry, self.parent, "agent__think", "private-note",
                           {"content": "A private reminder."}, {})
        events = self.parent.runtime_event_stream.events()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.visibility == "private" for event in events))

    def test_two_levels_and_denial_at_execution_boundary(self):
        first = json.loads(self.spawn())["task_id"]
        child = self.manager.tasks[first]["role"]
        second = json.loads(self.registry.execute("agent.spawn", {"task": "nested", "defer": True}, {}, child))["task_id"]
        grandchild = self.manager.tasks[second]["role"]
        self.assertEqual(grandchild.runtime_task_depth, 2)
        self.assertFalse(role_can_use_tool(grandchild, self.registry.get("agent.spawn")))
        self.assertTrue(self.registry.execute("agent.spawn", {"task": "forbidden"}, {}, grandchild).startswith("Error:"))
        self.assertFalse(role_can_use_tool(grandchild, self.registry.get("agent.focus")))
        self.assertTrue(self.registry.execute("agent.focus", {"task": "forbidden"}, {}, grandchild).startswith("Error:"))
        self.assertIn(second, self.manager.view(self.manager.tasks[first])["children"])

    def test_focus_exposes_only_a_task_argument_and_preserves_spawn_options(self):
        tools = self.registry.as_chat_completion_tools(self.parent)
        focus = next(t["function"] for t in tools if t["function"]["name"] == "agent__focus")
        spawn = next(t["function"] for t in tools if t["function"]["name"] == "agent__spawn")
        self.assertEqual(set(focus["parameters"]["properties"]), {"task"})
        self.assertTrue({"tools", "model", "defer"}.issubset(spawn["parameters"]["properties"]))

    def test_no_grant_escalation_and_context_isolation(self):
        task = self.manager.tasks[json.loads(self.spawn(tools=["builtin.shell_run", "builtin.web_search"]))["task_id"]]
        self.assertEqual(task["role"].allowed_tools, {"builtin.web_search"})
        self.assertEqual(len(task["messages"]), 2)
        self.assertEqual(task["messages"][-1]["content"], "focused work")
        with self.assertRaises(ValueError):
            self.manager.get(task["task_id"], role("Other"))
        with self.assertRaises(ValueError):
            self.manager.get(task["task_id"], role(session="other"))

    def test_pause_inspect_respond_resume_with_native_tool_history(self):
        task_id = json.loads(self.spawn())["task_id"]
        results = [message(tool="agent__elicit", arguments={"message": "Which city?"}), message("Paris forecast.")]
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            return results.pop(0)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(config, "client", client):
            pending = json.loads(subtask_control({}, task_id, "run"))
            self.assertEqual(pending["status"], "input_required")
            self.assertEqual(pending["input_request"], "Which city?")
            self.assertNotIn("events", pending)
            self.assertEqual(len(requests), 1)
            subtask_control({}, task_id, "respond", "Paris")
            completed = json.loads(subtask_control({}, task_id, "run"))
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["result"], "Paris forecast.")
        self.assertEqual(requests[1]["messages"][-1], {"role": "user", "content": "Paris"})
        self.assertEqual(requests[1]["messages"][-2]["tool_call_id"], "call-1")
        self.assertNotIn("events", completed)
        inspected = json.loads(subtask_control({}, task_id, "inspect"))
        self.assertIn("tool_result", [event["kind"] for event in inspected["events"]])

    def test_creation_returns_compact_handle_and_explicit_inspection_has_events(self):
        created = json.loads(self.spawn())
        self.assertNotIn("events", created)
        inspected = json.loads(subtask_control({}, created["task_id"], "inspect"))
        self.assertEqual(inspected["events"][0]["kind"], "created")

    def test_cancel_descendants_and_task_budget(self):
        task_id = json.loads(self.spawn())["task_id"]
        child = self.manager.tasks[task_id]["role"]
        nested = json.loads(self.registry.execute("agent.spawn", {"task": "nested", "defer": True}, {}, child))["task_id"]
        subtask_control({}, task_id, "cancel")
        self.assertEqual(self.manager.tasks[nested]["status"], "cancelled")
        with patch.object(config, "subtask_max_tasks", 2):
            self.assertIn("budget", self.spawn())

    def test_cancellation_preserves_terminal_descendant_results(self):
        task_id = json.loads(self.spawn())["task_id"]
        child = self.manager.tasks[task_id]["role"]
        nested = json.loads(self.registry.execute("agent.spawn", {"task": "nested", "defer": True}, {}, child))["task_id"]
        self.manager.tasks[nested].update(status="completed", result="verified result")
        subtask_control({}, task_id, "cancel")
        self.assertEqual(self.manager.tasks[nested]["status"], "completed")
        self.assertEqual(self.manager.tasks[nested]["result"], "verified result")

    def test_cancel_terminal_parent_still_cancels_queued_descendants(self):
        task_id = json.loads(self.spawn())["task_id"]
        child = self.manager.tasks[task_id]["role"]
        nested = json.loads(self.registry.execute("agent.spawn", {"task": "nested", "defer": True}, {}, child))["task_id"]
        self.manager.tasks[task_id].update(status="completed", result="delegated")
        subtask_control({}, task_id, "cancel")
        self.assertEqual(self.manager.tasks[task_id]["status"], "completed")
        self.assertEqual(self.manager.tasks[nested]["status"], "cancelled")

    def test_intermediate_parent_can_inspect_deeper_descendants(self):
        with patch.object(config, "subtask_max_depth", 3):
            first = json.loads(self.spawn())["task_id"]
            child = self.manager.tasks[first]["role"]
            second = json.loads(self.registry.execute("agent.spawn", {"task": "nested", "defer": True}, {}, child))["task_id"]
            grandchild = self.manager.tasks[second]["role"]
            third = json.loads(self.registry.execute("agent.spawn", {"task": "deeper", "defer": True}, {}, grandchild))["task_id"]
            self.assertEqual(self.manager.get(third, child)["task_id"], third)
            sibling = self.manager.tasks[json.loads(self.spawn())["task_id"]]["role"]
            with self.assertRaises(ValueError):
                self.manager.get(third, sibling)

    def test_depth_zero_disables_spawn(self):
        with patch.object(config, "subtask_max_depth", 0):
            self.assertTrue(self.spawn().startswith("Error:"))

    def test_event_inspection_is_paginated_with_bounded_previews(self):
        record = self.manager.tasks[json.loads(self.spawn())["task_id"]]
        for index in range(20):
            self.manager.event(record["task_id"], "tool_result", {"result": "x" * 3000})
        view = self.manager.view(record)
        self.assertEqual(len(view["events"]), 10)
        self.assertEqual(view["event_count"], 21)
        self.assertEqual(len(view["events"][-1]["result"]), 2048)
        first = self.manager.view(record, after_sequence=-1, limit=2)
        second = self.manager.view(record, after_sequence=first["next_sequence"], limit=2)
        self.assertEqual([e["sequence"] for e in first["events"] + second["events"]], [0, 1, 2, 3])


class TransportTests(unittest.TestCase):
    def test_complete_json_call_with_server_stripped_end_tag(self):
        from robits.core.providers import _extract_xml_tool_calls
        clean, calls = _extract_xml_tool_calls('<tool_call>{"name":"agent_spawn","arguments":{"task":"sum"}}')
        self.assertEqual(calls, [{"name": "agent_spawn", "arguments": {"task": "sum"}}])
        self.assertEqual(clean, "")
        _, calls = _extract_xml_tool_calls('<tool_call>{"name":"agent_spawn","arguments":')
        self.assertEqual(calls, [])

    def test_real_http_json_sse_and_metadata_headers(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((body, dict(self.headers)))
                response = {"jsonrpc": "2.0", "id": body["id"], "result": {"resultType": "complete", "content": []}}
                if body["method"] == "tools/list":
                    payload, content_type = json.dumps(response), "application/json"
                else:
                    notification = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
                    payload = "data: " + json.dumps(notification) + "\n\ndata: " + json.dumps(response) + "\n\n"
                    content_type = "text/event-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload.encode())))
                self.end_headers()
                self.wfile.write(payload.encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            transport = HttpTransport(f"http://127.0.0.1:{server.server_port}/mcp", timeout=2)
            transport.tool_headers["weather"] = [("Region", ("where", "region"))]
            manager = MCPManager({})
            manager.servers["fixture"] = transport
            active_role, notifications = role(), []
            transport.request("tools/list", {}, notifications.append)
            manager.call("fixture", "weather", {"where": {"region": "line1\nline2"}}, active_role)
            body, headers = requests[-1]
            lower = {k.lower(): v for k, v in headers.items()}
            self.assertEqual(lower["mcp-method"], "tools/call")
            self.assertEqual(lower["mcp-name"], "weather")
            self.assertEqual(lower["mcp-protocol-version"], VERSION)
            self.assertEqual(lower["mcp-param-region"], header_value("line1\nline2"))
            self.assertEqual(body["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"], VERSION)
            self.assertEqual(len(active_role.runtime_event_stream.events()), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_stdio_rpc_and_timeout_cleanup(self):
        import sys
        script = ('import json,sys; r=json.loads(sys.stdin.readline()); '
                  'print(json.dumps({"jsonrpc":"2.0","id":r["id"],"result":{"resultType":"complete"}}),flush=True)')
        self.assertEqual(StdioTransport(sys.executable, ["-u", "-c", script], timeout=2).request(
            "tools/list", {}, lambda item: None), {"resultType": "complete"})
        with self.assertRaises(TimeoutError):
            StdioTransport(sys.executable, ["-u", "-c", "import time; time.sleep(60)"], timeout=0.1).request(
                "tools/list", {}, lambda item: None)

    def test_stdio_deadline_includes_blocked_stdin_write(self):
        import sys
        import time
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            StdioTransport(sys.executable, ["-u", "-c", "import time; time.sleep(60)"], timeout=0.1).request(
                "tools/call", {"large": "x" * (2 * 1024 * 1024)}, lambda item: None)
        self.assertLess(time.monotonic() - start, 3)

    @unittest.skipUnless(__import__('os').name == "nt", "Windows job-object cleanup regression")
    def test_stdio_reaps_descendant_after_original_process_exits(self):
        import sys
        import ctypes
        from ctypes import wintypes as w
        script = ('import subprocess,sys,json,os; r=json.loads(sys.stdin.readline()); '
                  'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                  'print(json.dumps({"jsonrpc":"2.0","id":r["id"],"result":{"pid":p.pid}}),flush=True); '
                  'os._exit(0)')
        result = StdioTransport(sys.executable, ["-u", "-c", script], timeout=2).request("tools/list", {}, lambda item: None)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
        kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, result["pid"])
        if handle:
            try:
                self.assertEqual(kernel.WaitForSingleObject(handle, 1000), 0)
            finally:
                kernel.CloseHandle(handle)

    def test_http_total_deadline_covers_slow_headers_and_bodies(self):
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                try:
                    if self.path == "/headers":
                        payload = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    else:
                        content_type = "text/event-stream" if self.path == "/sse" else "application/json"
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", "1000")
                        self.end_headers()
                        payload = b"data: " if self.path == "/sse" else b'{"result":'
                    self.wfile.write(payload)
                    for _ in range(100):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.01)
                except (OSError, ConnectionError):
                    pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for path in ("headers", "json", "sse"):
                with self.subTest(path=path):
                    start = time.monotonic()
                    with self.assertRaises(TimeoutError):
                        HttpTransport(f"http://127.0.0.1:{server.server_port}/{path}", timeout=0.1).request("tools/list", {}, lambda item: None)
                    self.assertLess(time.monotonic() - start, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_header_encoding_and_invalid_annotations(self):
        self.assertEqual(header_value(True), "true")
        self.assertTrue(header_value("=?base64?literal?=").startswith("=?base64?"))
        self.assertTrue(header_value(" padded ").startswith("=?base64?"))
        self.assertEqual(header_paths({"properties": {"region": {
            "type": "string", "x-mcp-header": "Region"}}}), [("Region", ("region",))])
        for schema in [{"properties": {"p": {"type": "number", "x-mcp-header": "P"}}},
                       {"anyOf": [{"properties": {"p": {"type": "string", "x-mcp-header": "P"}}}]},
                       {"properties": {"p": {"type": "string", "x-mcp-header": "bad\nheader"}}}]:
            with self.assertRaises(ValueError):
                header_paths(schema)


if __name__ == "__main__":
    unittest.main()

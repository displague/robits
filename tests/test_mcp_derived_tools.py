"""Derived MCP tools: flat tools backed by another tool on the same server (e.g. a generic execute)."""
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.tools import ToolRegistry
from robits.runtime.mcp import MCPManager, registry_tool_name

POST_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"channel": {"type": "string"}, "text": {"type": "string"},
                               "priority": {"type": "string"}},
                "required": ["channel", "text"]}
DERIVED = {"post": {"via": "execute", "arguments": {"name": "post_update"}, "wrap": "arguments",
                     "idempotency_key": "idempotency_key", "description": "Post a status update.",
                     "input_schema": POST_SCHEMA}}


class FakeTransport:
    def __init__(self, url=None, **kwargs):
        self.calls = []

    def request(self, method, params, notify):
        if method == "tools/list":
            return {"tools": [{"name": "execute", "inputSchema": {"type": "object", "properties": {
                "name": {"type": "string"}, "arguments": {"type": "object"}}}}]}
        self.calls.append(params)
        return {"structuredContent": {"ok": True, "echo": params["arguments"]}}


class DerivedToolTests(unittest.TestCase):
    def setUp(self):
        self.transport = FakeTransport()
        self.manager = MCPManager({"tracker": {"url": "http://tracker/mcp", "derived_tools": DERIVED}},
                                  transport_factory=lambda **o: self.transport)
        self.role = SimpleNamespace(name="sam", capabilities={"mcp"})
        self.registry = ToolRegistry()
        self.manager.register(self.registry, self.role)
        self.name = registry_tool_name("tracker", "post")
        scope = patch.multiple(config, active_tool_caller=self.role, memory_store=None)
        scope.start()
        self.addCleanup(scope.stop)

    def run_tool(self, **arguments):
        return self.registry._tools[self.name].func({}, arguments)

    def test_registered_with_the_flat_schema(self):
        tool = self.registry._tools[self.name]
        self.assertEqual(tool.parameters, POST_SCHEMA)
        self.assertIn(registry_tool_name("tracker", "execute"), self.registry)

    def test_call_wraps_arguments_with_fixed_name_and_idempotency_key(self):
        result = json.loads(self.run_tool(channel="ops", text="A happened"))
        sent = self.transport.calls[-1]
        self.assertEqual(sent["name"], "execute")
        self.assertEqual(sent["arguments"]["name"], "post_update")
        inner = sent["arguments"]["arguments"]
        self.assertEqual((inner["channel"], inner["text"]), ("ops", "A happened"))
        self.assertTrue(inner["idempotency_key"].startswith("robits-"))
        self.assertTrue(result["structuredContent"]["ok"])

    def test_idempotency_key_is_deterministic_per_arguments(self):
        self.run_tool(channel="ops", text="A")
        self.run_tool(channel="ops", text="A")
        self.run_tool(channel="ops", text="B")
        keys = [c["arguments"]["arguments"]["idempotency_key"] for c in self.transport.calls]
        self.assertEqual(keys[0], keys[1], "a retry with the same arguments can't duplicate")
        self.assertNotEqual(keys[0], keys[2])

    def test_invalid_arguments_are_refused_before_the_call(self):
        self.assertTrue(self.run_tool(channel="ops").startswith("Error:"))
        self.assertTrue(self.run_tool(channel="ops", text="x", extra=1).startswith("Error:"))
        self.assertEqual(self.transport.calls, [])

    def test_not_registered_without_the_backing_tool(self):
        manager = MCPManager({"tracker": {"url": "http://tracker/mcp", "derived_tools": {**DERIVED, "other": {
            **DERIVED["post"], "via": "missing_tool"}}}}, transport_factory=lambda **o: FakeTransport())
        registry = ToolRegistry()
        manager.register(registry, self.role)
        self.assertIn(registry_tool_name("tracker", "post"), registry)
        self.assertNotIn(registry_tool_name("tracker", "other"), registry)

    def test_bad_specs_are_rejected(self):
        for bad in ({"x": {"input_schema": POST_SCHEMA}}, {"x": {"via": "execute"}},
                    {"x": {"via": "execute", "input_schema": {"type": "string"}}}, {"bad name!": DERIVED["post"]},
                    {"x": {**DERIVED["post"], "wrap": 3}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MCPManager({"tracker": {"url": "http://tracker/mcp", "derived_tools": bad}},
                           transport_factory=lambda **o: FakeTransport())

    def test_persona_can_list_only_the_derived_tool(self):
        with patch.object(config, "memory_store", None):
            org = main.build_employee_dict({"sam": {"role": "SE", "full_name": "Sam", "entries": [],
                                                     "tools": ["mcp:tracker.post"]}}, headless=True)
        sam = org["sam"]
        self.assertEqual(sam.tool_ceiling, (self.name,))
        offered = [t for t in self.registry._tools.values() if t.name in sam.allowed_tools]
        self.assertEqual([t.name for t in offered], [self.name], "the generic execute is not offered")

    def test_unwrapped_arguments_cannot_override_fixed_fields(self):
        open_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        manager = MCPManager({"tracker": {"url": "http://tracker/mcp", "derived_tools": {"post": {
            "via": "execute", "arguments": {"name": "post_update"}, "description": "x",
            "input_schema": open_schema}}}}, transport_factory=lambda **o: self.transport)
        registry = ToolRegistry()
        manager.register(registry, self.role)
        call = registry._tools[registry_tool_name("tracker", "post")].func
        self.assertIn("may not set name", call({}, {"text": "hi", "name": "delete_all"}))
        self.assertEqual(self.transport.calls, [])
        call({}, {"text": "hi"})
        self.assertEqual(self.transport.calls[-1]["arguments"], {"name": "post_update", "text": "hi"})

    def test_model_cannot_choose_the_idempotency_key(self):
        loose = {**POST_SCHEMA, "additionalProperties": True}
        manager = MCPManager({"tracker": {"url": "http://tracker/mcp", "derived_tools": {"post": {
            **DERIVED["post"], "input_schema": loose}}}}, transport_factory=lambda **o: self.transport)
        registry = ToolRegistry()
        manager.register(registry, self.role)
        result = registry._tools[registry_tool_name("tracker", "post")].func(
            {}, {"channel": "ops", "text": "x", "idempotency_key": "mine"})
        self.assertIn("may not set idempotency_key", result)

    def test_key_differs_across_sessions_and_personas(self):
        def key(role):
            with patch.object(config, "active_tool_caller", role):
                self.run_tool(channel="ops", text="A")
            return self.transport.calls[-1]["arguments"]["arguments"]["idempotency_key"]
        base = key(SimpleNamespace(name="sam", capabilities={"mcp"}, runtime_session_id="s1"))
        self.assertEqual(base, key(SimpleNamespace(name="sam", capabilities={"mcp"}, runtime_session_id="s1")))
        self.assertNotEqual(base, key(SimpleNamespace(name="sam", capabilities={"mcp"}, runtime_session_id="s2")))
        self.assertNotEqual(base, key(SimpleNamespace(name="kim", capabilities={"mcp"}, runtime_session_id="s1")))

    def test_alias_shadowing_a_remote_tool_is_refused(self):
        manager = MCPManager({"tracker": {"url": "http://tracker/mcp", "derived_tools": {"execute": DERIVED["post"]}}},
                             transport_factory=lambda **o: FakeTransport())
        with self.assertRaises(ValueError):
            manager.register(ToolRegistry(), self.role)


if __name__ == "__main__":
    unittest.main()

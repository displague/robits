"""Personas are told which of their tools are unavailable; tool calls reach the JSONL log."""
import io
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.io import JsonlLogger, set_logger
from robits.core.providers import ChatCompletionsProvider
from robits.core.tools import ToolDefinition
from robits.runtime.mcp import registry_tool_name
from tests.test_persona_tools import chat_client, chat_message


class UnavailableToolTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        scope = patch.multiple(config, memory_store=None, max_context_tokens=0, mcp_manager=None,
                               mcp_config_path="",
                               model_provider=SimpleNamespace(generate=lambda r, m, s, msgs: self.requests.append(msgs) or "ok"))
        scope.start()
        self.addCleanup(scope.stop)
        with redirect_stdout(StringIO()):
            self.org = main.build_employee_dict({"ada": {"role": "SE", "full_name": "Ada", "entries": [],
                                                          "tools": ["mcp:hub.brief", "agent.think"]}}, headless=True)
            main.tool_registry.clear()
            main.load_tools(main.System(self.org))
        self.addCleanup(main.tool_registry.clear)

    def system(self):
        with redirect_stdout(StringIO()):
            self.org["ada"].interact("ops", "hello")
        return self.requests[-1][0]["content"]

    def test_missing_mcp_tool_is_named_by_its_readable_name(self):
        prompt = self.system()
        self.assertIn("Unavailable right now: hub.brief.", prompt)
        self.assertIn("do not describe results you did not get", prompt)
        self.assertNotIn("agent.think.", prompt.split("Unavailable right now:")[1].split("\n")[0])

    def test_no_note_once_the_tool_is_registered(self):
        name = registry_tool_name("hub", "brief")
        main.tool_registry._tools[name] = ToolDefinition(name, "brief", {"type": "object", "properties": {}},
                                                         [], [], lambda *a: "ok", namespace="mcp")
        self.assertNotIn("Unavailable right now", self.system())


class ToolCallLogTests(unittest.TestCase):
    def test_tool_calls_are_logged_without_arguments(self):
        log = io.StringIO()
        set_logger(JsonlLogger(log))
        self.addCleanup(set_logger, None)
        registry = SimpleNamespace(
            as_chat_completion_tools=lambda role: [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            execute=lambda name, args, ed, caller=None: "found it",
            resolve_name=lambda n: n)
        client, _ = chat_client([chat_message(calls=[("lookup", json.dumps({"q": "private query text"}))]),
                                 chat_message("done")])
        role = SimpleNamespace(name="ada", max_tokens=100, temperature=0.1, employee_dict={})
        with patch.object(config, "memory_store", None):
            ChatCompletionsProvider(client, registry).generate(role, "m", "ops", [])
        entries = [json.loads(line) for line in log.getvalue().splitlines()]
        calls = [e for e in entries if e.get("type") == "tool_call"]
        self.assertEqual([(c["agent"], c["tool"], c["status"]) for c in calls], [("ada", "lookup", "ok")])
        self.assertNotIn("private query text", log.getvalue())
        self.assertNotIn("found it", log.getvalue())


if __name__ == "__main__":
    unittest.main()

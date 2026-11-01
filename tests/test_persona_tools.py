"""Per-persona tool allowlists, budgets and round caps; no model server required."""
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.persona import load_personas
from robits.core.providers import ChatCompletionsProvider, ResponsesProvider
from robits.core.tool_functions import grant_tool_access
from robits.runtime.mcp import MCPManager, registry_tool_name

WEATHER = registry_tool_name("example", "weather-lookup")


class FakeTransport:
    def __init__(self, url):
        self.url, self.calls = url, []

    def request(self, method, params, notify):
        self.calls.append((method, params))
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        if method == "tools/list":
            return {"tools": [{"name": "weather-lookup", "inputSchema": schema},
                              {"name": "weather-alerts", "inputSchema": schema}]}
        return {"content": [{"type": "text", "text": "sunny"}]}


def chat_message(content=None, calls=()):
    """Build a fake chat completion; calls is a list of (name, raw_arguments_json)."""
    tool_calls = [SimpleNamespace(id=f"call-{i}", type="function",
                                  function=SimpleNamespace(name=name, arguments=arguments))
                  for i, (name, arguments) in enumerate(calls)]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    message.model_dump = lambda: {"role": "assistant", "content": content, "tool_calls": [
        {"id": c.id, "type": c.type, "function": {"name": c.function.name, "arguments": c.function.arguments}}
        for c in tool_calls]}
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def chat_client(responses):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


class PersonaToolTests(unittest.TestCase):
    def setUp(self):
        main.tool_registry.clear()
        self.manager = MCPManager({"example": {"url": "http://localhost/mcp"}}, FakeTransport)
        self.scope = patch.multiple(config, mcp_manager=self.manager, mcp_config_path="",
                                    memory_store=None, active_tool_caller=None)
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.addCleanup(main.tool_registry.clear)

    def build(self, **personas):
        persona_map = {name: {"role": "SE", "full_name": name, "entries": [], **info}
                       for name, info in personas.items()}
        employee_dict = main.build_employee_dict(persona_map)
        with redirect_stdout(StringIO()):
            main.load_tools(main.System(employee_dict))
        return employee_dict

    def generate(self, role, responses):
        client, calls = chat_client(list(responses))
        reply = ChatCompletionsProvider(client, main.tool_registry).generate(
            role, "fake", "CEO", [{"role": "user", "content": "hello"}])
        return reply, calls

    def test_configured_tools_are_exactly_the_schemas_sent(self):
        employee_dict = self.build(small={"tools": ["memory.search", "agent.think", "mcp:example.weather-lookup"]})
        role = employee_dict["small"]
        reply, calls = self.generate(role, [chat_message("Done.")])
        self.assertEqual(reply, "Done.")
        sent = [tool["function"]["name"] for tool in calls[0]["tools"]]
        self.assertEqual(sent, ["memory__search", "agent__think", WEATHER.replace(".", "__")])

        responses = []
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: (
            responses.append(kwargs), SimpleNamespace(id="r1", output=[], output_text="Done."))[1]))
        ResponsesProvider(client, main.tool_registry).generate(role, "fake", "CEO", [])
        self.assertEqual([tool["name"] for tool in responses[0]["tools"]], sent)

    def test_persona_file_tool_settings_reach_the_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "personas.yaml"
            path.write_text(
                "- username: small\n  role: SE\n  tools: [memory.search, 'agent.*']\n"
                "  tool_budget: 3\n  max_tool_rounds: 2\n  memories: []\n"
                "- username: plain\n  role: SE\n  memories: []\n")
            employee_dict = main.build_employee_dict(load_personas(str(path)))
        small, plain = employee_dict["small"], employee_dict["plain"]
        self.assertEqual(small.allowed_tools, {"memory.search", "agent.*"})
        self.assertEqual((small.tool_budget, small.max_tool_rounds), (3, 2))
        default = main.SoftwareEngineer({}).allowed_tools
        self.assertEqual(plain.allowed_tools, default)
        self.assertIsNone(plain.tool_ceiling)

    def test_malformed_tool_settings_skip_the_persona(self):
        with self.assertLogs(level="WARNING"):
            employee_dict = main.build_employee_dict({
                "bad": {"role": "SE", "tools": "memory.search"},
                "worse": {"role": "SE", "max_tool_rounds": 0},
                "vague": {"role": "SE", "tools": ["mcp:example.*"]},
            })
        self.assertNotIn("bad", employee_dict)
        self.assertNotIn("worse", employee_dict)
        self.assertNotIn("vague", employee_dict)

    def test_bad_calls_get_a_corrective_message_and_the_turn_continues(self):
        employee_dict = self.build(small={"tools": ["memory.search", "agent.think"]})
        reply, calls = self.generate(employee_dict["small"], [
            chat_message(calls=[("builtin__shell_run", "{}"), ("no_such_tool", "{}"),
                                ("agent__think", "{not json"), ("agent__think", '{"bogus": 1}')]),
            chat_message("Recovered."),
        ])
        self.assertEqual(reply, "Recovered.")
        results = [m["content"] for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(results), 4)
        for content in results:
            self.assertTrue(content.startswith("Error:"), content)
            self.assertIn("Tools you may use: memory__search, agent__think.", content)
        self.assertIn("not allowed", results[0])
        self.assertIn("not found", results[1])

    def test_default_persona_errors_are_unchanged(self):
        employee_dict = self.build(plain={})
        _, calls = self.generate(employee_dict["plain"], [
            chat_message(calls=[("no_such_tool", "{}")]), chat_message("Ok.")])
        result = calls[1]["messages"][-1]["content"]
        self.assertEqual(result, "Error: Tool 'no_such_tool' not found.")

    def test_tool_budget_and_round_cap(self):
        employee_dict = self.build(small={"tools": ["agent.think", "memory.*"],
                                          "tool_budget": 2, "max_tool_rounds": 2})
        role = employee_dict["small"]
        looping = [chat_message(calls=[("no_such_tool", "{}")]) for _ in range(5)]
        reply, calls = self.generate(role, looping)
        self.assertEqual(reply, "Error: Too many consecutive tool-call rounds.")
        self.assertEqual(len(calls), 2)
        self.assertEqual([tool["function"]["name"] for tool in calls[0]["tools"]],
                         ["agent__think", "memory__search"])
        # Tools beyond the budget stay granted; only their schemas are withheld.
        self.assertIn("memory.expand_digest", [t.name for t in main.tool_registry.tools_for_role(role)])

        plain = self.build(plain={})["plain"]
        reply, calls = self.generate(plain, [chat_message(calls=[("no_such_tool", "{}")])
                                             for _ in range(config.max_tool_rounds + 1)])
        self.assertEqual(len(calls), config.max_tool_rounds)

    def test_grants_cannot_widen_configured_tools_without_an_operator(self):
        employee_dict = self.build(small={"tools": ["memory.*"]})
        hr = main.HR(employee_dict)
        with patch.object(config, "active_tool_caller", hr):
            denied = grant_tool_access(employee_dict, "small", "builtin.web_search", granted_by="HR")
            within = grant_tool_access(employee_dict, "small", "memory.search", granted_by="HR")
        self.assertIn("only an operator", denied)
        self.assertIn("Granted", within)
        self.assertNotIn("builtin.web_search", employee_dict["small"].allowed_tools)
        with patch.object(config, "active_tool_caller", main.Ops(employee_dict)):
            widened = grant_tool_access(employee_dict, "small", "builtin.web_search", granted_by="Ops")
        self.assertIn("Granted", widened)
        self.assertIn("builtin.web_search", employee_dict["small"].allowed_tools)

        plain = self.build(plain={})
        with patch.object(config, "active_tool_caller", hr):
            self.assertIn("Granted", grant_tool_access(plain, "plain", "builtin.file_search"))


if __name__ == "__main__":
    unittest.main()

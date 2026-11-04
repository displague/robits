"""Prompt profiles and tool-aware instructions for small local models."""
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.providers import ChatCompletionsProvider, ResponsesProvider
from robits.core.roles import _apply_persona_tools


class PromptTestCase(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def generate(role, model, sender, messages):
            self.requests.append(messages)
            return "ok"
        scope = patch.multiple(config, memory_store=None, max_context_tokens=0, prompt_profile="standard",
                               model_provider=SimpleNamespace(generate=generate))
        scope.start()
        self.addCleanup(scope.stop)
        with redirect_stdout(StringIO()):
            self.org = main.build_employee_dict()
            main.tool_registry.clear()
            main.load_tools(main.System(self.org))
        self.addCleanup(main.tool_registry.clear)

    def system_prompt(self, role, profile=None, tools=None):
        if tools is not None:
            _apply_persona_tools(role, {"tools": tools})
        if profile:
            role.prompt_profile = profile
        role.runtime_clock_state = "on"
        with redirect_stdout(StringIO()):
            role.interact("CEO", "hello")
        return self.requests[-1][0]["content"]


class ToolAwareInstructionTests(PromptTestCase):
    def test_standard_prompt_mentions_only_offered_tools(self):
        full = self.system_prompt(self.org["SE"])
        for text in ("agent.focus", "memory.search", "agent.wait", "IMPORTANT CODE RULES"):
            self.assertIn(text, full)
        lean = self.system_prompt(main.build_employee_dict()["SE"], tools=["agent.think"])
        self.assertIn("SILENT THINKING", lean)
        for text in ("agent.focus", "memory.search", "WAITING, SLEEPING", "IMPORTANT CODE RULES"):
            self.assertNotIn(text, lean)
        self.assertIn("1. ROLE BOUNDARIES", lean)
        self.assertIn("2. SILENT THINKING", lean)

    def test_tool_authoring_notes_need_tools_propose(self):
        se = main.build_employee_dict()["SE"]
        self.assertIn("IMPORTANT CODE RULES", self.system_prompt(se, tools=["tools.propose"]))
        self.assertNotIn("IMPORTANT CODE RULES", self.system_prompt(self.org["HR"]))


class CompactProfileTests(PromptTestCase):
    def test_compact_prompt_is_short_and_has_no_json_state(self):
        standard = self.system_prompt(main.build_employee_dict()["SE"], tools=["agent.wait", "memory.search"])
        compact = self.system_prompt(main.build_employee_dict()["SE"], profile="compact",
                                     tools=["agent.wait", "memory.search"])
        self.assertLess(len(compact), len(standard) * 0.7)
        self.assertIn("If the answer is there, give it", compact)
        self.assertIn("call agent.wait", compact)
        self.assertIn("call memory.search", compact)
        self.assertNotIn("agent.think", compact)
        state = compact[compact.index("State:"):]
        self.assertNotIn("agent_name", state)
        self.assertNotIn("{", state)
        self.assertIn("State: on the clock; work focus", compact)

    def test_env_profile_applies_without_persona_setting(self):
        with patch.object(config, "prompt_profile", "compact"):
            prompt = self.system_prompt(main.build_employee_dict()["SE"], tools=["agent.wait"])
        self.assertIn("Reply to the latest message briefly", prompt)


class PersonaModelSettingTests(unittest.TestCase):
    def test_settings_apply_and_bad_values_skip_the_persona(self):
        persona_map = {
            "small": {"role": "SE", "full_name": "small", "entries": [], "prompt_profile": "compact",
                      "reasoning_effort": "none", "max_tokens": 200},
            "bad": {"role": "SE", "full_name": "bad", "entries": [], "reasoning_effort": "maximum"},
        }
        with patch.object(config, "memory_store", None), self.assertLogs(level="WARNING") as logs:
            org = main.build_employee_dict(persona_map, headless=True)
        small = org["small"]
        self.assertEqual((small.prompt_profile, small.reasoning_effort, small.max_tokens), ("compact", "none", 200))
        self.assertNotIn("bad", org)
        self.assertTrue(any("invalid model settings" in line for line in logs.output))


class ReasoningEffortForwardingTests(unittest.TestCase):
    def test_chat_completions_forwards_reasoning_effort_none(self):
        sent = []

        def create(**kwargs):
            sent.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        registry = SimpleNamespace(as_chat_completion_tools=lambda role: [])
        role = SimpleNamespace(name="SE", max_tokens=100, temperature=0.1, reasoning_effort="none", employee_dict={})
        ChatCompletionsProvider(client, registry).generate(role, "m", "CEO", [])
        self.assertEqual(sent[0]["reasoning_effort"], "none")
        role.reasoning_effort = None
        with patch.object(config, "_reasoning_effort_env", None):
            ChatCompletionsProvider(client, registry).generate(role, "m", "CEO", [])
        self.assertNotIn("reasoning_effort", sent[1])

    def test_responses_accepts_none(self):
        sent = []
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: sent.append(kw) or SimpleNamespace(
            id="r1", output=[], output_text="ok")))
        role = SimpleNamespace(name="SE", max_tokens=100, temperature=0.1, reasoning_effort="none", employee_dict={})
        ResponsesProvider(client, SimpleNamespace(as_responses_tools=lambda r: []), stateful=False).generate(
            role, "m", "CEO", [])
        self.assertEqual(sent[0]["reasoning"], {"effort": "none"})


if __name__ == "__main__":
    unittest.main()

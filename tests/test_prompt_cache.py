"""The system message keeps its stable part first, so local servers can reuse their prompt cache."""
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config

STATE = "=== Persona Circadian Rhythm & State ==="


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.requests = []

        def generate(role, model, sender, messages):
            self.requests.append([dict(m) for m in messages])
            return f"reply {len(self.requests)}"
        scope = patch.multiple(config, memory_store=None, max_context_tokens=0,
                               model_provider=SimpleNamespace(generate=generate))
        scope.start()
        self.addCleanup(scope.stop)
        self.se = main.build_employee_dict()["SE"]
        self.se.runtime_clock_state = "on"

    def turn(self, text, clock="on"):
        self.se.runtime_clock_state = clock
        with redirect_stdout(StringIO()):
            self.se.interact("CEO", text)

    def test_per_turn_state_comes_after_the_stable_prompt(self):
        self.turn("first question")
        time.sleep(1.1)  # the runtime context's timestamp changes between turns
        self.turn("second question")
        s1, s2 = self.requests[0][0]["content"], self.requests[1][0]["content"]
        self.assertNotEqual(s1, s2, "the timestamp changed")
        stable1, stable2 = s1[:s1.index(STATE)], s2[:s2.index(STATE)]
        self.assertEqual(stable1, stable2, "everything before the state section is identical")
        self.assertIn("current_datetime_utc", s1[s1.index(STATE):])
        self.assertNotIn("Circadian phase", stable1)
        self.assertGreater(len(stable1), len(s1) - len(stable1), "most of the system prompt is cacheable")

    def test_user_messages_are_untouched(self):
        self.turn("first question")
        self.assertEqual(self.requests[0][-1]["content"], "first question")
        self.assertEqual(self.se.conversation_history[self.se.name][-2]["content"], "first question")


if __name__ == "__main__":
    unittest.main()

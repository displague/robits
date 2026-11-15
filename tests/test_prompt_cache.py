"""The system message keeps its stable part first, so local servers can reuse their prompt cache."""
import json
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config

STATE = "=== Persona Circadian Rhythm & State ==="
STATE_LABEL = "(Context for you, not part of the message above:)"


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

    def test_the_system_message_does_not_change_between_turns(self):
        self.turn("first question")
        time.sleep(1.1)  # the runtime context's timestamp changes between turns
        self.turn("second question")
        s1, s2 = self.requests[0][0]["content"], self.requests[1][0]["content"]
        self.assertEqual(s1, s2, "clock, phase and time stay out of the system message")
        self.assertNotIn(STATE, s1)
        self.assertNotIn("current_datetime_utc", s1)

    def test_state_is_sent_after_the_newest_message_only(self):
        self.turn("first question")
        self.turn("second question")
        first, second = self.requests[0][-1]["content"], self.requests[1][-1]["content"]
        self.assertTrue(first.startswith("first question\n\n" + STATE_LABEL + "\n" + STATE))
        self.assertTrue(second.startswith("second question\n\n" + STATE_LABEL + "\n" + STATE))
        self.assertIn("current_datetime_utc", second)
        earlier = [m["content"] for m in self.requests[1][1:-1]]
        self.assertEqual(earlier, ["first question", "reply 1"], "earlier turns are sent as they were")

    def test_user_messages_are_stored_untouched(self):
        self.turn("first question")
        self.assertEqual(self.se.conversation_history[self.se.name][-2]["content"], "first question")

    def test_trimming_keeps_its_starting_point_until_the_budget_is_reached_again(self):
        config.max_context_tokens = 1000  # ~4000 characters: the system message and several turns
        for i in range(20):
            self.turn(f"question {i} " + "x" * 200)
        starts = [next(m["content"] for m in r if m["role"] == "user") for r in self.requests]
        firsts = [s[s.index("question"):].split()[1] if "question" in s else s for s in starts]
        changes = sum(1 for a, b in zip(firsts, firsts[1:]) if a != b)
        self.assertGreater(len(set(firsts)), 1, "old turns were dropped")
        self.assertLess(changes, 5, "the starting point moves in steps, not on every turn")
        for r in self.requests:
            self.assertLessEqual(sum(len(json.dumps(m)) for m in r), 4000 + 1500)  # + the newest message's state


if __name__ == "__main__":
    unittest.main()

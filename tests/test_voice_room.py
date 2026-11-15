"""robits.voice without audio: line files, speakable replies, and who answers on a call."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import main
from robits.core.config import _config
from robits.voice import lines, speech
from robits.voice.room import Room

PERSONAS = {
    "alex_chen": {"role": "SE", "full_name": "Alex Chen", "voice": "nova"},
    "sam_ortiz": {"role": "SE", "full_name": "Sam Ortiz"},
}


class FakeRole:
    def __init__(self, key, full_name, responses=None):
        self.name = key
        self.full_name = full_name
        self.first_name, self.last_name = full_name.split(" ", 1)
        self.responses = list(responses or [])
        self.received = []
        self.alarms = []
        self.waiting_until = None

    def interact(self, sender=None, prompt=None):
        self.received.append((sender, prompt))
        return self.responses.pop(0) if self.responses else ""


class LineFileTest(unittest.TestCase):
    def test_group_and_persona_lines(self):
        parsed = lines.parse({"default_voice": "alloy", "lines": {
            "team": {"mode": "group", "personas": ["alex_chen", "sam_ortiz"], "greeting": "Team."},
            "editor": {"persona": "sam_ortiz"},
        }}, PERSONAS)
        self.assertTrue(parsed["team"].group)
        self.assertEqual(parsed["editor"].personas, ("sam_ortiz",))
        self.assertEqual(parsed["team"].voices, {"alex_chen": "nova", "sam_ortiz": "alloy"})

    def test_malformed_lines_are_refused(self):
        for bad in (
            {},
            {"lines": {}},
            {"lines": {"x": {"mode": "conference", "persona": "alex_chen"}}},
            {"lines": {"x": {"persona": "nobody"}}},
            {"lines": {"x": {"mode": "group", "personas": "alex_chen"}}},
            {"lines": {"x": {"mode": "group", "personas": ["alex_chen", "alex_chen"]}}},
            {"lines": {"x y": {"persona": "alex_chen"}}},
            {"lines": {"x": {"persona": "alex_chen", "max_seconds": -1}}},
            {"lines": {"x": {"mode": "group", "personas": ["alex_chen"], "handoffs": 9}}},
            {"lines": {"x": {"persona": "alex_chen", "heard_as": {"sam_ortiz": ["Psalm"]}}}},
            {"lines": {"x": {"persona": "alex_chen", "heard_as": {"alex_chen": "Ava"}}}},
            {"lines": {"x": {"persona": "alex_chen", "heard_as": {"alex_chen": ["Ava", "ava"]}}}},
        ):
            with self.subTest(bad=bad), self.assertRaises(lines.LineError):
                lines.parse(bad, PERSONAS)


class SpeakableTest(unittest.TestCase):
    def test_markdown_links_and_code_are_not_spoken(self):
        text = "**Lead:** the [vote](https://x.example/a) passed.\n- first\n- second\n```py\nx=1\n```"
        self.assertEqual(speech.speakable(text), "Lead: the vote passed. first second")

    def test_long_reply_is_cut_at_a_sentence(self):
        text = "One sentence here. " * 60
        out = speech.speakable(text, limit=100)
        self.assertLessEqual(len(out), 100)
        self.assertTrue(out.endswith("."))

    def test_group_note_names_the_others(self):
        note = speech.call_note("", ["Sam Ortiz"])
        self.assertIn("Others on the call: Sam Ortiz", note)
        self.assertTrue(note.startswith("[") and note.endswith("]"))


class RoomTest(unittest.TestCase):
    def setUp(self):
        for name, value in {"memory_store": None, "agent_workspace_store": None, "_org_workspace": None,
                            "headless": True, "break_schedule": []}.items():
            patcher = patch.object(_config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def room(self, line, alex=(), sam=()):
        self.alex = FakeRole("alex_chen", "Alex Chen", list(alex))
        self.sam = FakeRole("sam_ortiz", "Sam Ortiz", list(sam))
        session = main.Session(participants={"alex_chen": self.alex, "sam_ortiz": self.sam},
                               run_id="voice-test", headless=True)
        room = Room(line, session=session)
        self.addCleanup(room.close)
        return room

    def say(self, room, text):
        with redirect_stdout(StringIO()):
            return room.say(text)

    def lines(self):
        return lines.parse({"lines": {
            "team": {"mode": "group", "personas": ["alex_chen", "sam_ortiz"], "handoffs": 2},
            "quiet": {"mode": "group", "personas": ["alex_chen", "sam_ortiz"],
                      "heard_as": {"sam_ortiz": ["Psalm", "the  O"]}},
            "editor": {"persona": "sam_ortiz"},
        }}, PERSONAS)

    def test_persona_line_always_answers_with_its_persona(self):
        room = self.room(self.lines()["editor"], sam=["Hello, editor here.", "Yes."])
        replies = self.say(room, "Alex, are you there?")
        self.assertEqual([(r.persona, r.text) for r in replies], [("sam_ortiz", "Hello, editor here.")])
        self.assertEqual(self.alex.received, [])
        self.assertEqual(self.sam.received[0], ("caller", "Alex, are you there?"))
        # The call note rides in the persona's system message for the call, not in every message.
        self.assertIn("phone call", self.sam.situation_note)
        room.close()
        self.assertFalse(hasattr(self.sam, "situation_note"))

    def test_group_line_answers_the_named_persona_then_keeps_talking_to_them(self):
        room = self.room(self.lines()["team"], alex=["Hi."], sam=["The lead is the vote.", "Tomorrow."])
        first = self.say(room, "Sam, what's the lead?")
        self.assertEqual([r.persona for r in first], ["sam_ortiz"])
        self.assertTrue(self.sam.received[0][1].endswith("what's the lead?"))
        second = self.say(room, "When does it run?")
        self.assertEqual([r.persona for r in second], ["sam_ortiz"])
        self.assertEqual(self.alex.received, [])

    def test_group_line_starts_with_its_first_persona_and_follows_a_mention(self):
        room = self.room(self.lines()["team"], alex=["Alex here."], sam=["Sam here."])
        self.assertEqual([r.persona for r in self.say(room, "Hello?")], ["alex_chen"])
        self.assertEqual([r.persona for r in self.say(room, "What does Sam think?")], ["sam_ortiz"])

    def test_a_persona_can_hand_the_question_on(self):
        room = self.room(self.lines()["team"], alex=["Sam, can you take this one?"], sam=["Sure, it runs Friday."])
        replies = self.say(room, "When does the story run?")
        self.assertEqual([(r.persona, r.voice) for r in replies], [("alex_chen", "nova"), ("sam_ortiz", None)])
        self.assertIn("Alex Chen asked you", self.sam.received[0][1])
        self.assertEqual(self.sam.received[0][0], "alex_chen")

    def test_a_hand_off_needs_a_question_and_never_bounces_straight_back(self):
        room = self.room(self.lines()["team"], alex=["Sam, I'm on the vote story.", "Sam, what about you?", "Sam, and you?"],
                         sam=["Alex, what are you pitching?"])
        self.assertEqual([r.persona for r in self.say(room, "What are you working on?")], ["alex_chen"])
        replies = self.say(room, "Anything else?")
        self.assertEqual([r.persona for r in replies], ["alex_chen", "sam_ortiz"])
        self.assertEqual(len(self.alex.received), 2)

    def test_a_name_opening_a_reply_is_dropped_unless_it_hands_off(self):
        room = self.room(self.lines()["team"], alex=["Alex, I'm on the vote.", "Sam, the vote passed.", "Sam, when does it run?"],
                         sam=["Friday."])
        self.assertEqual([r.text for r in self.say(room, "Hi?")], ["I'm on the vote."])
        self.assertEqual([r.text for r in self.say(room, "And?")], ["The vote passed."])
        self.assertEqual([r.text for r in self.say(room, "When?")], ["Sam, when does it run?", "Friday."])

    def test_without_handoffs_personas_answer_the_caller_only(self):
        room = self.room(self.lines()["quiet"], alex=["Sam, what is new with the operations?", "Alex Chen, I haven't read it."],
                         sam=["Never asked."])
        self.assertEqual([(r.persona, r.text) for r in self.say(room, "What's new?")],
                         [("alex_chen", "What is new with the operations?")])
        self.assertEqual([r.text for r in self.say(room, "And the records?")], ["I haven't read it."])
        self.assertEqual(self.sam.received, [])
        self.assertIn("never address the others", self.alex.situation_note)

    def test_heard_as_words_reach_the_persona_they_name(self):
        room = self.room(self.lines()["quiet"], alex=["Hi."], sam=["The vote.", "Friday."])
        self.assertEqual([r.persona for r in self.say(room, "The O, what's the lead?")], ["sam_ortiz"])
        self.assertTrue(self.sam.received[0][1].endswith("what's the lead?"))
        self.assertEqual([r.persona for r in self.say(room, "What does psalm think of theory?")], ["sam_ortiz"])
        self.assertTrue(self.sam.received[1][1].endswith("What does Sam think of theory?"))

    def test_empty_words_and_empty_replies_say_nothing(self):
        room = self.room(self.lines()["editor"], sam=[""])
        self.assertEqual(self.say(room, "   "), [])
        self.assertEqual(self.say(room, "Hello"), [])


if __name__ == "__main__":
    unittest.main()

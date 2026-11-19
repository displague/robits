"""robits.voice's Pipecat pieces (skipped without pipecat: pip install -r requirements-voice.txt)."""
import asyncio
import unittest

try:
    from pipecat.frames.frames import LLMContextFrame, OutputAudioRawFrame
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.tests.utils import SleepFrame, run_test
except ImportError:  # pragma: no cover
    run_test = None

from robits.voice import lines
from robits.voice.room import Reply

LINE = lines.parse({"lines": {"team": {"mode": "group", "personas": ["alex", "sam"], "greeting": "Hi."}}},
                   {"alex": {"voice": "nova"}, "sam": {"voice": "echo"}})["team"]


class FakeRoom:
    line = LINE

    def __init__(self, delay=0.0):
        self.heard = []
        self.delay = delay

    def say(self, text, on_speech=None):
        import time
        time.sleep(self.delay)
        self.heard.append(text)
        return [Reply("alex", "Alex", "Alex here.", "nova"), Reply("sam", "Sam", "And Sam.", "echo")]


class StreamingRoom(FakeRoom):
    """Alex's reply streams sentence by sentence; Sam's comes back whole."""

    def say(self, text, on_speech=None):
        self.heard.append(text)
        on_speech(Reply("alex", "Alex", "Alex here.", "nova", spoken=True))
        on_speech(Reply("alex", "Alex", "Still here.", "nova", spoken=True))
        return [Reply("alex", "Alex", "Alex here. Still here.", "nova", spoken=True),
                Reply("sam", "Sam", "And Sam.", "echo")]


@unittest.skipIf(run_test is None, "pipecat is not installed")
class BrainTest(unittest.TestCase):
    def run_brain(self, room, frames, cue=False):
        from robits.voice import pipeline
        brain = pipeline.Brain(room, cue=cue)
        down, _ = asyncio.run(run_test(brain, frames_to_send=frames))
        return down

    def context(self, *texts):
        return LLMContext(messages=[{"role": "user", "content": t} for t in texts])

    def test_each_reply_is_spoken_in_its_persona_voice(self):
        from robits.voice.pipeline import PersonaSpeakFrame
        room = FakeRoom()
        down = self.run_brain(room, [LLMContextFrame(context=self.context("Who's there?")), SleepFrame(1.2)])
        spoken = [(f.text, f.voice) for f in down if isinstance(f, PersonaSpeakFrame)]
        self.assertEqual(spoken, [("Alex here.", "nova"), ("And Sam.", "echo")])
        self.assertEqual(room.heard, ["Who's there?"])
        self.assertFalse(any(isinstance(f, LLMContextFrame) for f in down))

    def test_streamed_sentences_are_spoken_once_in_order(self):
        from robits.voice.pipeline import PersonaSpeakFrame
        down = self.run_brain(StreamingRoom(), [LLMContextFrame(context=self.context("Who's there?")), SleepFrame(1.2)])
        spoken = [(f.text, f.voice) for f in down if isinstance(f, PersonaSpeakFrame)]
        self.assertEqual(spoken, [("Alex here.", "nova"), ("Still here.", "nova"), ("And Sam.", "echo")])

    def test_an_already_answered_context_is_not_answered_again(self):
        room = FakeRoom()
        ctx = self.context("One.")
        self.run_brain(room, [LLMContextFrame(context=ctx), SleepFrame(1.1), LLMContextFrame(context=ctx), SleepFrame(1.1)])
        self.assertEqual(room.heard, ["One."])

    def test_turns_close_together_are_answered_as_one(self):
        room = FakeRoom()
        self.run_brain(room, [LLMContextFrame(context=self.context("Hi Alex,")),
                              LLMContextFrame(context=self.context("Hi Alex,", "what's new?")), SleepFrame(1.5)])
        self.assertEqual(room.heard, ["Hi Alex, what's new?"])

    def test_the_caller_hears_a_cue_while_the_personas_think(self):
        from robits.voice import pipeline
        old = pipeline.CUE_FIRST_S, pipeline.CUE_EVERY_S
        pipeline.CUE_FIRST_S, pipeline.CUE_EVERY_S = 0.05, 0.1
        try:
            down = self.run_brain(FakeRoom(delay=0.5), [LLMContextFrame(context=self.context("Hm?")), SleepFrame(1.8)], cue=True)
        finally:
            pipeline.CUE_FIRST_S, pipeline.CUE_EVERY_S = old
        self.assertGreaterEqual(sum(isinstance(f, OutputAudioRawFrame) for f in down), 2)


@unittest.skipIf(run_test is None, "pipecat is not installed")
class ServerTest(unittest.TestCase):
    def test_relay_only_needs_turn_servers_and_keeps_only_turn_urls(self):
        from robits.voice import server
        with self.assertRaises(ValueError):
            server.connection({"ice_transport_policy": "relay", "ice_servers": [{"urls": "stun:x:3478"}]})
        servers = server.turn_servers([{"urls": ["stun:x", "turn:t:3478"], "username": "u", "credential": "c"},
                                       {"urls": "turn:y"}])
        self.assertEqual([s.urls for s in servers], [["turn:t:3478"]])

    def test_only_relay(self):
        from robits.voice.server import only_relay
        self.assertTrue(only_relay("a=candidate:1 1 udp 1 10.0.0.1 5 typ relay raddr 0.0.0.0 rport 0"))
        self.assertFalse(only_relay("a=candidate:1 1 udp 1 10.0.0.1 5 typ host\na=candidate:2 1 udp 1 1.1.1.1 5 typ relay"))
        self.assertFalse(only_relay("v=0"))


if __name__ == "__main__":
    unittest.main()

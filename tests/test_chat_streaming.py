"""Streamed chat-completions replies (role.runtime_text_stream); no model server required."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.providers import ChatCompletionsProvider, _streamed_completion


def chunk(content=None, calls=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if content or calls else [], usage=usage)


def call_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def stream_client(rounds):
    """A client whose create() answers stream=True requests with the next round's chunks."""
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return iter(rounds.pop(0))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), requests


class StreamedCompletionTest(unittest.TestCase):
    def test_text_is_delivered_as_written_and_the_message_is_whole(self):
        heard = []
        client, requests = stream_client([[chunk("Hello"), chunk(" there."),
                                           chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2,
                                                                       total_tokens=7))]])
        response = _streamed_completion(client, {"model": "m", "messages": []}, lambda *e: heard.append(e))
        self.assertEqual(heard, [("text", "Hello"), ("text", " there.")])
        self.assertEqual(response.choices[0].message.content, "Hello there.")
        self.assertIsNone(response.choices[0].message.tool_calls)
        self.assertEqual(response.usage.total_tokens, 7)
        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[0]["stream_options"], {"include_usage": True})

    def test_tool_call_fragments_are_joined(self):
        client, _ = stream_client([[chunk(calls=[call_delta(0, id="c1", name="lookup", arguments='{"q":')]),
                                    chunk(calls=[call_delta(0, arguments=' "vote"}')]),
                                    chunk(calls=[call_delta(1, id="c2", name="clock", arguments="")])]])
        message = _streamed_completion(client, {}, lambda *e: None).choices[0].message
        calls = [(c.id, c.function.name, c.function.arguments) for c in message.tool_calls]
        self.assertEqual(calls, [("c1", "lookup", '{"q": "vote"}'), ("c2", "clock", "{}")])

    def test_a_stream_that_fails_after_text_is_not_retryable(self):
        class Dropped(Exception):
            status_code = 503

        def broken():
            yield chunk("Half a")
            raise Dropped("connection reset")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: broken())))
        with self.assertRaises(RuntimeError):
            _streamed_completion(client, {}, lambda *e: None)


class StreamingGenerateTest(unittest.TestCase):
    def setUp(self):
        main.tool_registry.clear()
        self.scope = patch.multiple(config, memory_store=None, active_tool_caller=None)
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.addCleanup(main.tool_registry.clear)

    def role(self):
        employee_dict = main.build_employee_dict(
            {"small": {"role": "SE", "full_name": "small", "entries": [], "tools": ["agent.think"]}})
        with redirect_stdout(StringIO()):
            main.load_tools(main.System(employee_dict))
        return employee_dict["small"]

    def test_a_listener_hears_the_tool_round_then_the_reply(self):
        role = self.role()
        heard = []
        role.runtime_text_stream = lambda *event: heard.append(event)
        client, requests = stream_client([
            [chunk(calls=[call_delta(0, id="c1", name="agent__think", arguments='{"content": "plan"}')])],
            [chunk("Done"), chunk(".")],
        ])
        with redirect_stdout(StringIO()):
            reply = ChatCompletionsProvider(client, main.tool_registry).generate(
                role, "fake", "CEO", [{"role": "user", "content": "hello"}])
        self.assertEqual(reply, "Done.")
        self.assertEqual(heard, [("tool", ["agent__think"]), ("text", "Done"), ("text", ".")])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")

    def test_without_a_listener_nothing_is_streamed(self):
        role = self.role()
        message = SimpleNamespace(content="Plain.", tool_calls=None)
        requests = []
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: (
            requests.append(k), SimpleNamespace(choices=[SimpleNamespace(message=message)]))[1])))
        reply = ChatCompletionsProvider(client, main.tool_registry).generate(role, "fake", "CEO", [])
        self.assertEqual(reply, "Plain.")
        self.assertNotIn("stream", requests[0])


if __name__ == "__main__":
    unittest.main()

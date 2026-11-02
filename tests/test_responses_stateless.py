"""ResponsesProvider tool loops against backends that ignore previous_response_id (#102)."""

import unittest
from types import SimpleNamespace

from robits.core.config import responses_stateful
from robits.core.providers import ChatCompletionsProvider, ResponsesProvider

TOOL = {"type": "function", "name": "lookup", "description": "Look up a fact.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}


class FakeRegistry:
    def __init__(self):
        self.executed = []

    def as_responses_tools(self, role):
        return [TOOL]

    def execute(self, name, args, employee_dict, caller=None):
        self.executed.append((name, args))
        return f"fact about {args.get('q')}"

    def resolve_name(self, name):
        return name


def call_response(response_id, call_id, query):
    return SimpleNamespace(id=response_id, output_text="", output=[
        SimpleNamespace(type="reasoning", summary=[]),
        SimpleNamespace(type="function_call", call_id=call_id, name="lookup",
                        arguments=f'{{"q": "{query}"}}'),
    ])


def text_response(response_id, text):
    return SimpleNamespace(id=response_id, output_text=text, output=[
        SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])])


def make_client(responses):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    return SimpleNamespace(responses=SimpleNamespace(create=create)), calls


def make_role():
    return SimpleNamespace(name="SE", max_tokens=100, temperature=0.0, employee_dict={},
                           reasoning_effort=None, max_tool_rounds=4)


USER = [{"role": "user", "content": "What do we know about alpha and beta?"}]


class StatelessResponsesTests(unittest.TestCase):
    def run_two_rounds(self, stateful):
        client, calls = make_client([call_response("r1", "c1", "alpha"),
                                     call_response("r2", "c2", "beta"),
                                     text_response("r3", "Both found.")])
        provider = ResponsesProvider(client, FakeRegistry(), stateful=stateful)
        reply = provider.generate(make_role(), "small-model", "CEO", list(USER))
        self.assertEqual(reply, "Both found.")
        self.assertEqual(len(calls), 3)
        return calls

    def test_stateless_backend_gets_full_conversation_each_round(self):
        calls = self.run_two_rounds(stateful=False)
        second, third = calls[1], calls[2]
        self.assertNotIn("previous_response_id", second)
        # The failure in #102: only bare function_call_output items reached the model.
        self.assertEqual(second["input"][0], USER[0])
        self.assertEqual(second["input"][1], {"type": "function_call", "call_id": "c1",
                                              "name": "lookup", "arguments": '{"q": "alpha"}'})
        self.assertEqual(second["input"][2], {"type": "function_call_output", "call_id": "c1",
                                              "output": "fact about alpha"})
        self.assertEqual(len(second["input"]), 3, "reasoning items are not replayed")
        # Round two keeps round one and appends its own call and result.
        self.assertEqual(third["input"][:3], second["input"])
        self.assertEqual([item.get("call_id") for item in third["input"][3:]], ["c2", "c2"])
        for request in calls:
            self.assertEqual(request["tools"], [TOOL])
            self.assertEqual(request["max_output_tokens"], 100)

    def test_stateful_backend_chains_previous_response_id(self):
        calls = self.run_two_rounds(stateful=True)
        self.assertEqual(calls[1]["previous_response_id"], "r1")
        self.assertEqual(calls[2]["previous_response_id"], "r2")
        self.assertEqual([item["type"] for item in calls[1]["input"]], ["function_call_output"])
        # Tools are not inherited through previous_response_id, so every round resends them.
        self.assertEqual(calls[1]["tools"], [TOOL])

    def test_stateful_without_response_id_falls_back_to_replay(self):
        first = call_response(None, "c1", "alpha")
        client, calls = make_client([first, text_response("r2", "Found.")])
        reply = ResponsesProvider(client, FakeRegistry(), stateful=True).generate(
            make_role(), "small-model", "CEO", list(USER))
        self.assertEqual(reply, "Found.")
        self.assertNotIn("previous_response_id", calls[1])
        self.assertEqual([item.get("type") for item in calls[1]["input"][1:]],
                         ["function_call", "function_call_output"])


class ResponsesStatefulSettingTests(unittest.TestCase):
    def test_auto_trusts_only_openai(self):
        self.assertTrue(responses_stateful("auto", None))
        self.assertTrue(responses_stateful("auto", "https://api.openai.com/v1"))
        self.assertFalse(responses_stateful("auto", "http://127.0.0.1:11434/v1/"))
        self.assertFalse(responses_stateful("", "http://gateway.example/v1"))

    def test_explicit_setting_wins(self):
        self.assertTrue(responses_stateful("true", "http://127.0.0.1:1234/v1/"))
        self.assertFalse(responses_stateful("off", None))


class ObjectArgumentsTests(unittest.TestCase):
    """Some local servers send tool-call arguments as a JSON object instead of a string."""

    def test_chat_completions_accepts_object_arguments(self):
        call = SimpleNamespace(id="c1", type="function",
                               function=SimpleNamespace(name="lookup", arguments={"q": "alpha"}))
        first = SimpleNamespace(content="", tool_calls=[call], model_dump=lambda: {
            "role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": {"q": "alpha"}}}]})
        final = SimpleNamespace(content="Found.", tool_calls=None)
        sent = []

        def create(**kwargs):
            sent.append(kwargs)
            message = first if len(sent) == 1 else final
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        registry = FakeRegistry()
        registry.as_chat_completion_tools = lambda role: [{"type": "function", "function": TOOL}]
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        reply = ChatCompletionsProvider(client, registry).generate(make_role(), "small-model", "CEO", list(USER))
        self.assertEqual(reply, "Found.")
        self.assertEqual(registry.executed, [("lookup", {"q": "alpha"})])
        echoed = sent[1]["messages"][1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(echoed, '{"q": "alpha"}', "arguments go back as a JSON string")

    def test_responses_accepts_object_arguments(self):
        first = SimpleNamespace(id="r1", output_text="", output=[SimpleNamespace(
            type="function_call", call_id="c1", name="lookup", arguments={"q": "alpha"})])
        client, calls = make_client([first, text_response("r2", "Found.")])
        registry = FakeRegistry()
        reply = ResponsesProvider(client, registry, stateful=False).generate(
            make_role(), "small-model", "CEO", list(USER))
        self.assertEqual(reply, "Found.")
        self.assertEqual(registry.executed, [("lookup", {"q": "alpha"})])
        self.assertEqual(calls[1]["input"][1]["arguments"], '{"q": "alpha"}')


if __name__ == "__main__":
    unittest.main()

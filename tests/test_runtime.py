import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import main


class FakeRole:
    def __init__(self, name, responses=None):
        self.name = name
        self.responses = list(responses or [])
        self.received = []
        self.group_conversation_history = {}

    def interact(self, sender=None, prompt=None):
        self.received.append((sender, prompt))
        if self.responses:
            return self.responses.pop(0)
        return ""

    def update_group_conversations(self, message):
        self.group_conversation_history.setdefault(self.name, []).append(message)


def build_fake_participants():
    return {
        "CEO": FakeRole("CEO", ["initial"]),
        "Ops": FakeRole("Ops", ["SE, handoff"]),
        "SE": FakeRole("SE", ["HR, handoff"]),
        "HR": FakeRole("HR", ["Samandriel, handoff"]),
        "Samandriel": FakeRole("Samandriel", ["CEO, done"]),
    }


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        main.tool_registry.clear()

    def test_parse_tool_instruction_handles_surrounding_text(self):
        response = 'Ops should run this:\n{"exec": "create_role", "args": {"role_name": "QA"}}\nDone.'

        self.assertEqual(
            json.loads(main.parse_tool_instruction(response)),
            {"exec": "create_role", "args": {"role_name": "QA"}},
        )

    def test_parse_tool_instruction_handles_multiline_json(self):
        response = """Please execute:
{
  "exec": "create_role",
  "args": {
    "role_name": "QA",
    "role_description": "Tests the organization"
  }
}
"""

        self.assertEqual(
            json.loads(main.parse_tool_instruction(response)),
            {
                "exec": "create_role",
                "args": {
                    "role_name": "QA",
                    "role_description": "Tests the organization",
                },
            },
        )

    def test_loaded_create_role_executes_against_employee_dict(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        with redirect_stdout(StringIO()):
            response = system.interact(
                "  \n" + json.dumps(
                    {
                        "exec": "create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )
            )

        self.assertIn("Created a new role: QA", response)
        self.assertIn("QA", employee_dict)
        self.assertIsInstance(employee_dict["QA"], main.Role)
        self.assertEqual(employee_dict["QA"].lifecycle_state, "active")

    def test_system_accepts_json_instruction_arrays(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    [
                        {
                            "exec": "create_role",
                            "args": {
                                "role_name": "QA",
                                "role_description": "Tests the organization",
                            },
                        }
                    ]
                )
            )

        self.assertIn("Created a new role: QA", response)
        self.assertIn("QA", employee_dict)

    def test_tool_reports_missing_args(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "exec": "create_role",
                        "args": {
                            "role_name": "QA",
                        },
                    }
                )
            )

        self.assertIn("Missing args", response)
        self.assertNotIn("QA", employee_dict)

    def test_create_role_records_lifecycle_request_and_approval(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                            "requested_by": "CEO",
                            "approved_by": "HR",
                        },
                    }
                )
            )

        event = employee_dict["QA"].lifecycle_events[0]

        self.assertIn("Created a new role: QA", response)
        self.assertEqual(event.action, "create")
        self.assertEqual(event.lifecycle_state, "active")
        self.assertEqual(event.requested_by, "CEO")
        self.assertEqual(event.approved_by, "HR")
        self.assertIn("+00:00", event.created_at)

    def test_create_role_rejects_duplicate_role(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            first = system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )
            )
            duplicate = system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Duplicate role",
                        },
                    }
                )
            )

        self.assertIn("Created a new role: QA", first)
        self.assertIn("already exists", duplicate)

    def test_create_role_rejects_capacity_limit(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        original_limit = main.HR.max_organization_members
        main.HR.max_organization_members = len(employee_dict)
        try:
            with redirect_stdout(StringIO()):
                main.load_tools(system)
                response = system.interact(
                    json.dumps(
                        {
                            "exec": "org.create_role",
                            "args": {
                                "role_name": "QA",
                                "role_description": "Tests the organization",
                            },
                        }
                    )
                )
        finally:
            main.HR.max_organization_members = original_limit

        self.assertIn("maximum size", response)
        self.assertNotIn("QA", employee_dict)

    def test_pause_and_retire_role_update_lifecycle_state(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                            "requested_by": "CEO",
                            "approved_by": "HR",
                        },
                    }
                )
            )
            pause_response = system.interact(
                json.dumps(
                    {
                        "exec": "org.pause_role",
                        "args": {
                            "role_name": "QA",
                            "requested_by": "HR",
                            "approved_by": "HR",
                            "reason": "Rest period",
                        },
                    }
                )
            )
            retire_response = system.interact(
                json.dumps(
                    {
                        "exec": "org.retire_role",
                        "args": {
                            "role_name": "QA",
                            "requested_by": "HR",
                            "approved_by": "CEO",
                            "reason": "Assignment complete",
                        },
                    }
                )
            )

        qa = employee_dict["QA"]

        self.assertIn("paused", pause_response)
        self.assertIn("retired", retire_response)
        self.assertEqual(qa.lifecycle_state, "retired")
        self.assertEqual([event.action for event in qa.lifecycle_events], ["create", "pause", "retire"])
        self.assertEqual(qa.lifecycle_events[-1].approved_by, "CEO")

    def test_lifecycle_rejects_invalid_transitions_without_new_event(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )
            )
            first_pause = system.interact(
                json.dumps({"exec": "org.pause_role", "args": {"role_name": "QA"}})
            )
            second_pause = system.interact(
                json.dumps({"exec": "org.pause_role", "args": {"role_name": "QA"}})
            )
            retire = system.interact(
                json.dumps({"exec": "org.retire_role", "args": {"role_name": "QA"}})
            )
            retire_again = system.interact(
                json.dumps({"exec": "org.retire_role", "args": {"role_name": "QA"}})
            )

        qa = employee_dict["QA"]

        self.assertIn("paused", first_pause)
        self.assertIn("Cannot pause", second_pause)
        self.assertIn("retired", retire)
        self.assertIn("Cannot retire", retire_again)
        self.assertEqual([event.action for event in qa.lifecycle_events], ["create", "pause", "retire"])

    def test_tool_lifecycle_event_is_recorded_in_session_transcript(self):
        participants = build_fake_participants()
        participants["Ops"] = FakeRole("Ops", ["done"])
        system = main.System(participants)
        session = main.Session(participants=participants, system=system, run_id="session-1")
        payload = json.dumps(
            {
                "exec": "org.create_role",
                "args": {
                    "role_name": "QA",
                    "role_description": "Tests the organization",
                    "requested_by": "CEO",
                    "approved_by": "HR",
                },
            }
        )

        with redirect_stdout(StringIO()):
            main.load_tools(system)
            session.run(initial_message=payload, max_turns=1)

        event_text = session.transcript[0].system_events[0]

        self.assertIn("requested by CEO", event_text)
        self.assertIn("approved by HR", event_text)

    def test_tool_definition_validates_args_shape(self):
        system = main.System(main.build_employee_dict())

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "code_name": "bad_tool",
                        "args": [{"label": "value"}],
                        "code": "return value",
                    }
                ),
                trusted=True,
            )

        self.assertIn("args must be a list", response)
        self.assertNotIn("bad_tool", main.tool_registry)

    def test_exec_instruction_validates_args_object(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "exec": "create_role",
                        "args": [],
                    }
                )
            )

        self.assertIn("args must be an object", response)

    def test_load_tools_default_path_is_relative_to_module(self):
        system = main.System(main.build_employee_dict())
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                with redirect_stdout(StringIO()):
                    main.load_tools(system)
            finally:
                os.chdir(original_cwd)

        self.assertIn("org.create_role", main.tool_registry)

    def test_untrusted_tool_definition_is_rejected(self):
        system = main.System(main.build_employee_dict())

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "code_name": "surprise",
                        "args": [],
                        "code": "return 'nope'",
                    }
                )
            )

        self.assertIn("trusted tool", response)
        self.assertNotIn("surprise", main.tool_registry)

    def test_tool_builtins_are_restricted(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "code_name": "read_file",
                        "args": [],
                        "code": "return __import__('os').listdir('.')",
                    }
                ),
                trusted=True,
            )

        self.assertIn("Stored tool", response)
        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "exec": "read_file",
                        "args": {},
                    }
                )
            )

        self.assertIn("__import__", response)

    def test_tool_registry_exposes_responses_tool_metadata(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        tools = main.tool_registry.as_responses_tools()
        tool = next(tool for tool in tools if tool["name"] == "org__create_role")

        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["parameters"]["required"], ["role_name", "role_description"])

    def test_tool_registry_exposes_chat_completion_tool_metadata(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        tools = main.tool_registry.as_chat_completion_tools()
        tool = next(
            tool for tool in tools if tool["function"]["name"] == "org__create_role"
        )

        self.assertEqual(tool["type"], "function")
        self.assertEqual(
            tool["function"]["parameters"]["required"],
            ["role_name", "role_description"],
        )

    def test_namespaced_exec_resolves_tool(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )
            )

        self.assertIn("Created a new role: QA", response)
        self.assertIn("QA", employee_dict)

    def test_openai_tool_name_exec_resolves_tool(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "exec": "org__create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )
            )

        self.assertIn("Created a new role: QA", response)
        self.assertIn("QA", employee_dict)

    def test_optional_tool_property_can_be_supplied_or_omitted(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "name": "test.echo",
                        "description": "Echo a required value with an optional suffix.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "suffix": {"type": "string"},
                            },
                            "required": ["value"],
                        },
                        "code": "return value + (suffix or '')",
                    }
                ),
                trusted=True,
            )
            omitted = system.interact(
                json.dumps({"exec": "test.echo", "args": {"value": "a"}})
            )
            supplied = system.interact(
                json.dumps(
                    {"exec": "test.echo", "args": {"value": "a", "suffix": "b"}}
                )
            )

        self.assertIn("Stored tool", response)
        self.assertIn("Result: a", omitted)
        self.assertIn("Result: ab", supplied)

    def test_duplicate_tool_name_is_rejected(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "name": "org.create_role",
                        "description": "Duplicate",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                        "code": "return 'duplicate'",
                    }
                ),
                trusted=True,
            )

        self.assertIn("already exists", response)
        self.assertEqual(main.tool_registry.get("create_role").name, "org.create_role")

    def test_invalid_tool_alias_is_rejected(self):
        system = main.System(main.build_employee_dict())

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "name": "test.alias",
                        "aliases": ["bad-alias"],
                        "description": "Bad alias.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                        "code": "return 'bad'",
                    }
                ),
                trusted=True,
            )

        self.assertIn("Invalid tool name", response)
        self.assertNotIn("test.alias", main.tool_registry)

    def test_invalid_qualified_exec_name_is_rejected(self):
        system = main.System(main.build_employee_dict())

        with redirect_stdout(StringIO()):
            response = system.interact(
                json.dumps(
                    {
                        "exec": "bad-name.create_role",
                        "args": {},
                    }
                )
            )

        self.assertIn("Invalid tool name", response)

    def test_load_tools_requires_list_file(self):
        system = main.System(main.build_employee_dict())
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("name: not-a-list\n")
            path = handle.name

        try:
            with self.assertRaisesRegex(ValueError, "list of tool definitions"):
                main.load_tools(system, path)
        finally:
            os.unlink(path)

    def test_session_creation_records_run_id_participants_and_transcript(self):
        participants = build_fake_participants()
        session = main.Session(participants=participants, run_id="session-1", max_turns=3)

        self.assertEqual(session.run_id, "session-1")
        self.assertIs(session.participants, participants)
        self.assertEqual(session.max_turns, 3)
        self.assertEqual(session.turns_completed, 0)
        self.assertEqual(session.transcript, [])
        self.assertIn(
            "session.created",
            [event.event_type for event in session.event_stream.events()],
        )

    def test_round_robin_scheduler_skips_last_receiver(self):
        scheduler = main.RoundRobinScheduler(["CEO", "Ops", "SE"])

        self.assertEqual(scheduler.next("CEO"), "Ops")
        self.assertEqual(scheduler.next("Ops"), "SE")
        self.assertEqual(scheduler.next("SE"), "CEO")
        self.assertEqual(scheduler.next("CEO"), "Ops")

    def test_directed_message_routes_to_named_recipient_and_strips_prefix(self):
        session = main.Session(participants=build_fake_participants(), run_id="session-1")

        with redirect_stdout(StringIO()):
            routed = session.route_message(" HR , please handle this", "CEO")

        self.assertTrue(routed.directed)
        self.assertEqual(routed.receiver.name, "HR")
        self.assertEqual(routed.prompt, "please handle this")

    def test_unknown_directed_prefix_falls_back_to_scheduler(self):
        session = main.Session(participants=build_fake_participants(), run_id="session-1")

        with redirect_stdout(StringIO()):
            routed = session.route_message("Finance, please handle this", "CEO")

        self.assertFalse(routed.directed)
        self.assertEqual(routed.receiver.name, "Ops")
        self.assertEqual(routed.prompt, "Finance, please handle this")

    def test_directed_message_advances_scheduler_after_recipient(self):
        session = main.Session(participants=build_fake_participants(), run_id="session-1")

        with redirect_stdout(StringIO()):
            routed = session.route_message("HR, please handle this", "CEO")
            next_routed = session.route_message("continue", "HR")

        self.assertEqual(routed.receiver.name, "HR")
        self.assertEqual(next_routed.receiver.name, "Samandriel")

    def test_session_stops_at_configured_turn_limit_without_model_calls(self):
        participants = build_fake_participants()
        session = main.Session(participants=participants, run_id="session-1", max_turns=1)
        with redirect_stdout(StringIO()):
            result = session.run(initial_message="hello")

        self.assertIs(result, session)
        self.assertEqual(session.turns_completed, 1)
        self.assertEqual(len(session.transcript), 1)
        self.assertEqual(participants["Ops"].received, [("CEO", "hello")])
        self.assertEqual(participants["SE"].received, [])

    def test_run_turn_limit_overrides_session_default(self):
        participants = build_fake_participants()
        session = main.Session(participants=participants, run_id="session-1", max_turns=3)
        with redirect_stdout(StringIO()):
            session.run(initial_message="hello", max_turns=1)

        self.assertEqual(session.turns_completed, 1)

    def test_session_records_directed_transcript_entries(self):
        participants = build_fake_participants()
        session = main.Session(participants=participants, run_id="session-1")
        with redirect_stdout(StringIO()):
            session.run(initial_message="HR, status please", max_turns=1)

        entry = session.transcript[0]

        self.assertEqual(entry.turn, 1)
        self.assertEqual(entry.sender, "CEO")
        self.assertEqual(entry.receiver, "HR")
        self.assertEqual(entry.prompt, "status please")
        self.assertEqual(entry.response, "Samandriel, handoff")
        self.assertTrue(entry.directed)

    def test_empty_model_response_consumes_bounded_turn(self):
        participants = build_fake_participants()
        participants["Ops"] = FakeRole("Ops", [None])
        session = main.Session(participants=participants, run_id="session-1")
        with redirect_stdout(StringIO()):
            session.run(initial_message="hello", max_turns=1)

        self.assertEqual(session.turns_completed, 1)
        self.assertEqual(session.transcript[0].response, "")

    def test_none_initial_response_is_normalized_for_session_step(self):
        participants = build_fake_participants()
        participants["CEO"] = FakeRole("CEO", [None])
        session = main.Session(participants=participants, run_id="session-1")
        with redirect_stdout(StringIO()):
            session.run(max_turns=1)

        self.assertEqual(session.turns_completed, 1)
        self.assertEqual(session.transcript[0].prompt, "")

    def test_non_string_message_does_not_parse_tool_instruction(self):
        session = main.Session(participants=build_fake_participants(), run_id="session-1")

        self.assertEqual(session.process_tool_instruction(None), [])
        self.assertEqual(session.process_tool_instruction({"exec": "org.create_role"}), [])

    def test_tool_execution_records_system_event_and_updates_scheduler(self):
        participants = build_fake_participants()
        participants["Ops"] = FakeRole("Ops", ["done"])
        system = main.System(participants)
        session = main.Session(participants=participants, system=system, run_id="session-1")
        payload = json.dumps(
            {
                "exec": "org.create_role",
                "args": {
                    "role_name": "QA",
                    "role_description": "Tests the organization",
                },
            }
        )

        with redirect_stdout(StringIO()):
            main.load_tools(system)
            session.run(initial_message=payload, max_turns=1)

        self.assertIn("QA", participants)
        self.assertIn("QA", session.scheduler.participant_names)
        self.assertEqual(len(session.transcript[0].system_events), 1)
        self.assertIn("Created a new role: QA", session.transcript[0].system_events[0])

    def test_session_emits_headless_runtime_events(self):
        participants = build_fake_participants()
        event_stream = main.RuntimeEventStream()
        observed = []
        event_stream.subscribe(observed.append)
        session = main.Session(
            participants=participants,
            run_id="session-1",
            event_stream=event_stream,
        )

        with redirect_stdout(StringIO()):
            session.run(initial_message="HR, status please", max_turns=1)

        event_types = [event.event_type for event in observed]

        self.assertIn("session.created", event_types)
        self.assertIn("message.routed", event_types)
        self.assertIn("message.recorded", event_types)
        self.assertIn("session.completed", event_types)

    def test_event_subscriber_errors_do_not_break_runtime(self):
        event_stream = main.RuntimeEventStream()

        def broken_subscriber(_event):
            raise RuntimeError("observer failed")

        event_stream.subscribe(broken_subscriber)
        event = event_stream.emit("session.created", "session-1")

        self.assertEqual(event.event_type, "session.created")
        self.assertEqual(event_stream.subscriber_errors[0]["event_type"], "session.created")
        self.assertEqual(event_stream.subscriber_errors[0]["error"], "observer failed")

    def test_thought_events_are_private_by_default(self):
        session = main.Session(participants=build_fake_participants(), run_id="session-1")

        event = session.record_thought("SE", "Private implementation note.")

        self.assertEqual(event.event_type, "thought.recorded")
        self.assertEqual(event.visibility, "private")
        self.assertEqual(session.event_stream.events("private"), [event])


if __name__ == "__main__":
    unittest.main()

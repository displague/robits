import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.memory.sqlite import SQLiteMemoryStore


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


class FakeCreate:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class RecordingStream:
    def __init__(self):
        self.content = ""
        self.flushes = 0

    def write(self, text):
        self.content += text

    def flush(self):
        self.flushes += 1


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
        self.original_tool_proposal_store = main.tool_proposal_store
        self.original_agent_workspace_store = main.agent_workspace_store
        main.tool_proposal_store = main.ToolProposalStore()
        self.workspace_temp_dir = tempfile.TemporaryDirectory()
        main.agent_workspace_store = main.AgentWorkspaceStore(self.workspace_temp_dir.name)
        self.addCleanup(self.restore_tool_proposal_store)
        self.addCleanup(self.workspace_temp_dir.cleanup)

    def restore_tool_proposal_store(self):
        main.tool_proposal_store = self.original_tool_proposal_store
        main.agent_workspace_store = self.original_agent_workspace_store

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

    def test_parse_agent_action_recognizes_wait_think_reply_and_tool(self):
        self.assertEqual(
            main.parse_agent_action('{"action": "wait"}'),
            {"action": "wait"},
        )
        self.assertEqual(
            main.parse_agent_action('{"action": "think", "content": "Need a plan."}'),
            {"action": "think", "content": "Need a plan."},
        )
        self.assertEqual(
            main.parse_agent_action('{"action": "reply", "content": "hello"}'),
            {"action": "reply", "content": "hello"},
        )
        self.assertEqual(
            main.parse_agent_action('{"exec": "tools.list", "args": {}}'),
            {"exec": "tools.list", "args": {}},
        )
        self.assertIsNone(main.parse_agent_action('{"note": "plain JSON, not an action"}'))

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

    def test_create_and_list_roles_include_capabilities(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            create_response = system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "SRE",
                            "role_description": "Operates runtime infrastructure.",
                            "capabilities": ["operator", "kubeapi"],
                        },
                    }
                )
            )
            list_response = system.interact(
                json.dumps({"exec": "org.list_roles", "args": {}})
            )

        role_list = json.loads(list_response)
        sre = next(role for role in role_list if role["role_name"] == "SRE")

        self.assertIn("Created a new role: SRE", create_response)
        self.assertEqual(sre["capabilities"], ["kubeapi", "operator"])
        self.assertEqual(
            sre["tool_grants"],
            ["agent.*", "builtin.url_fetch", "builtin.web_search",
             "memory.expand_digest", "memory.list_digests", "memory.search"],
        )

    def test_archive_role_rejects_protected_roles_and_exits_eligible_role(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            protected_response = system.interact(
                json.dumps({"exec": "org.archive_role", "args": {"role_name": "CEO"}})
            )
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
            archived_response = system.interact(
                json.dumps({"exec": "org.archive_role", "args": {"role_name": "QA"}})
            )

        self.assertIn("protected", protected_response)
        self.assertIn("exited", archived_response)
        self.assertEqual(employee_dict["QA"].lifecycle_state, "exited")

    def test_create_role_can_seed_tool_grants(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "Research",
                            "role_description": "Explores memory.",
                            "tool_grants": ["tools.list"],
                        },
                    }
                ),
                caller=employee_dict["HR"],
            )

        self.assertIn("Created a new role: Research", response)
        self.assertIn("tools.list", employee_dict["Research"].allowed_tools)

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

    def test_responses_provider_executes_registered_function_call(self):
        employee_dict = main.build_employee_dict()
        role = employee_dict["HR"]
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)

        first_response = SimpleNamespace(
            id="response-1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name="org__create_role",
                    arguments=json.dumps(
                        {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        }
                    ),
                )
            ],
        )
        final_response = SimpleNamespace(id="response-2", output_text="Created QA.")
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(
                create=FakeCreate([first_response, final_response])
            )
        )
        provider = main.ResponsesProvider(fake_client)
        event_stream = main.RuntimeEventStream()
        role.runtime_event_stream = event_stream
        role.runtime_session_id = "session-1"
        role.runtime_role_name = "HR"

        response = provider.generate(
            role,
            "test-model",
            "CEO",
            [{"role": "user", "content": "create QA", "name": "CEO"}],
        )

        self.assertEqual(response, "Created QA.")
        self.assertIn("QA", employee_dict)
        second_call = fake_client.responses.create.calls[1]
        self.assertEqual(second_call["previous_response_id"], "response-1")
        self.assertEqual(second_call["input"][0]["type"], "function_call_output")
        self.assertIn("Created a new role: QA", second_call["input"][0]["output"])
        event_types = [event.event_type for event in event_stream.events()]
        self.assertIn("tool_call.requested", event_types)
        self.assertIn("tool_call.executed", event_types)
        self.assertTrue(role.runtime_tool_results)
        self.assertIn("tool_call.executed: org__create_role", role.runtime_tool_results[0])

    def test_responses_provider_exposes_only_role_allowed_tools(self):
        employee_dict = main.build_employee_dict()
        role = employee_dict["SE"]
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
        final_response = SimpleNamespace(id="response-1", output_text="ok")
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=FakeCreate([final_response]))
        )
        provider = main.ResponsesProvider(fake_client)

        provider.generate(role, "test-model", "CEO", [{"role": "user", "content": "hi"}])

        tool_names = {tool["name"] for tool in fake_client.responses.create.calls[0]["tools"]}
        self.assertIn("tools__propose", tool_names)
        self.assertIn("builtin__web_search", tool_names)
        self.assertIn("builtin__url_fetch", tool_names)
        self.assertNotIn("org__create_role", tool_names)

    def test_disallowed_tool_call_is_rejected_at_runtime(self):
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
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )

        self.assertIn("not allowed", response)
        self.assertNotIn("QA", employee_dict)

    def test_operator_can_grant_tool_access(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            grant_response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.grant",
                        "args": {
                            "role_name": "SE",
                            "tool_name": "memory__search",
                            "granted_by": "Ops",
                        },
                    }
                ),
                caller=employee_dict["Ops"],
            )

        self.assertIn("Granted tool access", grant_response)
        self.assertIn("memory.search", employee_dict["SE"].allowed_tools)

    def test_operator_can_revoke_tool_access_with_openai_alias(self):
        employee_dict = main.build_employee_dict()
        employee_dict["SE"].allowed_tools.add("memory.search")
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            revoke_response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.revoke",
                        "args": {
                            "role_name": "SE",
                            "tool_name": "memory__search",
                            "revoked_by": "Ops",
                        },
                    }
                ),
                caller=employee_dict["Ops"],
            )

        self.assertIn("Revoked tool access", revoke_response)
        self.assertNotIn("memory.search", employee_dict["SE"].allowed_tools)

    def test_se_cannot_propose_system_tool_update(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.propose",
                        "args": {
                            "requested_by": "SE",
                            "tool_name": "memory.search",
                            "description": "Change memory access.",
                            "action": "update",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )

        self.assertIn("System tool", response)

    def test_tool_proposal_lifecycle_can_be_listed_and_approved(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            propose_response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.propose",
                        "args": {
                            "requested_by": "SE",
                            "tool_name": "weather.lookup",
                            "description": "Look up current weather for an agent location.",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                            "owner_capability": "operator",
                            "safety_notes": "Network-backed; cache results.",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )
            proposal = json.loads(propose_response)
            list_response = system.interact(
                json.dumps({"exec": "tools.list_proposals", "args": {"status": "proposed"}}),
                caller=employee_dict["SE"],
            )
            approve_response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.approve_proposal",
                        "args": {
                            "proposal_id": proposal["proposal_id"],
                            "approved_by": "Ops",
                            "implementation_notes": "Implement as a provider-backed function tool.",
                        },
                    }
                ),
                caller=employee_dict["Ops"],
            )

        proposals = json.loads(list_response)
        approved = json.loads(approve_response)

        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["parameters"]["required"], ["location"])
        self.assertEqual(proposals[0]["proposal_id"], proposal["proposal_id"])
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["approver"], "Ops")

    def test_operator_can_reject_tool_proposal(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            proposal = json.loads(
                system.interact(
                    json.dumps(
                        {
                            "exec": "tools.propose",
                            "args": {
                                "requested_by": "SE",
                                "tool_name": "weather.lookup",
                                "description": "Look up current weather.",
                            },
                        }
                    ),
                    caller=employee_dict["SE"],
                )
            )
            reject_response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.reject_proposal",
                        "args": {
                            "proposal_id": proposal["proposal_id"],
                            "rejected_by": "Ops",
                            "reason": "Needs a safer provider design.",
                        },
                    }
                ),
                caller=employee_dict["Ops"],
            )

        rejected = json.loads(reject_response)

        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["rejection_reason"], "Needs a safer provider design.")

    def test_tool_proposal_rollout_grants_registered_tool_access(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            system.interact(
                json.dumps(
                    {
                        "name": "weather.lookup",
                        "description": "Look up current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                        "code": "return 'sunny'",
                    }
                ),
                trusted=True,
            )
            proposal = json.loads(
                system.interact(
                    json.dumps(
                        {
                            "exec": "tools.propose",
                            "args": {
                                "requested_by": "SE",
                                "tool_name": "weather.lookup",
                                "description": "Roll weather lookup out to SE.",
                                "action": "update",
                            },
                        }
                    ),
                    caller=employee_dict["SE"],
                )
            )
            system.interact(
                json.dumps(
                    {
                        "exec": "tools.approve_proposal",
                        "args": {"proposal_id": proposal["proposal_id"], "approved_by": "Ops"},
                    }
                ),
                caller=employee_dict["Ops"],
            )
            rollout_response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.rollout_proposal",
                        "args": {
                            "proposal_id": proposal["proposal_id"],
                            "role_name": "SE",
                            "granted_by": "Ops",
                        },
                    }
                ),
                caller=employee_dict["Ops"],
            )

        rollout = json.loads(rollout_response)

        self.assertEqual(rollout["proposal"]["status"], "operationalized")
        self.assertIn("weather.lookup", employee_dict["SE"].allowed_tools)

    def test_tool_proposal_rollout_requires_registered_tool(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            proposal = json.loads(
                system.interact(
                    json.dumps(
                        {
                            "exec": "tools.propose",
                            "args": {
                                "requested_by": "SE",
                                "tool_name": "weather.lookup",
                                "description": "Look up current weather.",
                            },
                        }
                    ),
                    caller=employee_dict["SE"],
                )
            )
            system.interact(
                json.dumps(
                    {
                        "exec": "tools.approve_proposal",
                        "args": {"proposal_id": proposal["proposal_id"], "approved_by": "Ops"},
                    }
                ),
                caller=employee_dict["Ops"],
            )
            response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.rollout_proposal",
                        "args": {"proposal_id": proposal["proposal_id"], "role_name": "SE"},
                    }
                ),
                caller=employee_dict["Ops"],
            )

        self.assertIn("not registered", response)

    def test_tools_list_reports_allowed_access_for_role(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "tools.list",
                        "args": {"role_name": "SE", "only_allowed": True},
                    }
                ),
                caller=employee_dict["SE"],
            )

        tools = json.loads(response)
        names = {tool["name"] for tool in tools}

        self.assertIn("tools.propose", names)
        self.assertNotIn("org.create_role", names)

    def test_tools_list_without_role_does_not_report_allowed_true(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps({"exec": "tools.list", "args": {}})
            )

        tools = json.loads(response)

        self.assertIsNone(tools[0]["allowed"])

    def test_tools_list_only_allowed_requires_role(self):
        system = main.System(main.build_employee_dict())
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps({"exec": "tools.list", "args": {"only_allowed": True}})
            )

        self.assertIn("role_name is required", response)


    def test_model_retry_retries_rate_limit_errors(self):
        attempts = []
        original_retries = main.max_api_retries
        original_base = main.api_retry_base_seconds
        original_max = main.api_retry_max_seconds
        try:
            main.max_api_retries = 2
            main.api_retry_base_seconds = 0
            main.api_retry_max_seconds = 0

            class FakeRateLimit(Exception):
                status_code = 429

            def operation():
                attempts.append("attempt")
                if len(attempts) == 1:
                    raise FakeRateLimit("too many requests")
                return "ok"

            with patch("robits.core.providers.time.sleep") as sleep:
                result = main._with_model_retries(operation)
        finally:
            main.max_api_retries = original_retries
            main.api_retry_base_seconds = original_base
            main.api_retry_max_seconds = original_max

        self.assertEqual(result, "ok")
        self.assertEqual(len(attempts), 2)
        sleep.assert_not_called()

    def test_chat_completion_parallelism_gate_covers_api_call(self):
        gate_was_held = []

        def fake_create(**_kwargs):
            gate_was_held.append(not main.model_call_gate.acquire(blocking=False))
            if not gate_was_held[-1]:
                main.model_call_gate.release()
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="hello", tool_calls=None)
                    )
                ]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create),
            )
        )
        provider = main.ChatCompletionsProvider(fake_client)
        role = SimpleNamespace(name="SE", max_tokens=10, temperature=0)

        response = provider.generate(role, "test-model", "CEO", [])

        self.assertEqual(response, "hello")
        self.assertEqual(gate_was_held, [True])

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
                ),
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
        self.assertEqual(omitted, "a")
        self.assertEqual(supplied, "ab")

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

    def test_parse_args_accepts_log_path(self):
        args = main.parse_args(["--prompt", "hello", "--turns", "1", "--log", "run.log"])

        self.assertEqual(args.prompt, "hello")
        self.assertEqual(args.turns, 1)
        self.assertEqual(args.log, "run.log")

    def test_main_creates_jsonl_log_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "logs", "run.jsonl")

            def fake_run_simulation(initial_message=None, max_turns=None):
                import main as _main
                _main.get_logger().write_event("test_event", msg=f"prompt={initial_message}")
                print(f"prompt={initial_message} turns={max_turns}")

            with patch.object(main, "run_simulation", side_effect=fake_run_simulation):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    main.main(["--prompt", "hello", "--turns", "2", "--log", log_path])

            # Console output still appears on stdout, not captured in log file
            self.assertIn("prompt=hello turns=2", stdout.getvalue())
            # Log file is created and contains JSONL events
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn('"type": "test_event"', content)
            self.assertNotIn("prompt=hello turns=2", content)

    def test_tee_stream_flushes_after_each_write(self):
        first = RecordingStream()
        second = RecordingStream()
        tee = main.TeeStream(first, second)

        written = tee.write("hello")

        self.assertEqual(written, 5)
        self.assertEqual(first.content, "hello")
        self.assertEqual(second.content, "hello")
        self.assertEqual(first.flushes, 1)
        self.assertEqual(second.flushes, 1)

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

    def test_wait_action_consumes_turn_without_replying(self):
        participants = build_fake_participants()
        participants["Ops"] = FakeRole("Ops", ['{"action": "wait"}'])
        event_stream = main.RuntimeEventStream()
        session = main.Session(
            participants=participants,
            run_id="session-1",
            event_stream=event_stream,
        )

        with redirect_stdout(StringIO()):
            session.run(initial_message="hello", max_turns=1)

        self.assertEqual(session.transcript[0].response, "")
        self.assertIn("agent.waited", [event.event_type for event in event_stream.events()])

    def test_think_action_records_private_thought_without_replying(self):
        participants = build_fake_participants()
        participants["Ops"] = FakeRole("Ops", ['{"action": "think", "content": "Check tool health later."}'])
        event_stream = main.RuntimeEventStream()
        session = main.Session(
            participants=participants,
            run_id="session-1",
            event_stream=event_stream,
        )

        with redirect_stdout(StringIO()):
            session.run(initial_message="hello", max_turns=1)

        thought_events = [
            event for event in event_stream.events() if event.event_type == "thought.recorded"
        ]
        self.assertEqual(session.transcript[0].response, "")
        self.assertEqual(len(thought_events), 1)
        self.assertEqual(thought_events[0].visibility, "private")
        self.assertEqual(thought_events[0].payload["agent"], "Ops")

    def test_agent_tool_action_executes_without_broadcast_reply(self):
        participants = build_fake_participants()
        participants["HR"] = FakeRole(
            "HR",
            [
                json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )
            ],
        )
        participants["HR"].capabilities = {"hr"}
        participants["HR"].allowed_tools = {"org.*"}
        system = main.System(participants)
        session = main.Session(participants=participants, system=system, run_id="session-1")

        with redirect_stdout(StringIO()):
            main.load_tools(system)
            session.run(initial_message="HR, create QA", max_turns=1)

        self.assertIn("QA", participants)
        self.assertEqual(session.transcript[0].response, "")
        self.assertEqual(len(session.transcript[0].system_events), 1)
        self.assertIn("Created a new role: QA", session.transcript[0].system_events[0])

    def test_plain_text_tool_claim_does_not_create_verified_result(self):
        participants = build_fake_participants()
        participants["HR"] = FakeRole("HR", ["Created QA."])
        system = main.System(participants)
        session = main.Session(participants=participants, system=system, run_id="session-1")

        with redirect_stdout(StringIO()):
            main.load_tools(system)
            session.run(initial_message="HR, create QA", max_turns=1)

        self.assertNotIn("QA", participants)
        self.assertEqual(session.transcript[0].response, "Created QA.")
        self.assertEqual(session.transcript[0].system_events, [])

    def test_agent_tool_action_result_is_returned_to_agent_context(self):
        class ToolActionProvider:
            def generate(self, *_):
                return json.dumps(
                    {
                        "exec": "org.create_role",
                        "args": {
                            "role_name": "QA",
                            "role_description": "Tests the organization",
                        },
                    }
                )

        participants = main.build_employee_dict()
        system = main.System(participants)
        session = main.Session(participants=participants, system=system, run_id="session-1")

        with patch.object(main, "model_provider", ToolActionProvider()):
            with redirect_stdout(StringIO()):
                main.load_tools(system)
                session.run(initial_message="HR, create QA", max_turns=1)

        history = participants["HR"].conversation_history["HR"]
        verified_messages = [
            message["content"]
            for message in history
            if message.get("role") == "system" and "Verified runtime results" in message.get("content", "")
        ]

        self.assertTrue(verified_messages)
        self.assertIn("Created a new role: QA", verified_messages[-1])

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

    def test_exited_role_is_removed_from_scheduler(self):
        participants = build_fake_participants()
        participants["QA"] = FakeRole("QA", ["done"])
        participants["QA"].lifecycle_state = "exited"
        session = main.Session(participants=participants, run_id="session-1")

        session.sync_scheduler_participants()

        self.assertNotIn("QA", session.scheduler.participant_names)

    def test_alarm_tools_create_list_and_cancel_alarm(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            create_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.create_alarm",
                        "args": {
                            "agent_name": "SE",
                            "reminder": "Check build health.",
                            "due_at": "2099-05-08T10:00:00+00:00",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )
            alarm_id = employee_dict["SE"].alarms[0].alarm_id
            list_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.list_alarms",
                        "args": {"agent_name": "SE"},
                    }
                ),
                caller=employee_dict["SE"],
            )
            cancel_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.cancel_alarm",
                        "args": {"agent_name": "SE", "alarm_id": alarm_id},
                    }
                ),
                caller=employee_dict["SE"],
            )

        alarms = json.loads(list_response)

        self.assertIn("Created alarm", create_response)
        self.assertEqual(alarms[0]["reminder"], "Check build health.")
        self.assertIn("Canceled alarm", cancel_response)
        self.assertEqual(employee_dict["SE"].alarms[0].status, "canceled")

    def test_agent_alarm_tools_reject_missing_caller(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.create_alarm",
                        "args": {
                            "agent_name": "SE",
                            "reminder": "Check build health.",
                            "due_at": "2099-05-08T10:00:00+00:00",
                        },
                    }
                )
            )

        self.assertIn("unknown caller", response)
        self.assertEqual(employee_dict["SE"].alarms, [])

    def test_agent_alarm_tools_reject_cross_agent_access(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.create_alarm",
                        "args": {
                            "agent_name": "HR",
                            "reminder": "Check people.",
                            "due_at": "2099-05-08T10:00:00+00:00",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )

        self.assertIn("cannot manage alarms", response)
        self.assertEqual(employee_dict["HR"].alarms, [])

    def test_agent_context_tool_returns_local_runtime_context(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with patch.object(main, "default_location", "Philadelphia, PA"), patch.object(
            main, "default_timezone", "America/New_York"
        ):
            with redirect_stdout(StringIO()):
                main.load_tools(system)
                response = system.interact(
                    json.dumps({"exec": "agent.context", "args": {}}),
                    caller=employee_dict["SE"],
                )

        context = json.loads(response)

        self.assertEqual(context["agent_name"], "SE")
        self.assertEqual(context["location"], "Philadelphia, PA")
        self.assertEqual(context["timezone"], "America/New_York")
        self.assertIn("current_datetime_local", context)

    def test_agent_workspace_tools_write_list_read_and_delete_private_files(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            write_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_write",
                        "args": {
                            "agent_name": "SE",
                            "path": "NOTES.md",
                            "content": "Remember the build plan.",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )
            list_response = system.interact(
                json.dumps({"exec": "agent.files_list", "args": {"agent_name": "SE"}}),
                caller=employee_dict["SE"],
            )
            read_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_read",
                        "args": {"agent_name": "SE", "path": "NOTES.md"},
                    }
                ),
                caller=employee_dict["SE"],
            )
            delete_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_delete",
                        "args": {"agent_name": "SE", "path": "NOTES.md"},
                    }
                ),
                caller=employee_dict["SE"],
            )

        written = json.loads(write_response)
        listed = json.loads(list_response)
        read = json.loads(read_response)
        deleted = json.loads(delete_response)

        self.assertEqual(written["path"], "NOTES.md")
        self.assertEqual(listed[0]["path"], "NOTES.md")
        self.assertEqual(read["content"], "Remember the build plan.")
        self.assertEqual(deleted["kind"], "file")

    def test_agent_workspace_rejects_cross_agent_and_escaping_paths(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            cross_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_write",
                        "args": {
                            "agent_name": "HR",
                            "path": "NOTES.md",
                            "content": "nope",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )
            escape_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_write",
                        "args": {
                            "agent_name": "SE",
                            "path": "../escape.txt",
                            "content": "nope",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )
            windows_escape_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_write",
                        "args": {
                            "agent_name": "SE",
                            "path": "..\\escape.txt",
                            "content": "nope",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )
            negative_read_response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.files_read",
                        "args": {
                            "agent_name": "SE",
                            "path": "NOTES.md",
                            "max_bytes": -1,
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )

        self.assertIn("cannot access workspace", cross_response)
        self.assertIn("Path must be relative", escape_response)
        self.assertIn("POSIX-style separators", windows_escape_response)
        self.assertIn("max_bytes must be between", negative_read_response)

    def test_alarm_creation_rejects_past_due_at(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.load_tools(system)
            response = system.interact(
                json.dumps(
                    {
                        "exec": "agent.create_alarm",
                        "args": {
                            "agent_name": "SE",
                            "reminder": "Check build health.",
                            "due_at": "2023-01-01T00:00:00+00:00",
                        },
                    }
                ),
                caller=employee_dict["SE"],
            )

        self.assertIn("not in the future", response)
        self.assertEqual(employee_dict["SE"].alarms, [])

    def test_memory_tools_search_list_and_expand_accessible_digests(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        original_store = main.memory_store
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(os.path.join(temp_dir, "memory.sqlite3"))
            try:
                store.create_session("session-1")
                store.upsert_agent("SE", "SoftwareEngineer", "SE")
                message_id = store.append_message(
                    "session-1",
                    "SE",
                    "SE",
                    "Robits should preserve source memory.",
                )
                digest_id = store.append_memory_digest(
                    "Digest: preserve source memory.",
                    [{"source_table": "messages", "source_id": message_id}],
                    agent_id="SE",
                    session_id="session-1",
                    digest_type="identity",
                )
                main.memory_store = store
                with redirect_stdout(StringIO()):
                    main.load_tools(system)
                    search_response = system.interact(
                        json.dumps(
                            {
                                "exec": "memory.search",
                                "args": {"agent_name": "SE", "query": "preserve"},
                            }
                        ),
                        caller=employee_dict["SE"],
                    )
                    list_response = system.interact(
                        json.dumps(
                            {
                                "exec": "memory.list_digests",
                                "args": {"agent_name": "SE", "digest_type": "identity"},
                            }
                        ),
                        caller=employee_dict["SE"],
                    )
                    expand_response = system.interact(
                        json.dumps(
                            {
                                "exec": "memory.expand_digest",
                                "args": {"agent_name": "SE", "digest_id": digest_id},
                            }
                        ),
                        caller=employee_dict["SE"],
                    )
            finally:
                main.memory_store = original_store
                store.close()

        search_results = json.loads(search_response)
        digest_results = json.loads(list_response)
        expanded = json.loads(expand_response)

        self.assertTrue(search_results)
        self.assertEqual(digest_results[0]["digest_id"], digest_id)
        self.assertEqual(expanded[0]["record"]["content"], "Robits should preserve source memory.")

    def test_memory_tools_reject_cross_agent_access(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        original_store = main.memory_store
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(os.path.join(temp_dir, "memory.sqlite3"))
            try:
                store.create_session("session-1")
                store.upsert_agent("HR", "HR", "HR")
                store.append_message(
                    "session-1",
                    "HR",
                    "HR",
                    "Private HR memory.",
                )
                main.memory_store = store
                with redirect_stdout(StringIO()):
                    main.load_tools(system)
                    response = system.interact(
                        json.dumps(
                            {
                                "exec": "memory.search",
                                "args": {"agent_name": "HR", "query": "private"},
                            }
                        ),
                        caller=employee_dict["SE"],
                    )
            finally:
                main.memory_store = original_store
                store.close()

        self.assertIn("cannot inspect memory", response)

    def test_memory_tools_reject_missing_caller(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        original_store = main.memory_store
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteMemoryStore(os.path.join(temp_dir, "memory.sqlite3"))
            try:
                store.create_session("session-1")
                store.upsert_agent("SE", "SoftwareEngineer", "SE")
                store.append_message(
                    "session-1",
                    "SE",
                    "SE",
                    "Private SE memory.",
                )
                main.memory_store = store
                with redirect_stdout(StringIO()):
                    main.load_tools(system)
                    response = system.interact(
                        json.dumps(
                            {
                                "exec": "memory.search",
                                "args": {"agent_name": "SE", "query": "private"},
                            }
                        )
                    )
            finally:
                main.memory_store = original_store
                store.close()

        self.assertIn("unknown caller", response)

    def test_due_alarm_is_injected_into_agent_prompt(self):
        participants = build_fake_participants()
        participants["Ops"].alarms = [
            main.Alarm(
                alarm_id="alarm-1",
                agent_name="Ops",
                reminder="Review runtime load.",
                due_at="2026-05-08T10:00:00+00:00",
            )
        ]
        session = main.Session(participants=participants, run_id="session-1")

        with redirect_stdout(StringIO()):
            session.run(initial_message="hello", max_turns=1)

        self.assertIn("Reminder due at", participants["Ops"].received[0][1])
        self.assertEqual(participants["Ops"].alarms[0].status, "completed")

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

    def test_session_records_native_tool_events_in_transcript(self):
        class NativeToolEventProvider:
            def generate(self, role, *_):
                role.runtime_tool_results.append("tool_call.executed: agent__context({}) -> {}")
                return ""

        participants = main.build_employee_dict()
        session = main.Session(participants=participants, run_id="session-1")

        with patch.object(main, "model_provider", NativeToolEventProvider()):
            with redirect_stdout(StringIO()):
                session.run(initial_message="HR, inspect context", max_turns=1)

        self.assertEqual(session.transcript[0].response, "")
        self.assertIn("tool_call.executed: agent__context", session.transcript[0].system_events[0])

    def test_prior_turn_tool_results_not_injected_into_unrelated_agent_prompt(self):
        # Tool results from HR→Ops in a prior turn must not pollute SE's prompt.
        participants = build_fake_participants()
        session = main.Session(participants=participants, run_id="session-1")
        session.transcript.append(
            main.TranscriptEntry(
                turn=1,
                sender="HR",
                receiver="Ops",
                prompt="create QA",
                response="",
                system_events=["tool_call.executed: org.create_role({}) -> Created a new role: QA."],
            )
        )
        session.turns_completed = 1

        with redirect_stdout(StringIO()):
            session.step("SE, what changed?")

        self.assertNotIn("Verified runtime results", participants["SE"].received[0][1])
        self.assertNotIn("Created a new role: QA", participants["SE"].received[0][1])

    def test_loop_detector_halts_on_identical_consecutive_responses(self):
        class GreetingRole:
            """Always replies with the same greeting — simulates a stuck model."""
            def __init__(self, name):
                self.name = name
                self.group_conversation_history = {}
            def interact(self, sender=None, prompt=None):
                return "Hi! How can I help you today?"
            def update_group_conversations(self, msg):
                pass

        participants = {"CEO": GreetingRole("CEO"), "SE": GreetingRole("SE")}
        with patch.object(main._config, "loop_detect_threshold", 3):
            session = main.Session(participants=participants, run_id="loop-test")
            with redirect_stdout(StringIO()):
                session.run(initial_message="Hi!", max_turns=20)
        # Loop detector should halt before max_turns.
        self.assertLess(session.turns_completed, 20)
        loop_events = [
            e for e in session.event_stream._events
            if getattr(e, "event_type", None) == "session.loop_detected"
        ]
        self.assertTrue(loop_events, "Expected session.loop_detected event")

    def test_loop_detector_resets_on_turn_with_tool_calls(self):
        """Loop window must clear when a turn produces system_events (tool calls)."""
        turn = [0]

        class MixedRole:
            def __init__(self, name):
                self.name = name
                self.group_conversation_history = {}
            def interact(self, sender=None, prompt=None):
                turn[0] += 1
                return "Hi! How can I help you today?"
            def update_group_conversations(self, msg):
                pass

        participants = {"CEO": MixedRole("CEO"), "SE": MixedRole("SE")}
        with patch.object(main._config, "loop_detect_threshold", 3):
            session = main.Session(participants=participants, run_id="loop-reset-test")
            # Inject a fake tool event into turn 2 to simulate a turn with tool activity.
            original_step = session.step
            call_count = [0]
            def patched_step(msg):
                call_count[0] += 1
                resp = original_step(msg)
                if call_count[0] == 2 and session.transcript:
                    session.transcript[-1].system_events.append("tool_call.executed: fake({}) -> ok")
                return resp
            session.step = patched_step
            with redirect_stdout(StringIO()):
                session.run(initial_message="Hi!", max_turns=20)
        loop_events = [
            e for e in session.event_stream._events
            if getattr(e, "event_type", None) == "session.loop_detected"
        ]
        # Turn 2 had a tool event so window resets — loop should not fire until
        # 3 more idle turns, requiring more than 4 total turns.
        self.assertGreater(session.turns_completed, 4)
        self.assertTrue(loop_events, "Loop should still eventually be detected")

    def test_session_returns_native_tool_events_to_agent_context(self):
        class NativeToolEventProvider:
            def generate(self, role, *_):
                role.runtime_tool_results.append("tool_call.executed: agent__context({}) -> {}")
                return ""

        participants = main.build_employee_dict()
        session = main.Session(participants=participants, run_id="session-1")

        with patch.object(main, "model_provider", NativeToolEventProvider()):
            with redirect_stdout(StringIO()):
                session.run(initial_message="HR, inspect context", max_turns=1)

        history = participants["HR"].conversation_history["HR"]
        self.assertIn("Verified runtime results", history[-1]["content"])
        self.assertIn("tool_call.executed: agent__context", history[-1]["content"])

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


class BuiltinToolTests(unittest.TestCase):
    """Tests for builtin.* tool implementations."""

    def setUp(self):
        self.workspace_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace_temp.cleanup)

        self.original_workspace = main.agent_workspace_store
        main.agent_workspace_store = main.AgentWorkspaceStore(self.workspace_temp.name)
        self.addCleanup(self._restore)

        # Simulate the tool caller being SE.
        self._prev_caller = main.active_tool_caller
        self._prev_caller_name = main.active_tool_caller_name
        fake_role = SimpleNamespace(name="SE", capabilities={"shell", "mcp", "computer"})
        main.active_tool_caller = fake_role
        main.active_tool_caller_name = "SE"

    def _restore(self):
        main.agent_workspace_store = self.original_workspace
        main.active_tool_caller = self._prev_caller
        main.active_tool_caller_name = self._prev_caller_name

    # ── builtin.web_search ───────────────────────────────────────────────────

    def test_web_search_validates_empty_query(self):
        result = main.builtin_web_search({}, "  ")
        self.assertTrue(result.startswith("Error:"))

    def test_web_search_uses_configured_url(self):
        original = main.builtin_search_url
        try:
            main.builtin_search_url = "http://custom-search.invalid"
            with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
                result = main.builtin_web_search({}, "test query")
        finally:
            main.builtin_search_url = original
        self.assertTrue(result.startswith("Error:"))

    # ── builtin.url_fetch ────────────────────────────────────────────────────

    def test_url_fetch_validates_empty_url(self):
        result = main.builtin_url_fetch({}, "")
        self.assertTrue(result.startswith("Error:"))

    def test_url_fetch_validates_non_http_scheme(self):
        result = main.builtin_url_fetch({}, "ftp://example.com/file")
        self.assertTrue(result.startswith("Error:"))

    def test_url_fetch_validates_non_string_url(self):
        result = main.builtin_url_fetch({}, None)
        self.assertTrue(result.startswith("Error:"))

    def test_url_fetch_validates_invalid_max_chars(self):
        result = main.builtin_url_fetch({}, "https://example.com", max_chars="bad")
        self.assertTrue(result.startswith("Error:"))

    def test_url_fetch_returns_error_on_network_failure(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = main.builtin_url_fetch({}, "https://example.com")
        self.assertTrue(result.startswith("Error:"))

    def test_url_fetch_returns_plain_text_unchanged(self):
        body = b"Hello, plain text world!"

        class FakeResp:
            headers = {"Content-Type": "text/plain"}
            def read(self, n): return body
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            result = main.builtin_url_fetch({}, "https://example.com")
        self.assertEqual(result, "Hello, plain text world!")

    def test_url_fetch_strips_html_tags(self):
        body = b"<html><body><p>Hello <b>world</b></p></body></html>"

        class FakeResp:
            headers = {"Content-Type": "text/html; charset=utf-8"}
            def read(self, n): return body
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            result = main.builtin_url_fetch({}, "https://example.com")
        self.assertNotIn("<b>", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_url_fetch_strips_script_and_style_content(self):
        body = b"<html><head><style>body{color:red}</style><script>alert(1)</script></head><body>Content</body></html>"

        class FakeResp:
            headers = {"Content-Type": "text/html"}
            def read(self, n): return body
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            result = main.builtin_url_fetch({}, "https://example.com")
        self.assertNotIn("color:red", result)
        self.assertNotIn("alert(1)", result)
        self.assertIn("Content", result)

    def test_url_fetch_respects_max_chars(self):
        body = b"A" * 1000

        class FakeResp:
            headers = {"Content-Type": "text/plain"}
            def read(self, n): return body[:n]
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            result = main.builtin_url_fetch({}, "https://example.com", max_chars=50)
        self.assertEqual(len(result), 50)

    # ── builtin.file_search ──────────────────────────────────────────────────

    def test_file_search_finds_text_in_workspace_file(self):
        main.agent_workspace_store.write("SE", "notes.txt", "The quick brown fox jumps.")
        result_json = main.builtin_file_search({}, "SE", "brown fox")
        results = json.loads(result_json)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "notes.txt")
        self.assertIn("brown fox", results[0]["snippet"])

    def test_file_search_returns_empty_for_no_match(self):
        main.agent_workspace_store.write("SE", "notes.txt", "Nothing relevant here.")
        result_json = main.builtin_file_search({}, "SE", "xyzzy12345")
        results = json.loads(result_json)
        self.assertEqual(results, [])

    def test_file_search_blocked_for_different_agent(self):
        result = main.builtin_file_search({}, "HR", "anything")
        self.assertTrue(result.startswith("Error:"))

    def test_file_search_validates_empty_query(self):
        result = main.builtin_file_search({}, "SE", "")
        self.assertTrue(result.startswith("Error:"))

    # ── builtin.shell_run ────────────────────────────────────────────────────

    def test_shell_run_executes_simple_command(self):
        result_json = main.builtin_shell_run({}, "SE", "echo hello", timeout=10)
        result = json.loads(result_json)
        self.assertIn("hello", result["stdout"])
        self.assertEqual(result["returncode"], 0)

    def test_shell_run_captures_nonzero_exit(self):
        result_json = main.builtin_shell_run({}, "SE", "exit 42", timeout=5)
        result = json.loads(result_json)
        self.assertEqual(result["returncode"], 42)

    def test_shell_run_blocked_for_different_agent(self):
        result = main.builtin_shell_run({}, "HR", "echo hi")
        self.assertTrue(result.startswith("Error:"))

    def test_shell_run_validates_empty_command(self):
        result = main.builtin_shell_run({}, "SE", "   ")
        self.assertTrue(result.startswith("Error:"))

    # ── agent.spawn ─────────────────────────────────────────────────────────

    def test_agent_spawn_validates_empty_task(self):
        result = main.agent_spawn({}, "   ")
        self.assertTrue(result.startswith("Error:"))

    def test_agent_spawn_validates_non_string_task(self):
        result = main.agent_spawn({}, None)
        self.assertTrue(result.startswith("Error:"))

    def test_agent_spawn_blocks_memory_tools(self):
        """memory.* tools must be stripped even when explicitly requested."""
        from robits.core.tool_functions import _SUBAGENT_BLOCKED_PREFIXES
        self.assertIn("memory.", _SUBAGENT_BLOCKED_PREFIXES)

    def test_agent_spawn_blocks_agent_spawn_recursion(self):
        """agent.spawn must not be available to sub-agents."""
        from robits.core.tool_functions import _SUBAGENT_BLOCKED_PREFIXES
        self.assertIn("agent.spawn", _SUBAGENT_BLOCKED_PREFIXES)

    def test_agent_spawn_returns_error_when_all_tools_blocked(self):
        result = main.agent_spawn({}, "do something", tools=["memory.search"])
        self.assertTrue(result.startswith("Error:"))

    def test_agent_spawn_rejects_wildcard_tools(self):
        """Wildcard tool entries must be rejected before any filtering."""
        result = main.agent_spawn({}, "do something", tools=["agent.*"])
        self.assertTrue(result.startswith("Error:"), result)

    def test_agent_spawn_rejects_non_string_tools(self):
        """Non-string entries in tools must be rejected."""
        result = main.agent_spawn({}, "do something", tools=[None, 42])
        self.assertTrue(result.startswith("Error:"), result)

    def test_agent_spawn_defaults_include_all_builtins(self):
        """Default tool set should contain all builtin.* tools."""
        from robits.core.tool_functions import _SUBAGENT_DEFAULT_TOOLS
        expected = {
            "builtin.web_search", "builtin.url_fetch", "builtin.file_search",
            "builtin.shell_run", "builtin.tool_search", "builtin.mcp_call",
            "builtin.computer_use", "builtin.image_generation",
        }
        self.assertTrue(expected.issubset(_SUBAGENT_DEFAULT_TOOLS), _SUBAGENT_DEFAULT_TOOLS)

    def test_agent_spawn_calls_chat_completions_provider(self):
        """agent_spawn should call ChatCompletionsProvider.generate() with the task."""
        captured = {}

        def fake_generate(self_provider, sub_role, model, caller, messages):
            captured["messages"] = messages
            captured["allowed_tools"] = sub_role.allowed_tools
            return "sub-agent result"

        from unittest.mock import patch
        with patch("robits.core.providers.ChatCompletionsProvider.generate", fake_generate):
            result = main.agent_spawn({}, "fetch https://example.com and summarise it")
        self.assertEqual(result, "sub-agent result")
        user_msgs = [m for m in captured["messages"] if m["role"] == "user"]
        self.assertTrue(any("fetch https://example.com" in m["content"] for m in user_msgs))

    def test_agent_spawn_default_tools_exclude_memory_and_org(self):
        """Default sub-agent tool set must not contain memory or org tools."""
        from robits.core.tool_functions import _SUBAGENT_DEFAULT_TOOLS
        for tool in _SUBAGENT_DEFAULT_TOOLS:
            self.assertFalse(
                tool.startswith("memory.") or tool.startswith("org."),
                f"Default sub-agent tool {tool!r} should not be in memory or org namespace",
            )

    def test_agent_spawn_custom_tools_merged_and_filtered(self):
        """Caller-supplied tools replace defaults; blocked prefixes are still stripped."""
        captured = {}

        def fake_generate(self_provider, sub_role, model, caller, messages):
            captured["allowed_tools"] = sub_role.allowed_tools
            return "done"

        from unittest.mock import patch
        with patch("robits.core.providers.ChatCompletionsProvider.generate", fake_generate):
            main.agent_spawn(
                {},
                "run a search",
                tools=["builtin.web_search", "memory.search", "builtin.url_fetch"],
            )
        self.assertIn("builtin.web_search", captured["allowed_tools"])
        self.assertIn("builtin.url_fetch", captured["allowed_tools"])
        self.assertNotIn("memory.search", captured["allowed_tools"])

    def test_agent_spawn_injects_shell_hint_for_shell_task(self):
        """System prompt should mention builtin.shell_run when task starts with python/bash."""
        captured = {}

        def fake_generate(self_provider, sub_role, model, caller, messages):
            captured["system"] = next(m["content"] for m in messages if m["role"] == "system")
            return "42"

        from unittest.mock import patch
        with patch("robits.core.providers.ChatCompletionsProvider.generate", fake_generate):
            main.agent_spawn({}, "python3 -c 'print(6*7)'")
        self.assertIn("builtin.shell_run", captured["system"])

    def test_agent_spawn_no_shell_hint_for_plain_task(self):
        """System prompt should NOT add the shell hint for non-shell tasks."""
        captured = {}

        def fake_generate(self_provider, sub_role, model, caller, messages):
            captured["system"] = next(m["content"] for m in messages if m["role"] == "system")
            return "ok"

        from unittest.mock import patch
        with patch("robits.core.providers.ChatCompletionsProvider.generate", fake_generate):
            main.agent_spawn({}, "summarise the readme")
        self.assertNotIn("builtin.shell_run", captured["system"])

    # ── builtin.tool_search ──────────────────────────────────────────────────

    def test_tool_search_finds_tools_by_name_fragment(self):
        main.tool_registry.clear()
        with open("tools.yaml", encoding="utf-8") as f:
            import yaml
            for entry in yaml.safe_load(f):
                try:
                    main.tool_registry.register_definition(entry)
                except Exception:
                    pass
        # Grant all tools so builtin_tool_search (which filters to allowed tools) finds results.
        main.active_tool_caller.allowed_tools = {"builtin.*"}
        result_json = main.builtin_tool_search({}, "web_search")
        results = json.loads(result_json)
        names = [r["name"] for r in results]
        self.assertIn("builtin.web_search", names)
        main.tool_registry.clear()

    def test_tool_search_validates_empty_query(self):
        result = main.builtin_tool_search({}, "")
        self.assertTrue(result.startswith("Error:"))

    def test_tool_search_returns_empty_list_for_no_match(self):
        result_json = main.builtin_tool_search({}, "xyzzy_no_such_tool_xyz")
        results = json.loads(result_json)
        self.assertEqual(results, [])

    # ── not-implemented stubs ────────────────────────────────────────────────

    def test_mcp_call_returns_not_implemented(self):
        result = main.builtin_mcp_call({}, "http://example.com/mcp", "some_tool")
        self.assertTrue(result.startswith("Error:"))

    def test_computer_use_returns_not_implemented(self):
        result = main.builtin_computer_use({}, "screenshot")
        self.assertTrue(result.startswith("Error:"))

    def test_image_generation_returns_not_implemented(self):
        result = main.builtin_image_generation({}, "a blue sky")
        self.assertTrue(result.startswith("Error:"))

    # ── tools.yaml integration ───────────────────────────────────────────────

    def test_builtin_tools_load_from_yaml_without_error(self):
        """All builtin.* entries must compile successfully."""
        import yaml
        main.tool_registry.clear()
        errors = []
        with open("tools.yaml", encoding="utf-8") as f:
            for entry in yaml.safe_load(f):
                try:
                    main.tool_registry.register_definition(entry)
                except Exception as exc:
                    errors.append(f"{entry.get('name')}: {exc}")
        builtin_names = [n for n in main.tool_registry._tools if n.startswith("builtin.")]
        self.assertEqual(
            sorted(builtin_names),
            sorted([
                "builtin.web_search",
                "builtin.url_fetch",
                "builtin.file_search",
                "builtin.shell_run",
                "builtin.tool_search",
                "builtin.mcp_call",
                "builtin.computer_use",
                "builtin.image_generation",
            ]),
        )
        self.assertEqual(errors, [], f"Tool registration errors: {errors}")
        main.tool_registry.clear()

    def test_shell_run_requires_shell_capability(self):
        """Role without shell capability must be denied."""
        main.tool_registry.clear()
        import yaml
        with open("tools.yaml", encoding="utf-8") as f:
            for entry in yaml.safe_load(f):
                try:
                    main.tool_registry.register_definition(entry)
                except Exception:
                    pass
        # Caller has no shell capability.
        main.active_tool_caller.capabilities = set()
        result = main.tool_registry.execute(
            "builtin.shell_run",
            {"agent_name": "SE", "command": "echo hi"},
            {},
            caller=main.active_tool_caller,
        )
        self.assertTrue(result.startswith("Error:"))
        main.tool_registry.clear()


class MemoryToolTests(unittest.TestCase):
    """Tests for memory_search, memory_list_digests, memory_expand_digest in main.py."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.workspace_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.addCleanup(self.workspace_temp.cleanup)

        from robits.memory.sqlite import SQLiteMemoryStore

        self.store = SQLiteMemoryStore(
            Path(self.store_temp.name) / "mem.sqlite3"
        )
        self.addCleanup(self.store.close)

        self.original_store = main.memory_store
        self.original_workspace = main.agent_workspace_store
        main.memory_store = self.store
        main.agent_workspace_store = main.AgentWorkspaceStore(self.workspace_temp.name)
        self.addCleanup(self._restore)

        # Seed baseline records.
        self.store.create_session("s1")
        self.store.upsert_agent("SE", "SoftwareEngineer", "SE")

        # Make main think the tool caller is SE.
        self._prev_caller = main.active_tool_caller
        self._prev_caller_name = main.active_tool_caller_name
        fake_role = SimpleNamespace(name="SE", capabilities=set())
        main.active_tool_caller = fake_role
        main.active_tool_caller_name = "SE"
        self.fake_role = fake_role

    def _restore(self):
        main.memory_store = self.original_store
        main.agent_workspace_store = self.original_workspace
        main.active_tool_caller = self._prev_caller
        main.active_tool_caller_name = self._prev_caller_name

    # ── cascade search ────────────────────────────────────────────────────────

    def test_memory_search_cascade_true_surfaces_parent_digest(self):
        msg_id = self.store.append_message("s1", "SE", "SE", "cascade content unique_kw")
        digest_id = self.store.append_memory_digest(
            "Digest summary.",
            [{"source_table": "messages", "source_id": msg_id}],
            agent_id="SE",
            session_id="s1",
        )

        result_json = main.memory_search({}, "SE", "unique_kw", cascade=True)
        results = json.loads(result_json)

        kinds = {r["kind"] for r in results}
        self.assertIn("memory_digest", kinds)

    def test_memory_search_cascade_false_returns_only_outer(self):
        msg_id = self.store.append_message("s1", "SE", "SE", "inner only content abc123")
        self.store.append_memory_digest(
            "Digest with inner content.",
            [{"source_table": "messages", "source_id": msg_id}],
            agent_id="SE",
            session_id="s1",
        )

        result_json = main.memory_search({}, "SE", "abc123", cascade=False)
        results = json.loads(result_json)

        # Without cascade, only the raw message hit is returned, not the digest.
        kinds = {r["kind"] for r in results}
        self.assertIn("message", kinds)
        self.assertNotIn("memory_digest", kinds)

    # ── large result offloading ───────────────────────────────────────────────

    def test_condense_if_large_returns_raw_when_small(self):
        small = json.dumps({"ok": True})
        result = main._condense_if_large("SE", "memory_search", small)
        self.assertEqual(result, small)

    def test_condense_if_large_caches_and_returns_snippet_when_big(self):
        big = json.dumps({"data": "x" * (main.memory_cache_threshold + 1)})
        original_threshold = main.memory_cache_threshold
        # Lower threshold temporarily.
        main.memory_cache_threshold = 64
        try:
            padded = json.dumps({"data": "y" * 200})
            result_json = main._condense_if_large("SE", "memory_search", padded)
        finally:
            main.memory_cache_threshold = original_threshold

        result = json.loads(result_json)
        self.assertTrue(result.get("truncated"))
        self.assertIn("snippet", result)
        self.assertIn("cache_path", result)
        self.assertIn(".memory-cache/", result["cache_path"])

    def test_large_memory_search_result_is_offloaded_to_workspace(self):
        # Insert many records so the result JSON exceeds the cache threshold.
        for i in range(20):
            self.store.append_message("s1", "SE", "SE", f"searchable content record number {i} verbosity padding")

        original_threshold = main.memory_cache_threshold
        main.memory_cache_threshold = 64
        main.tool_registry.clear()
        try:
            result_json = main.memory_search({}, "SE", "searchable content", limit=20)
            result = json.loads(result_json)
        finally:
            main.memory_cache_threshold = original_threshold

        self.assertTrue(result.get("truncated"))
        cache_path = result.get("cache_path")
        self.assertIsNotNone(cache_path)

        # The full result should be readable from the workspace.
        full = main.agent_workspace_store.read("SE", cache_path)
        self.assertFalse(full["truncated"])
        self.assertIn("searchable content", full["content"])

    # ── expand digest depth limit ─────────────────────────────────────────────

    def test_expand_digest_respects_max_depth_param(self):
        msg_id = self.store.append_message("s1", "SE", "SE", "leaf node content")
        d1 = self.store.append_memory_digest(
            "Depth-1 digest.",
            [{"source_table": "messages", "source_id": msg_id}],
            agent_id="SE",
            session_id="s1",
        )
        d2 = self.store.append_memory_digest(
            "Depth-2 digest.",
            [{"source_table": "memory_digests", "source_id": d1}],
            agent_id="SE",
            session_id="s1",
        )

        result_json = main.memory_expand_digest({}, "SE", d2, recursive=True, max_depth=0)
        rows = json.loads(result_json)

        tables = [r["source_table"] for r in rows]
        self.assertIn("memory_digests", tables)
        self.assertNotIn("messages", tables)

    def test_expand_digest_max_depth_capped_by_global_limit(self):
        original = main.memory_max_depth
        main.memory_max_depth = 1
        try:
            msg_id = self.store.append_message("s1", "SE", "SE", "deep leaf")
            d1 = self.store.append_memory_digest(
                "d1", [{"source_table": "messages", "source_id": msg_id}], agent_id="SE"
            )
            d2 = self.store.append_memory_digest(
                "d2", [{"source_table": "memory_digests", "source_id": d1}], agent_id="SE"
            )
            d3 = self.store.append_memory_digest(
                "d3", [{"source_table": "memory_digests", "source_id": d2}], agent_id="SE"
            )

            # Ask for max_depth=99 but global limit is 1.
            result_json = main.memory_expand_digest({}, "SE", d3, recursive=True, max_depth=99)
            rows = json.loads(result_json)
        finally:
            main.memory_max_depth = original

        tables = [r["source_table"] for r in rows]
        # At global max_depth=1 we see d2 (depth 0) and d1 (depth 1) but not messages (depth 2).
        self.assertIn("memory_digests", tables)
        self.assertNotIn("messages", tables)

    # ── agent isolation ───────────────────────────────────────────────────────

    def test_memory_search_blocked_for_different_agent(self):
        self.store.upsert_agent("HR", "HR", "HR")
        self.store.append_message("s1", "HR", "HR", "private HR content")

        # Caller is SE, trying to inspect HR's memory.
        result = main.memory_search({}, "HR", "private HR content")
        self.assertTrue(result.startswith("Error:"))

    def test_memory_list_digests_blocked_for_different_agent(self):
        self.store.upsert_agent("HR", "HR", "HR")
        result = main.memory_list_digests({}, "HR")
        self.assertTrue(result.startswith("Error:"))

    def test_memory_expand_digest_blocked_for_different_agent_digest(self):
        self.store.upsert_agent("HR", "HR", "HR")
        msg_id = self.store.append_message("s1", "HR", "HR", "HR private.")
        digest_id = self.store.append_memory_digest(
            "HR digest.", [{"source_table": "messages", "source_id": msg_id}], agent_id="HR"
        )
        result = main.memory_expand_digest({}, "HR", digest_id)
        self.assertTrue(result.startswith("Error:"))


class SessionMemoryCaptureTests(unittest.TestCase):
    """Tests that Session persists messages, thoughts, tool calls, and lifecycle events."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "mem.sqlite3")
        self.addCleanup(self.store.close)

        self.original_store = main.memory_store
        self.original_digest_interval = main.memory_digest_interval
        self.original_digest_context_chars = main.memory_digest_context_chars
        self.original_digest_elapsed_seconds = main.memory_digest_elapsed_seconds
        self.original_identity_digest_interval = main.memory_identity_digest_interval
        self.original_goal_digest_interval = main.memory_goal_digest_interval
        main.memory_store = self.store
        main.memory_digest_interval = 0
        main.memory_digest_context_chars = 0
        main.memory_digest_elapsed_seconds = 0
        main.memory_identity_digest_interval = 0
        main.memory_goal_digest_interval = 0
        self.addCleanup(self._restore)

    def _restore(self):
        main.memory_store = self.original_store
        main.memory_digest_interval = self.original_digest_interval
        main.memory_digest_context_chars = self.original_digest_context_chars
        main.memory_digest_elapsed_seconds = self.original_digest_elapsed_seconds
        main.memory_identity_digest_interval = self.original_identity_digest_interval
        main.memory_goal_digest_interval = self.original_goal_digest_interval

    def _make_session(self):
        # "SE" key with name="SoftwareEngineer" exercises key≠name FK resolution.
        participants = {
            "A": SimpleNamespace(
                name="A",
                lifecycle_state="active",
                interact=lambda *a, **kw: "hello",
                update_group_conversations=lambda m: None,
                runtime_tool_results=[],
            ),
            "SE": SimpleNamespace(
                name="SoftwareEngineer",
                lifecycle_state="active",
                interact=lambda *a, **kw: "world",
                update_group_conversations=lambda m: None,
                runtime_tool_results=[],
            ),
        }
        system = SimpleNamespace(interact=lambda *a, **kw: None)
        scheduler = main.RoundRobinScheduler(list(participants))
        return main.Session(
            participants=participants,
            system=system,
            scheduler=scheduler,
        )

    def test_session_creation_is_persisted(self):
        session = self._make_session()
        row = self.store.connection.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session.run_id,)
        ).fetchone()
        self.assertIsNotNone(row)

    def test_agents_are_upserted_on_session_init(self):
        session = self._make_session()
        agents = self.store.connection.execute(
            "SELECT agent_id FROM agents WHERE agent_id IN ('A', 'SE')"
        ).fetchall()
        self.assertEqual({r["agent_id"] for r in agents}, {"A", "SE"})

    def test_turn_messages_are_persisted(self):
        session = self._make_session()
        session.record_turn("A", "SE", "hello prompt", "world response")
        rows = self.store.connection.execute(
            "SELECT content FROM messages WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        contents = {r["content"] for r in rows}
        self.assertIn("hello prompt", contents)
        self.assertIn("world response", contents)

    def test_empty_prompt_and_response_not_persisted(self):
        session = self._make_session()
        session.record_turn("A", "SE", "", "")
        rows = self.store.connection.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session.run_id,)
        ).fetchone()
        self.assertEqual(rows["n"], 0)

    def test_passive_turn_prompt_not_persisted_as_memory_message(self):
        session = self._make_session()
        session.record_turn("A", "SE", "already expressed prompt", "")
        rows = self.store.connection.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session.run_id,)
        ).fetchone()
        self.assertEqual(rows["n"], 0)
        self.assertFalse(session.transcript[-1].memory_recorded)

    def test_thought_is_persisted(self):
        session = self._make_session()
        session.record_thought("A", "my private thought")
        rows = self.store.connection.execute(
            "SELECT content, visibility FROM thoughts WHERE agent_id = 'A'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "my private thought")
        self.assertEqual(rows[0]["visibility"], "private")

    def test_model_thinking_stored_as_thought_linked_to_response_message(self):
        session = self._make_session()
        session.record_turn("A", "SE", "what color?", "Green.", thinking="I considered blue but chose green.")
        thought_rows = self.store.connection.execute(
            "SELECT content, source, metadata_json FROM thoughts WHERE agent_id = 'SE'"
        ).fetchall()
        self.assertEqual(len(thought_rows), 1)
        self.assertEqual(thought_rows[0]["content"], "I considered blue but chose green.")
        self.assertEqual(thought_rows[0]["source"], "model_thinking")
        import json as _json
        meta = _json.loads(thought_rows[0]["metadata_json"])
        # linked_message_id should point to the response message row
        self.assertIn("linked_message_id", meta)
        response_msg = self.store.connection.execute(
            "SELECT content FROM messages WHERE message_id = ?",
            (meta["linked_message_id"],),
        ).fetchone()
        self.assertIsNotNone(response_msg)
        self.assertEqual(response_msg["content"], "Green.")

    def test_model_thinking_not_stored_when_no_response(self):
        session = self._make_session()
        session.record_turn("A", "SE", "hello", "", thinking="silent thought")
        thought_rows = self.store.connection.execute(
            "SELECT COUNT(*) AS n FROM thoughts WHERE source = 'model_thinking'"
        ).fetchone()
        self.assertEqual(thought_rows["n"], 0)

    def test_session_ended_at_set_on_run_completion(self):
        session = self._make_session()
        session.turns_completed = 0
        session.run(initial_message="go", max_turns=1)
        row = self.store.connection.execute(
            "SELECT ended_at FROM sessions WHERE session_id = ?", (session.run_id,)
        ).fetchone()
        self.assertIsNotNone(row["ended_at"])

    def test_auto_digest_fires_at_interval(self):
        main.memory_digest_interval = 2
        session = self._make_session()
        session.record_turn("A", "SE", "turn one prompt", "turn one response")
        session.record_turn("A", "SE", "turn two prompt", "turn two response")
        rows = self.store.connection.execute(
            "SELECT digest_id FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertGreater(len(rows), 0)

    def test_auto_digest_not_fired_before_interval(self):
        main.memory_digest_interval = 5
        session = self._make_session()
        session.record_turn("A", "SE", "only one turn", "response")
        rows = self.store.connection.execute(
            "SELECT digest_id FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertEqual(len(rows), 0)

    def test_auto_digest_fires_on_context_char_threshold(self):
        main.memory_digest_context_chars = 20
        session = self._make_session()
        session.record_turn("A", "SE", "short", "this response is long enough")
        rows = self.store.connection.execute(
            "SELECT metadata_json FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertGreater(len(rows), 0)
        self.assertIn("context_chars", rows[0]["metadata_json"])

    def test_auto_digest_fires_on_elapsed_activity_threshold(self):
        main.memory_digest_elapsed_seconds = 1
        session = self._make_session()
        session._last_digest_at -= 2
        session.record_turn("A", "SE", "elapsed prompt", "elapsed response")
        rows = self.store.connection.execute(
            "SELECT metadata_json FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertGreater(len(rows), 0)
        self.assertIn("elapsed_seconds", rows[0]["metadata_json"])

    def test_passive_turn_does_not_trigger_auto_digest(self):
        main.memory_digest_context_chars = 1
        session = self._make_session()
        session.record_turn("A", "SE", "large but passive prompt", "")
        rows = self.store.connection.execute(
            "SELECT digest_id FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertEqual(len(rows), 0)

    def test_identity_and_goal_interval_digests_fire(self):
        main.memory_identity_digest_interval = 1
        main.memory_goal_digest_interval = 1
        session = self._make_session()
        session.record_turn("A", "SE", "identity prompt", "goal response")
        rows = self.store.connection.execute(
            "SELECT digest_type FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        digest_types = {row["digest_type"] for row in rows}
        self.assertIn("identity", digest_types)
        self.assertIn("goal_short_term", digest_types)

    def test_passive_turn_does_not_advance_identity_or_goal_checkpoint_intervals(self):
        main.memory_identity_digest_interval = 2
        main.memory_goal_digest_interval = 2
        session = self._make_session()
        session.record_turn("A", "SE", "first meaningful prompt", "first response")
        session.record_turn("A", "SE", "passive prompt", "")
        rows = self.store.connection.execute(
            "SELECT digest_type FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertNotIn("identity", {row["digest_type"] for row in rows})
        self.assertNotIn("goal_short_term", {row["digest_type"] for row in rows})

        session.record_turn("A", "SE", "second meaningful prompt", "second response")
        rows = self.store.connection.execute(
            "SELECT digest_type FROM memory_digests WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        digest_types = {row["digest_type"] for row in rows}
        self.assertIn("identity", digest_types)
        self.assertIn("goal_short_term", digest_types)

    def test_mismatched_key_name_uses_participant_key_for_messages(self):
        """SE key with role.name='SoftwareEngineer' must persist under key 'SE'."""
        session = self._make_session()
        # record_turn with role.name values — should be resolved to participant keys.
        session.record_turn("SoftwareEngineer", "A", "hi from SE", "hi back")
        rows = self.store.connection.execute(
            "SELECT sender_agent_id, receiver_agent_id FROM messages WHERE session_id = ?",
            (session.run_id,),
        ).fetchall()
        agent_ids = {r["sender_agent_id"] for r in rows} | {r["receiver_agent_id"] for r in rows}
        self.assertIn("SE", agent_ids)
        self.assertNotIn("SoftwareEngineer", agent_ids)

    def test_mismatched_key_name_uses_participant_key_for_thoughts(self):
        """Thought recorded under role.name='SoftwareEngineer' must land on key 'SE'."""
        session = self._make_session()
        session.record_thought("SoftwareEngineer", "thinking hard")
        rows = self.store.connection.execute(
            "SELECT agent_id FROM thoughts WHERE session_id = ?", (session.run_id,)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agent_id"], "SE")

    def test_lifecycle_event_persisted_as_memory_entry(self):
        role = SimpleNamespace(
            name="NewRole",
            lifecycle_state="active",
            lifecycle_events=[],
        )
        main.record_lifecycle_event(role, "create", "active", requested_by="CEO")
        rows = self.store.connection.execute(
            "SELECT content FROM memory_entries WHERE agent_id = 'NewRole' AND kind = 'lifecycle'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("create", rows[0]["content"])
        self.assertIn("active", rows[0]["content"])

    def test_tool_call_persisted_on_executed_event(self):
        role = SimpleNamespace(
            name="SE",
            runtime_event_stream=None,
            runtime_session_id="test-session",
        )
        self.store.create_session("test-session")
        self.store.upsert_agent("SE", "SoftwareEngineer", "SE")
        main._emit_role_tool_event(role, "tool_call.executed", {
            "call_id": "call-1",
            "tool_name": "org.create_role",
            "arguments": {"role_name": "QA"},
            "result": "Created role QA",
        })
        rows = self.store.connection.execute(
            "SELECT tool_name, status FROM tool_calls WHERE agent_id = 'SE'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool_name"], "org.create_role")
        self.assertEqual(rows[0]["status"], "executed")


class OrgChatContextTests(unittest.TestCase):
    """Tests for format_org_chat_context and JSONL write."""

    def test_empty_transcript_returns_empty_string(self):
        self.assertEqual(main.format_org_chat_context([], 10), "")

    def test_zero_limit_returns_empty_string(self):
        entries = [main.TranscriptEntry(1, "A", "B", "hello", "world")]
        self.assertEqual(main.format_org_chat_context(entries, 0), "")

    def test_formats_recent_entries(self):
        entries = [
            main.TranscriptEntry(1, "Alice", "Bob", "question", "answer"),
            main.TranscriptEntry(2, "Bob", "Alice", "follow-up", "ok"),
        ]
        result = main.format_org_chat_context(entries, 5)
        self.assertIn("Turn 1", result)
        self.assertIn("Alice", result)
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("Turn 2", result)

    def test_limit_truncates_to_last_n_entries(self):
        entries = [main.TranscriptEntry(i, "A", "B", f"msg{i}", "") for i in range(1, 6)]
        result = main.format_org_chat_context(entries, 2)
        self.assertNotIn("msg1", result)
        self.assertIn("msg4", result)
        self.assertIn("msg5", result)

    def test_prompt_truncated_at_400_chars(self):
        long_prompt = "x" * 600
        entries = [main.TranscriptEntry(1, "A", "B", long_prompt, "")]
        result = main.format_org_chat_context(entries, 5)
        self.assertNotIn("x" * 401, result)

    def test_write_org_chat_jsonl_writes_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = main.AgentWorkspaceStore(tmp)
            session = main.Session(
                participants={"A": SimpleNamespace(
                    name="A", lifecycle_state="active",
                    interact=lambda *a, **kw: "hi",
                    update_group_conversations=lambda m: None,
                )},
            )
            session._org_workspace = store
            entry = main.TranscriptEntry(1, "A", "B", "hello", "world")
            session._write_org_chat_jsonl(entry)
            result = store.read("org", "org_chat.jsonl")
            parsed = json.loads(result["content"].strip())
            self.assertEqual(parsed["turn"], 1)
            self.assertEqual(parsed["sender"], "A")

    def test_org_chat_injected_in_step(self):
        received_prompts = []

        class CapturingRole:
            name = "B"
            lifecycle_state = "active"
            conversation_history = {}
            template = ""
            max_tokens = 100
            temperature = 0.0
            runtime_tool_results = []
            runtime_event_stream = None
            runtime_session_id = None
            runtime_role_name = None
            allowed_tools = set()

            def interact(self, sender=None, prompt=None):
                received_prompts.append(prompt or "")
                return "response"

            def update_group_conversations(self, m):
                pass

        a = SimpleNamespace(
            name="A", lifecycle_state="active",
            interact=lambda *a, **kw: "initial",
            update_group_conversations=lambda m: None,
        )
        b = CapturingRole()
        old_lines = main.org_chat_context_lines
        main.org_chat_context_lines = 5
        try:
            session = main.Session(participants={"A": a, "B": b})
            # seed transcript with one entry manually
            session.transcript.append(main.TranscriptEntry(1, "A", "B", "seed prompt", "seed response"))
            session.turns_completed = 1
            session.last_receiver = a
            session.step("hello")
        finally:
            main.org_chat_context_lines = old_lines

        self.assertTrue(any("Recent org chat" in p for p in received_prompts))


class OrgDigestTests(unittest.TestCase):
    """Tests for _auto_org_digest and list_recent_message_ids_by_channel."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "org.sqlite3")
        self.addCleanup(self.store.close)
        self.original_store = main.memory_store
        self.original_org_interval = main.org_digest_interval
        main.memory_store = self.store
        main.org_digest_interval = 2
        self.addCleanup(self._restore)

    def _restore(self):
        main.memory_store = self.original_store
        main.org_digest_interval = self.original_org_interval

    def _make_session(self):
        participants = {
            "X": SimpleNamespace(
                name="X", lifecycle_state="active",
                interact=lambda *a, **kw: "resp",
                update_group_conversations=lambda m: None,
            ),
            "Y": SimpleNamespace(
                name="Y", lifecycle_state="active",
                interact=lambda *a, **kw: "resp",
                update_group_conversations=lambda m: None,
            ),
        }
        return main.Session(participants=participants)

    def test_messages_tagged_with_org_chat_channel(self):
        session = self._make_session()
        session.record_turn("X", "Y", "hello", "world")
        rows = self.store.connection.execute(
            "SELECT channel_id FROM messages WHERE session_id=?", (session.run_id,)
        ).fetchall()
        # All messages should reference the org_chat channel
        self.assertTrue(all(r["channel_id"] == session._org_chat_channel_id for r in rows))

    def test_list_recent_message_ids_by_channel_filters_correctly(self):
        session = self._make_session()
        session.record_turn("X", "Y", "msg1", "resp1")
        ids = self.store.list_recent_message_ids_by_channel(
            session.run_id, 10, session._org_chat_channel_id
        )
        self.assertGreater(len(ids), 0)
        # Non-matching channel returns empty
        ids_other = self.store.list_recent_message_ids_by_channel(session.run_id, 10, 9999)
        self.assertEqual(ids_other, [])

    def test_org_digest_fires_at_interval(self):
        session = self._make_session()
        # Two turns should trigger digest (interval=2)
        session.record_turn("X", "Y", "first", "one")
        session.record_turn("Y", "X", "second", "two")
        digests = self.store.list_memory_digests(
            agent_id="X", digest_type="episodic", limit=10
        )
        org_digests = [d for d in digests if d.get("conversation_type") == "org_chat"]
        self.assertEqual(len(org_digests), 1)
        self.assertIn("Org chat digest", org_digests[0]["content"])

    def test_org_digest_not_fired_before_interval(self):
        session = self._make_session()
        session.record_turn("X", "Y", "only one turn", "yes")
        digests = self.store.list_memory_digests(
            agent_id="X", digest_type="episodic", limit=10
        )
        org_digests = [d for d in digests if d.get("conversation_type") == "org_chat"]
        self.assertEqual(len(org_digests), 0)


class OrgChatReadToolFixTests(unittest.TestCase):
    """Tests for org_chat_read sandbox inclusion and kwargs fix."""

    def setUp(self):
        self._old_org_workspace = main._org_workspace
        main._org_workspace = None

    def tearDown(self):
        main._org_workspace = self._old_org_workspace

    def test_org_chat_read_in_tool_exec_globals(self):
        """org_chat_read must be reachable from compiled tool code (not raise NameError)."""
        # Compile a tiny tool that calls org_chat_read and verify it doesn't NameError.
        func = main.tool_registry.compile_tool(
            "_test_org_read", ["limit"], [], "return org_chat_read(employee_dict, limit=1)"
        )
        old = main._org_workspace
        main._org_workspace = None
        try:
            result = json.loads(func(employee_dict={}))
        finally:
            main._org_workspace = old
        self.assertIn("note", result)

    def test_org_chat_read_limit_zero_safe(self):
        result = json.loads(main.org_chat_read({}, limit=0))
        self.assertEqual(result["lines"], [])

    def test_org_chat_read_negative_limit_safe(self):
        result = json.loads(main.org_chat_read({}, limit=-5))
        self.assertEqual(result["lines"], [])

    def test_org_context_not_stored_in_transcript(self):
        """Org chat context should be injected for the model but not persisted in transcript."""
        received_model_prompts = []

        class CapturingRole:
            name = "B"
            lifecycle_state = "active"
            conversation_history = {}
            template = ""
            max_tokens = 100
            temperature = 0.0
            runtime_tool_results = []
            runtime_event_stream = None
            runtime_session_id = None
            runtime_role_name = None
            runtime_clock_state = "on"
            allowed_tools = set()

            def interact(self, sender=None, prompt=None):
                received_model_prompts.append(prompt or "")
                return "answer"

            def update_group_conversations(self, m):
                pass

        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "initial",
                            update_group_conversations=lambda m: None)
        b = CapturingRole()
        old_lines = main.org_chat_context_lines
        main.org_chat_context_lines = 5
        try:
            session = main.Session(participants={"A": a, "B": b}, clock_state="on")
            session.transcript.append(main.TranscriptEntry(1, "A", "B", "raw seed", "seed resp"))
            session.turns_completed = 1
            session.last_receiver = a
            session.step("next message")
        finally:
            main.org_chat_context_lines = old_lines

        # Model prompt should contain org context
        self.assertTrue(any("Recent org chat" in p for p in received_model_prompts))
        # But stored transcript prompt should NOT contain org context
        last_entry = session.transcript[-1]
        self.assertNotIn("Recent org chat", last_entry.prompt)

    def test_prepare_agent_runtime_sets_clock_state(self):
        """prepare_agent_runtime must stamp runtime_clock_state so interact() can read it."""
        role = SimpleNamespace(name="B", lifecycle_state="active",
                               interact=lambda *a, **kw: "hi",
                               update_group_conversations=lambda m: None)
        session = main.Session(participants={"A": SimpleNamespace(
            name="A", lifecycle_state="active",
            interact=lambda *a, **kw: "hi",
            update_group_conversations=lambda m: None,
        ), "B": role}, clock_state="off")
        session.prepare_agent_runtime(role)
        self.assertEqual(getattr(role, "runtime_clock_state", None), "off")

    def test_interact_uses_runtime_clock_state_for_guidance(self):
        """main.interact() picks guidance from role.runtime_clock_state, not module global."""
        captured = []

        def fake_generate(role, model, sender, messages):
            # Capture the system message content
            if messages and messages[0].get("role") == "system":
                captured.append(messages[0]["content"])
            return "ok"

        role = SimpleNamespace(
            name="TestBot",
            template="You are helpful.",
            conversation_history={},
            max_tokens=100,
            temperature=0.0,
            employee_dict={},
            runtime_clock_state="off",
            runtime_session_id=None,
            runtime_role_name=None,
            runtime_event_stream=None,
        )
        old_provider = main.model_provider
        main.model_provider = SimpleNamespace(generate=fake_generate)
        try:
            main.interact(role, "gpt-4o-mini", "user", "hello")
        finally:
            main.model_provider = old_provider
        self.assertTrue(any("off the clock" in c for c in captured))


class ReasoningEffortTests(unittest.TestCase):
    """Tests that ResponsesProvider forwards reasoning_effort when set."""

    def _make_fake_response(self, text="done"):
        output = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])
        return SimpleNamespace(id="resp-1", output=[output])

    def _make_provider_and_role(self, role_effort=None):
        role = SimpleNamespace(
            name="TestRole",
            max_tokens=100,
            temperature=0.0,
            employee_dict=None,
            allowed_tools=set(),
            reasoning_effort=role_effort,
        )
        registry = SimpleNamespace(
            as_responses_tools=lambda r: [],
        )
        return role, registry

    def test_valid_effort_forwarded_to_api(self):
        role, registry = self._make_provider_and_role()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return self._make_fake_response()

        provider = main.ResponsesProvider.__new__(main.ResponsesProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        old_effort = main._reasoning_effort_env
        main._reasoning_effort_env = "low"
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                provider.generate(role, "gpt-4o", "sender", [])
        finally:
            main._reasoning_effort_env = old_effort

        self.assertEqual(captured.get("reasoning"), {"effort": "low"})

    def test_invalid_effort_not_forwarded(self):
        role, registry = self._make_provider_and_role()
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return self._make_fake_response()

        provider = main.ResponsesProvider.__new__(main.ResponsesProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        old_effort = main._reasoning_effort_env
        main._reasoning_effort_env = "ultra"
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                provider.generate(role, "gpt-4o", "sender", [])
        finally:
            main._reasoning_effort_env = old_effort

        self.assertNotIn("reasoning", captured)

    def test_role_effort_overrides_env(self):
        role, registry = self._make_provider_and_role(role_effort="high")
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return self._make_fake_response()

        provider = main.ResponsesProvider.__new__(main.ResponsesProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

        old_effort = main._reasoning_effort_env
        main._reasoning_effort_env = "low"
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                provider.generate(role, "gpt-4o", "sender", [])
        finally:
            main._reasoning_effort_env = old_effort

        self.assertEqual(captured.get("reasoning"), {"effort": "high"})


class ThinkingExtractionTests(unittest.TestCase):
    """Tests for _extract_thinking_chat and _extract_thinking_responses."""

    def _make_role(self):
        return SimpleNamespace(
            name="SE",
            max_tokens=100,
            temperature=0.0,
            employee_dict=None,
            allowed_tools=set(),
            runtime_thinking=None,
        )

    # --- _extract_thinking_chat ---

    def test_reasoning_content_field(self):
        msg = SimpleNamespace(content="Green.", reasoning_content="I like blue but green more.")
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "Green.")
        self.assertEqual(thinking, "I like blue but green more.")

    def test_thinking_field(self):
        msg = SimpleNamespace(content="Sure.", thinking="hidden reasoning", reasoning_content=None)
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "Sure.")
        self.assertEqual(thinking, "hidden reasoning")

    def test_content_blocks_thinking_type(self):
        blocks = [
            {"type": "thinking", "thinking": "private thoughts"},
            {"type": "text", "text": "public answer"},
        ]
        msg = SimpleNamespace(content=blocks, reasoning_content=None, thinking=None)
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "public answer")
        self.assertEqual(thinking, "private thoughts")

    def test_inline_think_tags_stripped(self):
        msg = SimpleNamespace(
            content="<think>considered this</think>Final answer.",
            reasoning_content=None,
            thinking=None,
        )
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "Final answer.")
        self.assertEqual(thinking, "considered this")

    def test_inline_thinking_tags_stripped(self):
        msg = SimpleNamespace(
            content="<thinking>deep thoughts</thinking>Result.",
            reasoning_content=None,
            thinking=None,
        )
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "Result.")
        self.assertEqual(thinking, "deep thoughts")

    def test_inline_think_tags_multiple_blocks_stripped(self):
        msg = SimpleNamespace(
            content="A<think>t1</think>B<thinking>t2</thinking>C",
            reasoning_content=None,
            thinking=None,
        )
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "ABC")
        self.assertEqual(thinking, "t1\n\nt2")

    def test_reasoning_content_and_inline_tags_both_captured(self):
        msg = SimpleNamespace(
            content="<think>inline</think>Answer.",
            reasoning_content="field reasoning",
            thinking=None,
        )
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "Answer.")
        self.assertEqual(thinking, "field reasoning\n\ninline")

    def test_no_thinking_returns_none(self):
        msg = SimpleNamespace(content="plain answer", reasoning_content=None, thinking=None)
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "plain answer")
        self.assertIsNone(thinking)

    def test_multiple_think_blocks_all_collected(self):
        msg = SimpleNamespace(
            content="<think>block one</think>middle<think>block two</think>end.",
            reasoning_content=None,
            thinking=None,
        )
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "middleend.")
        self.assertIn("block one", thinking)
        self.assertIn("block two", thinking)

    def test_reasoning_content_plus_inline_tag_stripped(self):
        # Some providers populate both reasoning_content AND embed a tag in content.
        msg = SimpleNamespace(
            content="<think>leaked tag</think>Final answer.",
            reasoning_content="structured reasoning",
            thinking=None,
        )
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "Final answer.")
        self.assertIn("structured reasoning", thinking)
        self.assertIn("leaked tag", thinking)

    # --- _extract_thinking_responses ---

    def test_reasoning_output_item(self):
        part = SimpleNamespace(text="reasoning text")
        item = SimpleNamespace(type="reasoning", content=[part], text=None, summary=[])
        response = SimpleNamespace(output=[item])
        thinking = main._extract_thinking_responses(response)
        self.assertEqual(thinking, "reasoning text")

    def test_reasoning_summary_text_item(self):
        summary_part = SimpleNamespace(type="summary_text", text="brief summary of thinking")
        item = SimpleNamespace(type="reasoning", summary=[summary_part], content=[], text=None)
        response = SimpleNamespace(output=[item])
        thinking = main._extract_thinking_responses(response)
        self.assertEqual(thinking, "brief summary of thinking")

    def test_no_reasoning_output_returns_none(self):
        item = SimpleNamespace(type="message", content=[], text="hi")
        response = SimpleNamespace(output=[item])
        thinking = main._extract_thinking_responses(response)
        self.assertIsNone(thinking)

    # --- provider sets runtime_thinking ---

    def test_chat_provider_sets_runtime_thinking(self):
        role = self._make_role()
        msg = SimpleNamespace(
            content="<think>I deliberated</think>Answer.",
            tool_calls=None,
            reasoning_content=None,
            thinking=None,
            model_dump=lambda: {},
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        registry = SimpleNamespace(
            as_chat_completion_tools=lambda r: [],
            resolve_name=lambda n: n,
        )
        provider = main.ChatCompletionsProvider.__new__(main.ChatCompletionsProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            )
        )
        result = provider.generate(role, "model", "CEO", [])
        self.assertEqual(result, "Answer.")
        self.assertEqual(role.runtime_thinking, "I deliberated")

    def test_responses_provider_sets_runtime_thinking(self):
        role = self._make_role()
        part = SimpleNamespace(text="thought about it")
        reasoning_item = SimpleNamespace(type="reasoning", content=[part], text=None, summary=[])
        text_item = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="Done.")])
        response = SimpleNamespace(id="r1", output=[reasoning_item, text_item], output_text=None)
        registry = SimpleNamespace(as_responses_tools=lambda r: [])
        provider = main.ResponsesProvider.__new__(main.ResponsesProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kw: response)
        )
        result = provider.generate(role, "model", "CEO", [])
        self.assertEqual(result, "Done.")
        self.assertEqual(role.runtime_thinking, "thought about it")

    # --- thinking accumulated across agentic loop iterations ---

    def test_chat_provider_accumulates_thinking_across_tool_iterations(self):
        role = self._make_role()
        role.employee_dict = {}
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="noop", arguments="{}"),
        )
        msg1 = SimpleNamespace(
            content="",
            tool_calls=[tool_call],
            reasoning_content="step 1 thinking",
            thinking=None,
            model_dump=lambda: {"role": "assistant", "content": "", "tool_calls": []},
        )
        msg2 = SimpleNamespace(
            content="Final answer.",
            tool_calls=None,
            reasoning_content="step 2 thinking",
            thinking=None,
            model_dump=lambda: {},
        )
        responses_iter = iter([
            SimpleNamespace(choices=[SimpleNamespace(message=msg1)]),
            SimpleNamespace(choices=[SimpleNamespace(message=msg2)]),
        ])
        registry = SimpleNamespace(
            as_chat_completion_tools=lambda r: [{"type": "function", "function": {"name": "noop"}}],
            execute=lambda name, args, ed, caller=None: "ok",
            resolve_name=lambda n: n,
        )
        provider = main.ChatCompletionsProvider.__new__(main.ChatCompletionsProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: next(responses_iter))
            )
        )
        result = provider.generate(role, "model", "CEO", [])
        self.assertEqual(result, "Final answer.")
        self.assertIn("step 1 thinking", role.runtime_thinking)
        self.assertIn("step 2 thinking", role.runtime_thinking)

    def test_responses_provider_accumulates_thinking_across_tool_iterations(self):
        role = self._make_role()
        role.employee_dict = {}
        # First response: reasoning item + function call
        func_call = SimpleNamespace(type="function_call", call_id="c1", name="noop", arguments="{}")
        r1_thinking = SimpleNamespace(type="reasoning", content=[], text="step 1 thinking", summary=[])
        response1 = SimpleNamespace(id="r1", output=[r1_thinking, func_call], output_text=None)
        # Second response: reasoning item + text message, no function calls
        r2_thinking = SimpleNamespace(type="reasoning", content=[], text="step 2 thinking", summary=[])
        text_item = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="Done.")])
        response2 = SimpleNamespace(id="r2", output=[r2_thinking, text_item], output_text=None)

        call_count = [0]
        def mock_create(**kw):
            call_count[0] += 1
            return response1 if call_count[0] == 1 else response2

        registry = SimpleNamespace(
            as_responses_tools=lambda r: [],
            execute=lambda name, args, ed, caller=None: "ok",
            resolve_name=lambda n: n,
        )
        provider = main.ResponsesProvider.__new__(main.ResponsesProvider)
        provider.registry = registry
        provider.client = SimpleNamespace(
            responses=SimpleNamespace(create=mock_create)
        )
        result = provider.generate(role, "model", "CEO", [])
        self.assertEqual(result, "Done.")
        self.assertIn("step 1 thinking", role.runtime_thinking)
        self.assertIn("step 2 thinking", role.runtime_thinking)

    # --- P2: thinking block using text field instead of thinking field ---

    def test_content_blocks_thinking_type_with_text_field(self):
        blocks = [
            {"type": "thinking", "text": "private via text field", "thinking": None},
            {"type": "text", "text": "public answer"},
        ]
        msg = SimpleNamespace(content=blocks, reasoning_content=None, thinking=None)
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "public answer")
        self.assertEqual(thinking, "private via text field")

    # --- MEDIUM: provider-specific attrs checked even when content is a list ---

    def test_content_blocks_list_with_reasoning_content_attr(self):
        blocks = [{"type": "text", "text": "visible"}]
        msg = SimpleNamespace(content=blocks, reasoning_content="structured reasoning", thinking=None)
        text, thinking = main._extract_thinking_chat(msg)
        self.assertEqual(text, "visible")
        self.assertEqual(thinking, "structured reasoning")


class OrgChatReadToolTests(unittest.TestCase):
    """Tests for the org_chat_read function."""

    def test_returns_empty_when_no_org_workspace(self):
        old = main._org_workspace
        main._org_workspace = None
        try:
            result = json.loads(main.org_chat_read({}))
        finally:
            main._org_workspace = old
        self.assertEqual(result["lines"], [])
        self.assertIn("not available", result["note"])

    def test_returns_empty_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = main.AgentWorkspaceStore(tmp)
            old = main._org_workspace
            main._org_workspace = store
            try:
                result = json.loads(main.org_chat_read({}))
            finally:
                main._org_workspace = old
            self.assertEqual(result["lines"], [])

    def test_returns_parsed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = main.AgentWorkspaceStore(tmp)
            entry = {"turn": 1, "sender": "A", "receiver": "B", "prompt": "hi", "response": "hey"}
            store.write("org", "org_chat.jsonl", json.dumps(entry) + "\n", append=False)
            old = main._org_workspace
            main._org_workspace = store
            try:
                result = json.loads(main.org_chat_read({}))
            finally:
                main._org_workspace = old
            self.assertEqual(len(result["lines"]), 1)
            self.assertEqual(result["lines"][0]["sender"], "A")
            self.assertEqual(result["total"], 1)

    def test_limit_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = main.AgentWorkspaceStore(tmp)
            for i in range(5):
                entry = {"turn": i, "sender": "A", "receiver": "B", "prompt": f"p{i}", "response": ""}
                store.write("org", "org_chat.jsonl", json.dumps(entry) + "\n", append=True)
            old = main._org_workspace
            main._org_workspace = store
            try:
                result = json.loads(main.org_chat_read({}, limit=2))
            finally:
                main._org_workspace = old
            self.assertEqual(len(result["lines"]), 2)
            self.assertEqual(result["total"], 5)


class ClockStateTests(unittest.TestCase):
    """Tests for on/off-the-clock mode affecting prompts, org context, and identity."""

    def setUp(self):
        self.original_clock = main.clock_state
        self.addCleanup(self._restore)

    def _restore(self):
        main.clock_state = self.original_clock

    def test_on_clock_guidance_used_when_on(self):
        main.clock_state = "on"
        self.assertIn("work-focused", main._ON_CLOCK_GUIDANCE)

    def test_off_clock_guidance_used_when_off(self):
        main.clock_state = "off"
        self.assertIn("off the clock", main._OFF_CLOCK_GUIDANCE)

    def test_session_stores_clock_state(self):
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        session = main.Session(participants={"A": a}, clock_state="off")
        self.assertEqual(session.clock_state, "off")

    def test_session_clock_state_in_event(self):
        events = []
        stream = main.RuntimeEventStream()
        stream.subscribe(lambda e: events.append(e))
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        main.Session(participants={"A": a}, event_stream=stream, clock_state="off")
        created = next(e for e in events if e.event_type == "session.created")
        self.assertEqual(created.payload["clock_state"], "off")

    def test_off_clock_suppresses_org_chat_injection(self):
        received_prompts = []

        class CapturingRole:
            name = "B"
            lifecycle_state = "active"
            conversation_history = {}
            template = ""
            max_tokens = 100
            temperature = 0.0
            runtime_tool_results = []
            runtime_event_stream = None
            runtime_session_id = None
            runtime_role_name = None
            allowed_tools = set()

            def interact(self, sender=None, prompt=None):
                received_prompts.append(prompt or "")
                return "response"

            def update_group_conversations(self, m):
                pass

        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "initial",
                            update_group_conversations=lambda m: None)
        b = CapturingRole()
        old_lines = main.org_chat_context_lines
        main.org_chat_context_lines = 5
        try:
            session = main.Session(participants={"A": a, "B": b}, clock_state="off")
            session.transcript.append(main.TranscriptEntry(1, "A", "B", "seed", "response"))
            session.turns_completed = 1
            session.last_receiver = a
            session.step("hello")
        finally:
            main.org_chat_context_lines = old_lines

        self.assertFalse(any("Recent org chat" in p for p in received_prompts))

    def test_identity_primary_secondary_swapped_by_clock(self):
        store_temp = tempfile.TemporaryDirectory()
        store = SQLiteMemoryStore(Path(store_temp.name) / "id.sqlite3")
        store.upsert_agent("alice", "Role", "Alice")
        store.seed_memory_digest("identity", "personal identity content",
                                  agent_id="alice", relationship_type="personal")
        store.seed_memory_digest("identity", "work identity content",
                                  agent_id="alice", relationship_type="work")
        old_store = main.memory_store
        main.memory_store = store
        try:
            main.clock_state = "on"
            primary_on, secondary_on = main._get_identity_digests("alice")
            main.clock_state = "off"
            primary_off, secondary_off = main._get_identity_digests("alice")
        finally:
            main.memory_store = old_store
            main.clock_state = self.original_clock
            store.close()
            store_temp.cleanup()

        self.assertIn("work", primary_on)
        self.assertIn("personal", secondary_on)
        self.assertIn("personal", primary_off)
        self.assertIn("work", secondary_off)


class MemorySearchParkingTests(unittest.TestCase):
    """Tests for off-clock parking reminder when org_chat results surface."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "p.sqlite3")
        self.addCleanup(self.store.close)
        self.original_store = main.memory_store
        self.original_clock = main.clock_state
        main.memory_store = self.store
        self.addCleanup(self._restore)

    def _restore(self):
        main.memory_store = self.original_store
        main.clock_state = self.original_clock

    def _seed_org_message(self, agent_id="bob"):
        self.store.upsert_agent(agent_id, "Role", agent_id)
        self.store.create_session("s1")
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL
        ch = self.store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
        self.store.append_message("s1", agent_id, agent_id, "project deadline is Friday",
                                   channel_id=ch)

    def test_off_clock_org_result_prepends_parking_note(self):
        self._seed_org_message()
        main.clock_state = "off"
        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="bob", allowed_tools={"memory.*"})
        try:
            result = main.memory_search({}, "bob", "project deadline")
        finally:
            main.active_tool_caller = old_caller
        self.assertIn("work.todo", result)
        self.assertIn("System note", result)

    def test_on_clock_no_parking_note(self):
        self._seed_org_message()
        main.clock_state = "on"
        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="bob", allowed_tools={"memory.*"})
        try:
            result = main.memory_search({}, "bob", "project deadline")
        finally:
            main.active_tool_caller = old_caller
        self.assertNotIn("System note", result)

    def test_off_clock_personal_result_no_parking_note(self):
        self.store.upsert_agent("carol", "Role", "carol")
        self.store.create_session("s2")
        self.store.append_message("s2", "carol", "carol", "family picnic plans")
        main.clock_state = "off"
        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="carol", allowed_tools={"memory.*"})
        try:
            result = main.memory_search({}, "carol", "picnic")
        finally:
            main.active_tool_caller = old_caller
        self.assertNotIn("System note", result)


class WorkTodoToolTests(unittest.TestCase):
    """Tests for work_todo_add function."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "t.sqlite3")
        self.addCleanup(self.store.close)
        self.original_store = main.memory_store
        self.original_caller = main.active_tool_caller
        main.memory_store = self.store
        self.addCleanup(self._restore)

    def _restore(self):
        main.memory_store = self.original_store
        main.active_tool_caller = self.original_caller

    def test_creates_todo_record(self):
        self.store.upsert_agent("dave", "Role", "dave")
        main.active_tool_caller = SimpleNamespace(name="dave")
        result = json.loads(main.work_todo_add({}, title="Follow up on Q3 budget"))
        self.assertEqual(result["title"], "Follow up on Q3 budget")
        self.assertEqual(result["status"], "open")
        rows = self.store.list_todos(agent_id="dave")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Follow up on Q3 budget")

    def test_no_memory_store_returns_error(self):
        main.memory_store = None
        result = json.loads(main.work_todo_add({}, title="something"))
        self.assertIn("error", result)

    def test_no_caller_returns_error(self):
        main.active_tool_caller = SimpleNamespace()  # no .name
        result = json.loads(main.work_todo_add({}, title="something"))
        self.assertIn("error", result)

    def test_content_stored(self):
        self.store.upsert_agent("eve", "Role", "eve")
        main.active_tool_caller = SimpleNamespace(name="eve")
        main.work_todo_add({}, title="Research topic", content="detailed notes here")
        rows = self.store.list_todos(agent_id="eve")
        self.assertEqual(rows[0]["content"], "detailed notes here")


class ChannelScopedMemoryTests(unittest.TestCase):
    """Tests for channel-scoped memory tagging and filtering (issue #69)."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "ch.sqlite3")
        self.addCleanup(self.store.close)
        self.original_store = main.memory_store
        self.original_clock = main.clock_state
        main.memory_store = self.store
        main.clock_state = "on"
        self.addCleanup(self._restore)
        self.store.create_session("s1")
        self.store.upsert_agent("A", "Role", "A")
        self.store.upsert_agent("B", "Role", "B")

    def _restore(self):
        main.memory_store = self.original_store
        main.clock_state = self.original_clock

    def _make_session(self, participants=None):
        if participants is None:
            participants = {
                "A": SimpleNamespace(
                    name="A", lifecycle_state="active",
                    interact=lambda *a, **kw: "resp",
                    update_group_conversations=lambda m: None,
                ),
                "B": SimpleNamespace(
                    name="B", lifecycle_state="active",
                    interact=lambda *a, **kw: "resp",
                    update_group_conversations=lambda m: None,
                ),
            }
        return main.Session(participants=participants)

    def test_record_thought_tagged_with_agent_thought_channel(self):
        from robits.memory.sqlite import CHANNEL_AGENT_THOUGHT
        session = self._make_session()
        session.record_thought("A", "private reflection")
        rows = self.store.connection.execute(
            "SELECT t.channel_id, c.channel_type FROM thoughts t"
            " LEFT JOIN channels c ON c.channel_id = t.channel_id"
            " WHERE t.agent_id = 'A'"
        ).fetchall()
        self.assertTrue(rows, "No thought row found")
        self.assertEqual(rows[0]["channel_type"], CHANNEL_AGENT_THOUGHT)

    def test_thought_channel_is_stable_across_calls(self):
        session = self._make_session()
        session.record_thought("A", "first")
        session.record_thought("A", "second")
        channel_ids = self.store.connection.execute(
            "SELECT DISTINCT channel_id FROM thoughts WHERE agent_id='A'"
        ).fetchall()
        self.assertEqual(len(channel_ids), 1, "Thoughts should share a single agent_thought channel")

    def test_directed_message_tagged_with_agent_dm_channel(self):
        from robits.memory.sqlite import CHANNEL_AGENT_DM
        session = self._make_session()
        session.record_turn("A", "B", "hey", "sup", directed=True)
        rows = self.store.connection.execute(
            "SELECT m.channel_id, c.channel_type, m.visibility FROM messages m"
            " LEFT JOIN channels c ON c.channel_id = m.channel_id"
            " WHERE m.session_id=?", (session.run_id,)
        ).fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(r["channel_type"] == CHANNEL_AGENT_DM for r in rows))
        self.assertTrue(all(r["visibility"] == "private" for r in rows))

    def test_directed_message_excluded_from_org_chat_context(self):
        from robits.core.context import format_org_chat_context
        session = self._make_session()
        session.record_turn("A", "B", "public update", "ack", directed=False)
        session.record_turn("A", "B", "private dm", "private reply", directed=True)
        ctx = format_org_chat_context(session.transcript, 10)
        self.assertIn("public update", ctx)
        self.assertNotIn("private dm", ctx)

    def test_undirected_message_tagged_with_org_chat_channel(self):
        from robits.memory.sqlite import CHANNEL_ORG_CHAT
        session = self._make_session()
        session.record_turn("A", "B", "org update", "acknowledged", directed=False)
        rows = self.store.connection.execute(
            "SELECT m.channel_id, c.channel_type FROM messages m"
            " LEFT JOIN channels c ON c.channel_id = m.channel_id"
            " WHERE m.session_id=?", (session.run_id,)
        ).fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(r["channel_type"] == CHANNEL_ORG_CHAT for r in rows))

    def test_dm_channel_is_stable_for_pair(self):
        session = self._make_session()
        session.record_turn("A", "B", "msg1", "reply1", directed=True)
        session.record_turn("A", "B", "msg2", "reply2", directed=True)
        channel_ids = self.store.connection.execute(
            "SELECT DISTINCT channel_id FROM messages WHERE session_id=?",
            (session.run_id,)
        ).fetchall()
        self.assertEqual(len(channel_ids), 1, "All directed messages share one DM channel")

    def test_memory_search_filters_by_channel_type(self):
        from robits.memory.sqlite import CHANNEL_AGENT_THOUGHT, CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL
        self.store.upsert_agent("alice", "Role", "alice")
        self.store.create_session("s2")
        org_ch = self.store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
        thought_ch = self.store.get_or_create_channel(
            CHANNEL_AGENT_THOUGHT, participants=["alice"], visibility="private", social_distance=0.0
        )
        self.store.append_message("s2", "alice", "alice", "standup meeting notes xkw1", channel_id=org_ch)
        self.store.append_thought("alice", "personal reflection xkw1", session_id="s2", channel_id=thought_ch)

        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="alice", capabilities=set())
        try:
            all_results = json.loads(main.memory_search({}, "alice", "xkw1"))
            thought_results = json.loads(main.memory_search({}, "alice", "xkw1", channel_type=CHANNEL_AGENT_THOUGHT))
            org_results = json.loads(main.memory_search({}, "alice", "xkw1", channel_type=CHANNEL_ORG_CHAT))
        finally:
            main.active_tool_caller = old_caller

        self.assertGreaterEqual(len(all_results), 2, "Unfiltered search should return both")
        self.assertTrue(all(r["conversation_type"] == CHANNEL_AGENT_THOUGHT for r in thought_results))
        self.assertTrue(all(r["conversation_type"] == CHANNEL_ORG_CHAT for r in org_results))

    def test_off_clock_work_peer_result_prepends_parking_note(self):
        """work_peer channel content should also trigger the parking note when off-clock."""
        from robits.memory.sqlite import CHANNEL_WORK_PEER, SOCIAL_PROFESSIONAL
        self.store.upsert_agent("dave", "Role", "dave")
        self.store.create_session("s3")
        wp_ch = self.store.get_or_create_channel(
            CHANNEL_WORK_PEER, participants=["dave"], social_distance=SOCIAL_PROFESSIONAL
        )
        self.store.append_message("s3", "dave", "dave", "project timeline xkw2", channel_id=wp_ch)
        original_clock = main.clock_state
        main.clock_state = "off"
        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="dave", capabilities=set())
        try:
            result = main.memory_search({}, "dave", "xkw2")
        finally:
            main.active_tool_caller = old_caller
            main.clock_state = original_clock
        self.assertIn("System note", result)

    def test_personal_experience_scenario(self):
        """Agents searching for personal/work experience get channel-scoped results."""
        from robits.memory.sqlite import CHANNEL_AGENT_THOUGHT, CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL
        self.store.upsert_agent("frank", "Role", "frank")
        self.store.create_session("s4")
        org_ch = self.store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
        thought_ch = self.store.get_or_create_channel(
            CHANNEL_AGENT_THOUGHT, participants=["frank"], visibility="private", social_distance=0.0
        )
        self.store.append_message("s4", "frank", "frank", "I used Python at work for data pipelines xkw3", channel_id=org_ch)
        self.store.append_thought("frank", "Personally I love cooking Italian food xkw3", session_id="s4", channel_id=thought_ch)

        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="frank", capabilities=set())
        try:
            personal = json.loads(main.memory_search({}, "frank", "xkw3", channel_type=CHANNEL_AGENT_THOUGHT))
            work = json.loads(main.memory_search({}, "frank", "xkw3", channel_type=CHANNEL_ORG_CHAT))
        finally:
            main.active_tool_caller = old_caller

        self.assertEqual(len(personal), 1)
        self.assertIn("cooking", personal[0]["content"])
        self.assertEqual(len(work), 1)
        self.assertIn("Python", work[0]["content"])


class PersonaRedesignTests(unittest.TestCase):
    """Tests for #93 persona redesign: username/full_name schema and @mention detection."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "persona93.sqlite3")
        self.addCleanup(self.store.close)

    def test_load_personas_new_schema(self):
        import tempfile as _tf, yaml as _yaml
        from robits.core.persona import load_personas
        content = [
            {"username": "alex_chen", "full_name": "Alex Chen", "role": "SE",
             "memories": [{"kind": "thought", "content": "I like Python."}]},
            {"username": "jamie", "full_name": "Jamie Okonkwo", "role": "SE",
             "memories": [{"kind": "thought", "content": "Distributed systems fan."}]},
        ]
        with _tf.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(content, f)
            path = f.name
        try:
            result = load_personas(path)
            self.assertIn("alex_chen", result)
            self.assertIn("jamie", result)
            self.assertEqual(result["alex_chen"]["role"], "SE")
            self.assertEqual(result["alex_chen"]["full_name"], "Alex Chen")
            self.assertEqual(len(result["alex_chen"]["entries"]), 1)
        finally:
            import os; os.unlink(path)

    def test_load_personas_legacy_schema_still_works(self):
        import tempfile as _tf, yaml as _yaml
        from robits.core.persona import load_personas
        content = [
            {"agent": "SE", "memories": [{"kind": "thought", "content": "Old schema."}]},
        ]
        with _tf.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(content, f)
            path = f.name
        try:
            result = load_personas(path)
            self.assertIn("SE", result)
            self.assertEqual(result["SE"]["role"], "SE")
            self.assertEqual(len(result["SE"]["entries"]), 1)
        finally:
            import os; os.unlink(path)

    def test_multiple_personas_same_role(self):
        import tempfile as _tf, yaml as _yaml
        from robits.core.persona import load_personas
        content = [
            {"username": "eng1", "full_name": "Alice Smith", "role": "SE",
             "memories": [{"kind": "thought", "content": "Alice's memory."}]},
            {"username": "eng2", "full_name": "Bob Jones", "role": "SE",
             "memories": [{"kind": "thought", "content": "Bob's memory."}]},
        ]
        with _tf.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(content, f)
            path = f.name
        try:
            result = load_personas(path)
            self.assertIn("eng1", result)
            self.assertIn("eng2", result)
            self.assertEqual(result["eng1"]["full_name"], "Alice Smith")
            self.assertEqual(result["eng2"]["full_name"], "Bob Jones")
        finally:
            import os; os.unlink(path)

    def test_upsert_agent_stores_identity_fields(self):
        self.store.upsert_agent(
            "alex_chen", "SE", display_name="Alex Chen",
            username="alex_chen", first_name="Alex", last_name="Chen", full_name="Alex Chen"
        )
        rows = self.store._rows("SELECT * FROM agents WHERE agent_id = 'alex_chen'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["username"], "alex_chen")
        self.assertEqual(rows[0]["first_name"], "Alex")
        self.assertEqual(rows[0]["last_name"], "Chen")
        self.assertEqual(rows[0]["full_name"], "Alex Chen")

    def test_get_agent_name_tokens_returns_all_forms(self):
        self.store.upsert_agent(
            "alex_chen", "SE", username="alex_chen",
            first_name="Alex", last_name="Chen", full_name="Alex Chen"
        )
        tokens_map = self.store.get_agent_name_tokens()
        self.assertIn("alex_chen", tokens_map)
        tokens = tokens_map["alex_chen"]
        self.assertIn("alex_chen", tokens)
        self.assertIn("alex", tokens)
        self.assertIn("chen", tokens)
        self.assertIn("alex chen", tokens)

    def test_build_employee_dict_uses_persona_usernames(self):
        from robits.core.roles import build_employee_dict
        persona_map = {
            "alex_chen": {"role": "SE", "full_name": "Alex Chen", "entries": []},
        }
        d = build_employee_dict(persona_map)
        self.assertIn("alex_chen", d)
        self.assertNotIn("SE", d)
        self.assertEqual(d["alex_chen"].name, "alex_chen")

    def test_build_employee_dict_multiple_same_role(self):
        from robits.core.roles import build_employee_dict
        persona_map = {
            "eng1": {"role": "SE", "full_name": "Alice Smith", "entries": []},
            "eng2": {"role": "SE", "full_name": "Bob Jones", "entries": []},
        }
        d = build_employee_dict(persona_map)
        self.assertIn("eng1", d)
        self.assertIn("eng2", d)
        self.assertNotIn("SE", d)

    def test_se_has_shell_capability(self):
        """SoftwareEngineer must carry the 'shell' capability so sub-agents can call builtin.shell_run."""
        from robits.core.roles import SoftwareEngineer
        se = SoftwareEngineer({})
        self.assertIn("shell", se.capabilities)

    def test_se_has_shell_run_in_allowed_tools(self):
        """SoftwareEngineer.allowed_tools must include builtin.shell_run explicitly."""
        from robits.core.roles import SoftwareEngineer
        se = SoftwareEngineer({})
        self.assertIn("builtin.shell_run", se.allowed_tools)

    def test_mention_detection_at_username(self):
        from robits.core.session import Session
        self.store.upsert_agent("alex_chen", "SE", username="alex_chen",
                                first_name="Alex", last_name="Chen", full_name="Alex Chen")
        a = SimpleNamespace(name="CEO", lifecycle_state="active",
                            interact=lambda *a, **kw: "",
                            update_group_conversations=lambda m: None)
        b_role = SimpleNamespace(name="alex_chen", lifecycle_state="active",
                                 interact=lambda *a, **kw: "",
                                 update_group_conversations=lambda m: None)
        session = Session(participants={"CEO": a, "alex_chen": b_role})
        old_store = main.memory_store
        main.memory_store = self.store
        try:
            mentioned = session._detect_mentions("Hey @alex_chen can you review this?")
            self.assertIn("alex_chen", mentioned)
        finally:
            main.memory_store = old_store

    def test_mention_detection_by_first_name(self):
        from robits.core.session import Session
        self.store.upsert_agent("alex_chen", "SE", username="alex_chen",
                                first_name="Alex", last_name="Chen", full_name="Alex Chen")
        a = SimpleNamespace(name="CEO", lifecycle_state="active",
                            interact=lambda *a, **kw: "",
                            update_group_conversations=lambda m: None)
        b_role = SimpleNamespace(name="alex_chen", lifecycle_state="active",
                                 interact=lambda *a, **kw: "",
                                 update_group_conversations=lambda m: None)
        session = Session(participants={"CEO": a, "alex_chen": b_role})
        old_store = main.memory_store
        main.memory_store = self.store
        try:
            mentioned = session._detect_mentions("Alex, what do you think about this?")
            self.assertIn("alex_chen", mentioned)
        finally:
            main.memory_store = old_store

    def test_no_mention_when_no_match(self):
        from robits.core.session import Session
        self.store.upsert_agent("alex_chen", "SE", username="alex_chen",
                                first_name="Alex", last_name="Chen", full_name="Alex Chen")
        a = SimpleNamespace(name="CEO", lifecycle_state="active",
                            interact=lambda *a, **kw: "",
                            update_group_conversations=lambda m: None)
        b_role = SimpleNamespace(name="alex_chen", lifecycle_state="active",
                                 interact=lambda *a, **kw: "",
                                 update_group_conversations=lambda m: None)
        session = Session(participants={"CEO": a, "alex_chen": b_role})
        old_store = main.memory_store
        main.memory_store = self.store
        try:
            mentioned = session._detect_mentions("What does everyone think?")
            self.assertNotIn("alex_chen", mentioned)
        finally:
            main.memory_store = old_store

    def test_mention_no_false_positive_common_word(self):
        from robits.core.session import Session
        self.store.upsert_agent("will_smith", "SE", username="will_smith",
                                first_name="Will", last_name="Smith", full_name="Will Smith")
        a = SimpleNamespace(name="CEO", lifecycle_state="active",
                            interact=lambda *a, **kw: "",
                            update_group_conversations=lambda m: None)
        b_role = SimpleNamespace(name="will_smith", lifecycle_state="active",
                                 interact=lambda *a, **kw: "",
                                 update_group_conversations=lambda m: None)
        session = Session(participants={"CEO": a, "will_smith": b_role})
        old_store = main.memory_store
        main.memory_store = self.store
        try:
            # lowercase "will" in a sentence should NOT trigger a mention
            mentioned = session._detect_mentions("I will look into it, no worries.")
            self.assertNotIn("will_smith", mentioned)
            # Capitalised "Will" (addressing the agent) SHOULD trigger
            mentioned2 = session._detect_mentions("Will, can you review the PR?")
            self.assertIn("will_smith", mentioned2)
        finally:
            main.memory_store = old_store

    def test_build_employee_dict_ceo_persona(self):
        from robits.core.roles import build_employee_dict, Human
        persona_map = {
            "CEO": {"role": "CEO", "full_name": "CEO", "entries": []},
        }
        d = build_employee_dict(persona_map)
        self.assertIn("CEO", d)
        self.assertIsInstance(d["CEO"], Human)


class PersonaSeedingTests(unittest.TestCase):
    """Tests for persona seeding into memory on first session."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "persona.sqlite3")
        self.addCleanup(self.store.close)

    def test_seed_persona_writes_thought(self):
        self.store.upsert_agent("alice", "Role", "alice")
        from robits.core.persona import seed_persona
        written = seed_persona(self.store, "alice", [
            {"kind": "thought", "content": "I enjoy hiking.", "visibility": "private"},
        ])
        self.assertEqual(written, 1)
        rows = self.store._rows("SELECT * FROM thoughts WHERE agent_id = 'alice'")
        self.assertEqual(len(rows), 1)
        self.assertIn("hiking", rows[0]["content"])

    def test_seed_persona_writes_digest(self):
        self.store.upsert_agent("bob", "Role", "bob")
        from robits.core.persona import seed_persona
        written = seed_persona(self.store, "bob", [
            {"kind": "digest", "digest_type": "identity", "content": "Bob is a backend engineer."},
        ])
        self.assertEqual(written, 1)
        digests = self.store.list_memory_digests(agent_id="bob", digest_type="identity")
        self.assertEqual(len(digests), 1)
        self.assertIn("backend engineer", digests[0]["content"])

    def test_seed_persona_is_idempotent(self):
        self.store.upsert_agent("carol", "Role", "carol")
        from robits.core.persona import seed_persona
        entries = [{"kind": "thought", "content": "First seed.", "visibility": "private"}]
        first = seed_persona(self.store, "carol", entries)
        second = seed_persona(self.store, "carol", entries)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        rows = self.store._rows("SELECT * FROM thoughts WHERE agent_id = 'carol'")
        self.assertEqual(len(rows), 1)

    def test_load_personas_missing_file_returns_empty(self):
        from robits.core.persona import load_personas
        result = load_personas("/nonexistent/path/personas.yaml")
        self.assertEqual(result, {})

    def test_seed_persona_writes_memory_entry(self):
        self.store.upsert_agent("dave", "Role", "dave")
        from robits.core.persona import seed_persona
        written = seed_persona(self.store, "dave", [
            {"kind": "entry", "digest_type": "identity", "content": "Dave values honesty."},
        ])
        self.assertEqual(written, 1)
        rows = self.store._rows("SELECT * FROM memory_entries WHERE agent_id = 'dave'")
        self.assertEqual(len(rows), 1)
        self.assertIn("honesty", rows[0]["content"])


class BreakScheduleTests(unittest.TestCase):
    """Tests for scheduled break windows and org.schedule tool."""

    def test_parse_break_schedule_single_window(self):
        from robits.core.config import _parse_break_schedule
        result = _parse_break_schedule("12:00-13:00")
        self.assertEqual(result, [("12:00", "13:00")])

    def test_parse_break_schedule_multiple_windows(self):
        from robits.core.config import _parse_break_schedule
        result = _parse_break_schedule("12:00-13:00,15:00-15:30")
        self.assertEqual(result, [("12:00", "13:00"), ("15:00", "15:30")])

    def test_parse_break_schedule_empty(self):
        from robits.core.config import _parse_break_schedule
        self.assertEqual(_parse_break_schedule(""), [])
        self.assertEqual(_parse_break_schedule(None), [])

    def test_parse_break_schedule_single_digit_hour(self):
        from robits.core.config import _parse_break_schedule
        result = _parse_break_schedule("9:00-9:30")
        self.assertEqual(result, [("09:00", "09:30")])

    def test_parse_break_schedule_rejects_midnight_wrap(self):
        from robits.core.config import _parse_break_schedule
        result = _parse_break_schedule("22:00-02:00")
        self.assertEqual(result, [])

    def test_parse_break_schedule_rejects_malformed(self):
        from robits.core.config import _parse_break_schedule
        result = _parse_break_schedule("notawindow,12:00-13:00")
        self.assertEqual(result, [("12:00", "13:00")])

    def test_effective_clock_state_off_not_promoted_to_break(self):
        from robits.core.session import Session
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        session = Session(participants={"A": a}, clock_state="off")
        old_schedule = main.break_schedule
        main.break_schedule = [("00:00", "23:59")]
        try:
            self.assertEqual(session._effective_clock_state(), "off")
        finally:
            main.break_schedule = old_schedule

    def test_effective_clock_state_within_window(self):
        from robits.core.session import Session
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        session = Session(participants={"A": a}, clock_state="on")
        session.clock_state = "on"
        old_schedule = main.break_schedule
        main.break_schedule = [("00:00", "23:59")]
        try:
            self.assertEqual(session._effective_clock_state(), "break")
        finally:
            main.break_schedule = old_schedule

    def test_effective_clock_state_outside_window(self):
        from robits.core.session import Session
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        session = Session(participants={"A": a}, clock_state="on")
        old_schedule = main.break_schedule
        main.break_schedule = [("99:00", "99:30")]
        try:
            self.assertEqual(session._effective_clock_state(), "on")
        finally:
            main.break_schedule = old_schedule

    def test_effective_clock_state_no_schedule(self):
        from robits.core.session import Session
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        session = Session(participants={"A": a}, clock_state="off")
        old_schedule = main.break_schedule
        main.break_schedule = []
        try:
            self.assertEqual(session._effective_clock_state(), "off")
        finally:
            main.break_schedule = old_schedule

    def test_break_schedule_in_agent_context(self):
        from robits.core.context import agent_runtime_context
        old_schedule = main.break_schedule
        main.break_schedule = [("09:00", "09:30")]
        try:
            ctx = agent_runtime_context()
            self.assertIn("break_schedule", ctx)
            self.assertEqual(ctx["break_schedule"], [("09:00", "09:30")])
        finally:
            main.break_schedule = old_schedule


class BreakClockStateTests(unittest.TestCase):
    """Tests for the 'break' clock state — transitional between on and off."""

    def setUp(self):
        self.original_clock = main.clock_state
        self.addCleanup(self._restore)

    def _restore(self):
        main.clock_state = self.original_clock

    def test_break_is_valid_session_clock_state(self):
        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "hi",
                            update_group_conversations=lambda m: None)
        session = main.Session(participants={"A": a}, clock_state="break")
        self.assertEqual(session.clock_state, "break")

    def test_break_suppresses_org_chat_context(self):
        received_prompts = []

        class CapturingRole:
            name = "B"
            lifecycle_state = "active"
            conversation_history = {}
            template = ""
            max_tokens = 100
            temperature = 0.7
            base_temperature = 0.7
            runtime_tool_results = []
            runtime_event_stream = None
            runtime_session_id = None
            runtime_role_name = None
            runtime_clock_state = None
            allowed_tools = set()

            def interact(self, sender=None, prompt=None):
                received_prompts.append(prompt or "")
                return "response"

            def update_group_conversations(self, m):
                pass

        a = SimpleNamespace(name="A", lifecycle_state="active",
                            interact=lambda *a, **kw: "initial",
                            update_group_conversations=lambda m: None)
        b = CapturingRole()
        old_lines = main.org_chat_context_lines
        main.org_chat_context_lines = 5
        try:
            session = main.Session(participants={"A": a, "B": b}, clock_state="break")
            session.transcript.append(main.TranscriptEntry(1, "A", "B", "seed", "response"))
            session.turns_completed = 1
            session.last_receiver = a
            session.step("hello")
        finally:
            main.org_chat_context_lines = old_lines

        self.assertFalse(any("Recent org chat" in p for p in received_prompts))

    def test_break_temperature_modulation(self):
        from robits.core.session import _modulate_temperature
        role = SimpleNamespace(temperature=0.7, base_temperature=0.7)
        _modulate_temperature(role, "on")
        on_temp = role.temperature
        role.temperature = 0.7
        _modulate_temperature(role, "break")
        break_temp = role.temperature
        role.temperature = 0.7
        _modulate_temperature(role, "off")
        off_temp = role.temperature
        self.assertLess(on_temp, break_temp)
        self.assertLess(break_temp, off_temp)

    def test_break_parking_note_is_softer(self):
        store_temp = tempfile.TemporaryDirectory()
        store = SQLiteMemoryStore(Path(store_temp.name) / "bp.sqlite3")
        store.upsert_agent("eve", "Role", "eve")
        store.create_session("s_break")
        from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL
        ch = store.get_or_create_channel(CHANNEL_ORG_CHAT, social_distance=SOCIAL_PROFESSIONAL)
        store.append_message("s_break", "eve", "eve", "sprint review xkw_break", channel_id=ch)
        original_store = main.memory_store
        original_clock = main.clock_state
        main.memory_store = store
        main.clock_state = "break"
        old_caller = main.active_tool_caller
        main.active_tool_caller = SimpleNamespace(name="eve", capabilities=set())
        try:
            result = main.memory_search({}, "eve", "xkw_break")
        finally:
            main.active_tool_caller = old_caller
            main.memory_store = original_store
            main.clock_state = original_clock
            store.close()
            store_temp.cleanup()
        self.assertIn("System note", result)
        self.assertIn("break", result)

    def test_top_p_modulation_follows_clock_state(self):
        from robits.core.session import _modulate_temperature
        role = SimpleNamespace(temperature=0.7, base_temperature=0.7, top_p=0.9, base_top_p=0.9)
        _modulate_temperature(role, "on")
        on_top_p = role.top_p
        role.top_p = 0.9
        _modulate_temperature(role, "break")
        break_top_p = role.top_p
        role.top_p = 0.9
        _modulate_temperature(role, "off")
        off_top_p = role.top_p
        self.assertLess(on_top_p, break_top_p)
        self.assertLess(break_top_p, off_top_p)

    def test_top_p_clamped_to_valid_range(self):
        from robits.core.session import _modulate_temperature
        role = SimpleNamespace(temperature=0.7, base_temperature=0.7, top_p=1.0, base_top_p=1.0)
        _modulate_temperature(role, "off")
        self.assertLessEqual(role.top_p, 1.0)
        role2 = SimpleNamespace(temperature=0.7, base_temperature=0.7, top_p=0.1, base_top_p=0.1)
        _modulate_temperature(role2, "on")
        self.assertGreaterEqual(role2.top_p, 0.1)

    def test_role_base_top_p_initialised(self):
        from robits.core.roles import Role
        role = Role("test_role", "template", {})
        self.assertTrue(hasattr(role, "base_top_p"))
        self.assertEqual(role.base_top_p, role.top_p)

    def test_providers_pass_top_p(self):
        from robits.core.providers import ChatCompletionsProvider
        calls = []

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls.append(kwargs)
                        msg = SimpleNamespace(
                            tool_calls=None,
                            content="done",
                            model_dump=lambda: {"role": "assistant", "content": "done"},
                        )
                        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        role = SimpleNamespace(
            name="tester",
            runtime_role_name="tester",
            max_tokens=100,
            temperature=0.5,
            top_p=0.75,
            employee_dict={},
            allowed_tools=set(),
        )

        class FakeRegistry:
            def as_chat_completion_tools(self, role):
                return []
            def execute(self, *a, **kw):
                return ""
            def resolve_name(self, n):
                return n

        provider = ChatCompletionsProvider(FakeClient(), registry=FakeRegistry())
        provider.generate(role, "model-x", "sender", [{"role": "user", "content": "hi"}])
        self.assertTrue(calls)
        self.assertIn("top_p", calls[0])
        self.assertAlmostEqual(calls[0]["top_p"], 0.75)


class EmbeddingSearchTests(unittest.TestCase):
    """Tests for embedding-based semantic search in SQLiteMemoryStore."""

    def setUp(self):
        self.store_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.store_temp.cleanup)
        self.store = SQLiteMemoryStore(Path(self.store_temp.name) / "embed.sqlite3")
        self.addCleanup(self.store.close)
        self.store.upsert_agent("zoe", "Role", "zoe")
        self.store.create_session("s_embed")

    def test_store_embedding_and_search_semantic(self):
        if not self.store._vec_enabled:
            self.skipTest("sqlite-vec not available")
        msg_id = self.store.append_message(
            "s_embed", "zoe", "zoe", "semantic pool chemistry test"
        )
        vector = [0.1] * 768
        self.store.store_embedding("messages", str(msg_id), vector, "test-model")
        results = self.store.search_semantic(
            "pool chemistry", "test-model", agent_id="zoe", limit=5,
            _query_vector=vector,
        )
        record_ids = [r.record_id for r in results]
        self.assertIn(str(msg_id), record_ids)

    def test_store_embedding_idempotent(self):
        if not self.store._vec_enabled:
            self.skipTest("sqlite-vec not available")
        msg_id = self.store.append_message("s_embed", "zoe", "zoe", "embedding idempotency check")
        vector = [0.2] * 384
        eid1 = self.store.store_embedding("messages", str(msg_id), vector, "model-a")
        eid2 = self.store.store_embedding("messages", str(msg_id), vector, "model-a")
        self.assertEqual(eid1, eid2)

    def test_get_pending_embedding_records_returns_unembedded(self):
        if not self.store._vec_enabled:
            self.skipTest("sqlite-vec not available")
        self.store.append_message("s_embed", "zoe", "zoe", "unembedded record abc")
        pending = self.store.get_pending_embedding_records("test-model", limit=50)
        contents = [r["content"] for r in pending]
        self.assertTrue(any("unembedded record abc" in c for c in contents))

    def test_search_hybrid_falls_back_to_fts_without_model(self):
        self.store.append_message("s_embed", "zoe", "zoe", "hybrid fts only search xyz99")
        results = self.store.search_hybrid(
            "xyz99", "nonexistent-model", agent_id="zoe", limit=10
        )
        contents = [r.content for r in results]
        self.assertTrue(any("xyz99" in c for c in contents))

    def test_search_semantic_cascades_parent_digest(self):
        if not self.store._vec_enabled:
            self.skipTest("sqlite-vec not available")
        self.store.create_session("s_casc")
        msg_id = self.store.append_message(
            "s_casc", "zoe", "zoe", "semantic cascade unique_sem_kw"
        )
        digest_id = self.store.append_memory_digest(
            "Semantic cascade digest summary.",
            [{"source_table": "messages", "source_id": msg_id}],
            agent_id="zoe",
            session_id="s_casc",
        )
        vector = [0.5] * 768
        self.store.store_embedding("messages", str(msg_id), vector, "test-model")
        results = self.store.search_semantic(
            "cascade", "test-model", agent_id="zoe", limit=10,
            _query_vector=vector,
        )
        kinds = {r.kind for r in results}
        record_ids = {r.record_id for r in results}
        self.assertIn("memory_digest", kinds)
        self.assertIn(str(digest_id), record_ids)

    def test_search_semantic_cascade_deduplicates_existing_digest_hits(self):
        if not self.store._vec_enabled:
            self.skipTest("sqlite-vec not available")
        self.store.create_session("s_dedup")
        msg_id = self.store.append_message(
            "s_dedup", "zoe", "zoe", "dedup cascade content"
        )
        digest_id = self.store.append_memory_digest(
            "Digest already in semantic hits.",
            [{"source_table": "messages", "source_id": msg_id}],
            agent_id="zoe",
            session_id="s_dedup",
        )
        msg_vector = [0.3] * 768
        digest_vector = [0.3] * 768
        self.store.store_embedding("messages", str(msg_id), msg_vector, "test-model")
        self.store.store_embedding("memory_digests", str(digest_id), digest_vector, "test-model")
        results = self.store.search_semantic(
            "dedup", "test-model", agent_id="zoe", limit=10,
            _query_vector=digest_vector,
        )
        digest_hits = [r for r in results if r.kind == "memory_digest" and r.record_id == str(digest_id)]
        self.assertEqual(len(digest_hits), 1)

    def test_vec_table_created_per_dimension(self):
        if not self.store._vec_enabled:
            self.skipTest("sqlite-vec not available")
        self.store._ensure_vec_table(128)
        self.store._ensure_vec_table(256)
        self.assertIn(128, self.store._vec_tables)
        self.assertIn(256, self.store._vec_tables)


class DirectedRoutingTests(unittest.TestCase):
    """Tests for #96 — route_message resolves by username, full_name, role key, and first name."""

    def _make_participants(self):
        se = FakeRole("alex_chen", ["ok"])
        se.role_key = "SE"
        se.full_name = "Alex Chen"
        se.first_name = "Alex"
        hr = FakeRole("HR", ["ok"])
        participants = {
            "CEO": FakeRole("CEO", ["initial"]),
            "Ops": FakeRole("Ops", ["ok"]),
            "alex_chen": se,
            "HR": hr,
        }
        return participants

    def _route(self, participants, message, last="CEO"):
        session = main.Session(participants=participants, run_id="routing-test")
        with redirect_stdout(StringIO()):
            return session.route_message(message, last)

    def test_route_by_participant_key(self):
        p = self._make_participants()
        routed = self._route(p, "alex_chen, do this")
        self.assertTrue(routed.directed)
        self.assertIs(routed.receiver, p["alex_chen"])
        self.assertEqual(routed.prompt, "do this")

    def test_route_by_role_key_alias(self):
        p = self._make_participants()
        routed = self._route(p, "SE, do this")
        self.assertTrue(routed.directed)
        self.assertIs(routed.receiver, p["alex_chen"])
        self.assertEqual(routed.prompt, "do this")

    def test_route_by_full_name(self):
        p = self._make_participants()
        routed = self._route(p, "Alex Chen, do this")
        self.assertTrue(routed.directed)
        self.assertIs(routed.receiver, p["alex_chen"])

    def test_route_by_first_name(self):
        p = self._make_participants()
        routed = self._route(p, "Alex, do this")
        self.assertTrue(routed.directed)
        self.assertIs(routed.receiver, p["alex_chen"])

    def test_route_case_insensitive(self):
        p = self._make_participants()
        routed = self._route(p, "se, do this")
        self.assertTrue(routed.directed)
        self.assertIs(routed.receiver, p["alex_chen"])

    def test_unknown_prefix_falls_through(self):
        p = self._make_participants()
        routed = self._route(p, "Finance, do this")
        self.assertFalse(routed.directed)

    def test_name_to_key_includes_role_alias(self):
        p = self._make_participants()
        session = main.Session(participants=p, run_id="routing-test")
        self.assertEqual(session._name_to_key.get("se"), "alex_chen")

    def test_name_to_key_includes_full_name(self):
        p = self._make_participants()
        session = main.Session(participants=p, run_id="routing-test")
        self.assertEqual(session._name_to_key.get("alex chen"), "alex_chen")

    def test_canonical_agent_id_resolves_role_alias(self):
        p = self._make_participants()
        session = main.Session(participants=p, run_id="routing-test")
        self.assertEqual(session._canonical_agent_id("SE"), "alex_chen")

    def test_canonical_agent_id_passthrough_for_unknown(self):
        p = self._make_participants()
        session = main.Session(participants=p, run_id="routing-test")
        self.assertEqual(session._canonical_agent_id("Unknown"), "Unknown")

    def test_route_by_role_key_with_real_persona_build(self):
        """Role alias routing works for persona-built participants (role_key set by build_employee_dict)."""
        from robits.core.roles import build_employee_dict
        persona_map = {
            "alex_chen": {"role": "SE", "full_name": "Alex Chen", "entries": []},
        }
        participants = build_employee_dict(persona_map)
        session = main.Session(participants=participants, run_id="routing-real")
        with redirect_stdout(StringIO()):
            routed = session.route_message("SE, implement the feature", "CEO")
        self.assertTrue(routed.directed)
        self.assertEqual(routed.receiver.name, "alex_chen")


class ContextTrimTests(unittest.TestCase):
    """Tests for #103 — ROBITS_MAX_CONTEXT_TOKENS trims conversation_history before generate()."""

    def _fake_role(self, history):
        """Build a minimal role-like object with the given conversation history."""
        role = SimpleNamespace(
            name="SE",
            template="system prompt",
            conversation_history={"SE": history},
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
            employee_dict={},
        )
        return role

    def test_trimming_disabled_when_zero(self):
        """No trimming when ROBITS_MAX_CONTEXT_TOKENS=0 (default)."""
        captured = []

        def fake_generate(role, model, sender, messages):
            captured.append(list(messages))
            return "ok"

        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1", "name": "CEO"},
            {"role": "assistant", "content": "rep1", "name": "SE"},
            {"role": "user", "content": "msg2", "name": "CEO"},
        ]
        role = self._fake_role(history)
        expected_len = len(history)
        orig = main._config.max_context_tokens
        orig_provider = main._config.model_provider
        try:
            main._config.max_context_tokens = 0
            main._config.model_provider = SimpleNamespace(generate=fake_generate)
            with redirect_stdout(StringIO()):
                from robits.core.roles import interact
                interact(role, "model", "CEO", None)
        finally:
            main._config.max_context_tokens = orig
            main._config.model_provider = orig_provider
        self.assertEqual(len(captured[0]), expected_len)

    def test_trimming_drops_oldest_non_system_messages(self):
        """Old turns are dropped to fit within the token budget; system + newest kept."""
        captured = []

        def fake_generate(role, model, sender, messages):
            captured.append(list(messages))
            return "ok"

        long_content = "x" * 400
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long_content, "name": "CEO"},
            {"role": "assistant", "content": long_content, "name": "SE"},
            {"role": "user", "content": "new msg", "name": "CEO"},
        ]
        role = self._fake_role(history)
        orig = main._config.max_context_tokens
        orig_provider = main._config.model_provider
        try:
            # Budget: 50 tokens * 4 chars = 200 chars — fits system + new msg but not the long history
            main._config.max_context_tokens = 50
            main._config.model_provider = SimpleNamespace(generate=fake_generate)
            with redirect_stdout(StringIO()):
                from robits.core.roles import interact
                interact(role, "model", "CEO", None)
        finally:
            main._config.max_context_tokens = orig
            main._config.model_provider = orig_provider
        sent = captured[0]
        self.assertEqual(sent[0]["role"], "system")
        self.assertLess(len(sent), len(history))
        # Most recent message must always be present
        self.assertTrue(any(m.get("content") == "new msg" for m in sent))

    def test_trimming_does_not_mutate_conversation_history(self):
        """Trimming is applied only to the messages sent to generate(), not stored history."""
        captured = []

        def fake_generate(role, model, sender, messages):
            captured.append(list(messages))
            return "ok"

        long_content = "y" * 400
        history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long_content, "name": "CEO"},
            {"role": "assistant", "content": long_content, "name": "SE"},
            {"role": "user", "content": "check history", "name": "CEO"},
        ]
        role = self._fake_role(history)
        orig_len = len(history)
        orig = main._config.max_context_tokens
        orig_provider = main._config.model_provider
        try:
            main._config.max_context_tokens = 50
            main._config.model_provider = SimpleNamespace(generate=fake_generate)
            with redirect_stdout(StringIO()):
                from robits.core.roles import interact
                interact(role, "model", "CEO", None)
        finally:
            main._config.max_context_tokens = orig
            main._config.model_provider = orig_provider
        # The stored history grows by one (the assistant reply), trimming must not shrink it
        self.assertGreaterEqual(len(role.conversation_history["SE"]), orig_len)


if __name__ == "__main__":
    unittest.main()

"""End-to-end tests for the agent-driven tool proposal and activation pipeline.

Verifies that SE can propose a tool with code, Ops can approve and roll it out,
and the newly registered tool is immediately callable within the same session.
"""
import json
import tempfile
import unittest
from io import StringIO
from contextlib import redirect_stdout

import main
from robits.runtime.tool_proposals import ToolProposalStore


def _make_employee_dict():
    """Minimal employee dict with SE and Ops roles for permission checks."""
    employee_dict = {}
    employee_dict["Ops"] = main.Ops(employee_dict)
    employee_dict["SE"] = main.SoftwareEngineer(employee_dict)
    return employee_dict


class ToolProposalPipelineTests(unittest.TestCase):
    def setUp(self):
        main.tool_registry.clear()
        self.original_store = main.tool_proposal_store
        main.tool_proposal_store = ToolProposalStore()
        system = main.System()
        main.load_tools(system)
        self.addCleanup(self._restore)

    def _restore(self):
        main.tool_proposal_store = self.original_store
        main.tool_registry.clear()
        system = main.System()
        main.load_tools(system)

    # --- ToolProposalStore code field ---

    def test_proposal_stores_code_field(self):
        store = ToolProposalStore()
        proposal = store.create(
            requested_by="SE",
            tool_name="team.pulse",
            description="Report calling agent status.",
            code='return f"{get_caller_name()}: {status}"',
        )
        self.assertEqual(proposal["code"], 'return f"{get_caller_name()}: {status}"')

    def test_proposal_code_defaults_to_empty_string(self):
        store = ToolProposalStore()
        proposal = store.create(
            requested_by="SE",
            tool_name="team.pulse",
            description="Report calling agent status.",
        )
        self.assertEqual(proposal["code"], "")

    def test_proposal_code_survives_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            import pathlib
            path = pathlib.Path(tmp) / "proposals.json"
            store = ToolProposalStore(path)
            store.create(
                requested_by="SE",
                tool_name="team.pulse",
                description="Status tool.",
                code='return f"{get_caller_name()}: {status}"',
            )
            reloaded = ToolProposalStore(path)
            proposals = reloaded.list()
        self.assertEqual(proposals[0]["code"], 'return f"{get_caller_name()}: {status}"')

    # --- propose_tool_change validates code at proposal time ---

    def test_propose_with_valid_code_succeeds(self):
        employee_dict = _make_employee_dict()
        result = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="Report calling agent status.",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            code='return f"{get_caller_name()}: {status}"',
        ))
        self.assertEqual(result["tool_name"], "team.pulse")
        self.assertEqual(result["code"], 'return f"{get_caller_name()}: {status}"')
        self.assertEqual(result["status"], "proposed")

    def test_propose_with_syntax_error_is_rejected(self):
        employee_dict = _make_employee_dict()
        result = main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.broken",
            description="Broken tool.",
            code="return (",
        )
        self.assertIn("Error:", result)
        self.assertIn("syntax error", result.lower())

    def test_propose_with_function_definition_is_rejected(self):
        employee_dict = _make_employee_dict()
        result = main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.bad",
            description="Uses nested def.",
            parameters={"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
            code='def helper(s):\n    return s\nreturn helper(status)',
        )
        self.assertIn("Error:", result)
        self.assertIn("function", result)

    def test_propose_without_code_still_succeeds(self):
        employee_dict = _make_employee_dict()
        result = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.future",
            description="TBD.",
        ))
        self.assertEqual(result["code"], "")
        self.assertEqual(result["status"], "proposed")

    # --- rollout auto-registers from proposal code ---

    def test_rollout_registers_and_grants_tool_from_code(self):
        employee_dict = _make_employee_dict()
        proposal = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="Report calling agent status.",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            code='return f"{get_caller_name()}: {status}"',
        ))
        proposal_id = proposal["proposal_id"]

        main.approve_tool_proposal(employee_dict, proposal_id, approved_by="Ops")

        result = json.loads(main.rollout_tool_proposal(
            employee_dict, proposal_id, role_name="SE", granted_by="Ops"
        ))
        self.assertEqual(result["proposal"]["status"], "operationalized")
        self.assertIn("team.pulse", main.tool_registry)

    def test_rolled_out_tool_is_immediately_callable(self):
        employee_dict = _make_employee_dict()
        se_role = employee_dict["SE"]

        proposal = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="Report calling agent status.",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            code='return f"{get_caller_name()}: {status}"',
        ))
        proposal_id = proposal["proposal_id"]
        main.approve_tool_proposal(employee_dict, proposal_id, approved_by="Ops")
        main.rollout_tool_proposal(employee_dict, proposal_id, role_name="SE", granted_by="Ops")

        result = main.tool_registry.execute(
            "team.pulse", {"status": "on-task"}, employee_dict, caller=se_role
        )
        self.assertIn("SE: on-task", result)

    def test_rollout_without_code_fails_when_tool_unregistered(self):
        employee_dict = _make_employee_dict()
        proposal = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.future",
            description="Not implemented yet.",
        ))
        proposal_id = proposal["proposal_id"]
        main.approve_tool_proposal(employee_dict, proposal_id, approved_by="Ops")

        result = main.rollout_tool_proposal(
            employee_dict, proposal_id, role_name="SE", granted_by="Ops"
        )
        self.assertIn("Error:", result)
        self.assertNotIn("team.future", main.tool_registry)

    # --- update proposals replace existing tools ---

    def test_update_proposal_replaces_tool_implementation(self):
        employee_dict = _make_employee_dict()
        se_role = employee_dict["SE"]

        # Create initial version
        create_proposal = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="v1: plain status.",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            code='return status',
        ))
        main.approve_tool_proposal(employee_dict, create_proposal["proposal_id"], approved_by="Ops")
        main.rollout_tool_proposal(employee_dict, create_proposal["proposal_id"], role_name="SE", granted_by="Ops")

        # Update with richer implementation
        update_proposal = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="v2: status with caller.",
            action="update",
            parameters={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            code='return f"{get_caller_name()}: {status}"',
        ))
        main.approve_tool_proposal(employee_dict, update_proposal["proposal_id"], approved_by="Ops")
        main.rollout_tool_proposal(employee_dict, update_proposal["proposal_id"], role_name="SE", granted_by="Ops")

        result = main.tool_registry.execute(
            "team.pulse", {"status": "reviewing"}, employee_dict, caller=se_role
        )
        self.assertIn("SE: reviewing", result)

    # --- flat property-map parameters are auto-normalized ---

    def test_propose_flat_property_map_is_normalized_to_json_schema(self):
        employee_dict = _make_employee_dict()
        result = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="Status tool.",
            parameters={"status": {"type": "string", "description": "Current status."}},
            code='return f"{get_caller_name()}: {status}"',
        ))
        params = result["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("status", params["properties"])
        self.assertIn("status", params["required"])

    def test_propose_already_correct_schema_is_unchanged(self):
        employee_dict = _make_employee_dict()
        full_schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }
        result = json.loads(main.propose_tool_change(
            employee_dict,
            requested_by="SE",
            tool_name="team.pulse",
            description="Status tool.",
            parameters=full_schema,
            code='return f"{get_caller_name()}: {status}"',
        ))
        self.assertEqual(result["parameters"], full_schema)

    # --- get_caller_name is available in sandbox ---

    def test_get_caller_name_available_in_compiled_tool(self):
        func = main.tool_registry.compile_tool(
            "test.identity", [], [], "return get_caller_name()"
        )
        saved_caller = main.active_tool_caller
        saved_name = main.active_tool_caller_name
        main.active_tool_caller_name = "TestAgent"
        try:
            result = func(employee_dict={})
        finally:
            main.active_tool_caller = saved_caller
            main.active_tool_caller_name = saved_name
        self.assertEqual(result, "TestAgent")

    def test_get_caller_name_returns_unknown_when_no_caller(self):
        func = main.tool_registry.compile_tool(
            "test.identity", [], [], "return get_caller_name()"
        )
        saved_caller = main.active_tool_caller
        saved_name = main.active_tool_caller_name
        main.active_tool_caller = None
        main.active_tool_caller_name = None
        try:
            result = func(employee_dict={})
        finally:
            main.active_tool_caller = saved_caller
            main.active_tool_caller_name = saved_name
        self.assertEqual(result, "unknown")

    # --- agent.think: silent, no side effects ---

    def test_agent_think_returns_empty_string(self):
        func = main.tool_registry.compile_tool(
            "test.think", ["content"], ["content"], "return agent_think(employee_dict, content)"
        )
        result = func(employee_dict={}, content="This is a private thought.")
        self.assertEqual(result, "")

    def test_agent_think_available_after_load(self):
        employee_dict = _make_employee_dict()
        result = main.tool_registry.execute(
            "agent.think", {"content": "Planning next step."}, employee_dict, caller=employee_dict["SE"]
        )
        self.assertEqual(result, "")

    # --- agent.wait: sets waiting_until on the caller role ---

    def test_agent_wait_sets_waiting_until(self):
        from datetime import datetime, timezone, timedelta
        employee_dict = _make_employee_dict()
        se_role = employee_dict["SE"]
        saved_caller = main.active_tool_caller
        saved_transcript_len = main.active_session_transcript_length
        main.active_tool_caller = se_role
        main.active_session_transcript_length = 0
        before = datetime.now(timezone.utc)
        try:
            result = main.agent_wait(employee_dict={}, minutes=10)
        finally:
            main.active_tool_caller = saved_caller
            main.active_session_transcript_length = saved_transcript_len
        after = datetime.now(timezone.utc)
        self.assertEqual(result, "")
        self.assertIsNotNone(se_role.waiting_until)
        expected_min = before + timedelta(minutes=10)
        expected_max = after + timedelta(minutes=10)
        self.assertGreaterEqual(se_role.waiting_until, expected_min)
        self.assertLessEqual(se_role.waiting_until, expected_max)
        self.assertEqual(se_role.wait_started_turn, 0)

    def test_agent_wait_rejects_nonpositive_minutes(self):
        employee_dict = _make_employee_dict()
        se_role = employee_dict["SE"]
        saved_caller = main.active_tool_caller
        main.active_tool_caller = se_role
        try:
            result = main.agent_wait(employee_dict={}, minutes=0)
        finally:
            main.active_tool_caller = saved_caller
        self.assertIn("Error:", result)

    def test_agent_wait_rejects_no_caller(self):
        main.active_tool_caller = None
        result = main.agent_wait(employee_dict={}, minutes=5)
        self.assertIn("Error:", result)

    # --- Session: waiting agents are skipped in round-robin ---

    def test_session_skips_waiting_agent_in_round_robin(self):
        from datetime import datetime, timezone, timedelta
        from types import SimpleNamespace
        import main as m

        employee_dict = _make_employee_dict()
        system = m.System(employee_dict)
        with redirect_stdout(StringIO()):
            m.load_tools(system)

        session = m.Session(participants=employee_dict, clock_state="on")

        # Mark SE as waiting for 60 minutes
        se_role = employee_dict["SE"]
        se_role.waiting_until = datetime.now(timezone.utc) + timedelta(minutes=60)
        se_role.wait_started_turn = 0
        se_role.wait_clock_state = "on"

        # Route from Ops — SE should be skipped
        routed = session.route_message("test message", "Ops")
        self.assertNotEqual(routed.receiver.name, "SE")

    def test_session_wake_summary_injected_on_expiry(self):
        from datetime import datetime, timezone, timedelta
        import main as m

        employee_dict = _make_employee_dict()
        system = m.System(employee_dict)
        with redirect_stdout(StringIO()):
            m.load_tools(system)

        session = m.Session(participants=employee_dict, clock_state="on")

        # Simulate a past wait (already expired)
        se_role = employee_dict["SE"]
        se_role.waiting_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        se_role.wait_started_turn = 0
        se_role.wait_clock_state = "on"

        summary = session._build_wait_summary(se_role)
        self.assertIn("wait has ended", summary.lower())
        self.assertIsNone(se_role.waiting_until)

    def test_session_wait_interrupted_by_clock_state_change(self):
        from datetime import datetime, timezone, timedelta
        import main as m

        employee_dict = _make_employee_dict()
        system = m.System(employee_dict)
        with redirect_stdout(StringIO()):
            m.load_tools(system)

        session = m.Session(participants=employee_dict, clock_state="on")

        se_role = employee_dict["SE"]
        se_role.waiting_until = datetime.now(timezone.utc) + timedelta(minutes=60)
        se_role.wait_started_turn = 0
        se_role.wait_clock_state = "on"
        session.clock_state = "off"  # simulate circadian phase transition at session level

        interrupted = session._interrupt_wait_for_phase(se_role)
        self.assertTrue(interrupted)
        self.assertIsNone(se_role.waiting_until)


if __name__ == "__main__":
    unittest.main()

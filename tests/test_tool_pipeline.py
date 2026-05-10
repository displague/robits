"""End-to-end tests for the agent-driven tool proposal and activation pipeline.

Verifies that SE can propose a tool with code, Ops can approve and roll it out,
and the newly registered tool is immediately callable within the same session.
"""
import json
import tempfile
import unittest

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
        main.active_tool_caller_name = "TestAgent"
        try:
            result = func(employee_dict={})
        finally:
            main.active_tool_caller_name = None
        self.assertEqual(result, "TestAgent")

    def test_get_caller_name_returns_unknown_when_no_caller(self):
        func = main.tool_registry.compile_tool(
            "test.identity", [], [], "return get_caller_name()"
        )
        main.active_tool_caller = None
        main.active_tool_caller_name = None
        result = func(employee_dict={})
        self.assertEqual(result, "unknown")


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import main
from robits.core.persona import load_personas
from robits.core.roles import Role, build_employee_dict, SoftwareEngineer
from robits.core.session import Session
from robits.memory.sqlite import SQLiteMemoryStore


class PrepromptTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore(":memory:")
        self.store.ensure_schema()
        # Backup global state
        from robits.core.config import _config as _m
        self.old_store = _m.memory_store
        _m.memory_store = self.store

    def tearDown(self):
        from robits.core.config import _config as _m
        self.store.close()
        _m.memory_store = self.old_store

    def test_load_personas_custom_preprompts(self):
        content = """
- username: custom_agent
  role: SoftwareEngineer
  full_name: Custom Developer
  preprompt_work: "Work instructions"
  preprompt_personal: "Personal instructions"
  memories:
    - kind: thought
      content: "Initial seed thought"
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name

        try:
            personas = load_personas(temp_path)
            self.assertIn("custom_agent", personas)
            self.assertEqual(personas["custom_agent"]["preprompt_work"], "Work instructions")
            self.assertEqual(personas["custom_agent"]["preprompt_personal"], "Personal instructions")
        finally:
            Path(temp_path).unlink()

    def test_build_employee_dict_copies_preprompts(self):
        persona_map = {
            "custom_se": {
                "role": "SoftwareEngineer",
                "full_name": "Custom Software Engineer",
                "preprompt_work": "Write robust code.",
                "preprompt_personal": "Code in leisure time.",
                "entries": []
            }
        }
        employee_dict = build_employee_dict(persona_map)
        self.assertIn("custom_se", employee_dict)
        se = employee_dict["custom_se"]
        self.assertEqual(se.preprompt_work, "Write robust code.")
        self.assertEqual(se.preprompt_personal, "Code in leisure time.")

    def test_agent_metadata_operations(self):
        self.store.upsert_agent(
            "test_agent",
            "SoftwareEngineer",
            metadata={"preprompt_work": "Base work instruction", "tag": "test"}
        )
        metadata = self.store.get_agent_metadata("test_agent")
        self.assertEqual(metadata.get("preprompt_work"), "Base work instruction")
        self.assertEqual(metadata.get("tag"), "test")

        self.store.update_agent_metadata("test_agent", {"tag": "updated", "new_field": 42})
        metadata = self.store.get_agent_metadata("test_agent")
        self.assertEqual(metadata.get("preprompt_work"), "Base work instruction")
        self.assertEqual(metadata.get("tag"), "updated")
        self.assertEqual(metadata.get("new_field"), 42)

    @patch("robits.core.config._config.model_provider")
    def test_interact_circadian_blend(self, mock_provider):
        mock_provider.generate.return_value = "Mocked Response"
        employee_dict = build_employee_dict()
        se = employee_dict["SE"]
        se.preprompt_work = "Focus on software architecture."
        se.preprompt_personal = "Play video games."

        self.store.upsert_agent("SE", "SoftwareEngineer")
        self.store.set_agent_phase("SE", 0.5)

        # On-clock state
        se.runtime_clock_state = "on"
        se.interact("CEO", "Hello")

        calls = mock_provider.generate.call_args_list
        self.assertTrue(calls)
        messages = calls[-1][0][3]  # messages argument
        system_content = messages[0]["content"]

        self.assertIn("=== Persona Circadian Rhythm & State ===", system_content)
        self.assertIn("Active clock state: on", system_content)
        self.assertIn("Circadian phase: 0.50", system_content)
        self.assertIn("Work Persona Instructions", system_content)
        self.assertIn("Focus on software architecture.", system_content)
        self.assertIn("Play video games.", system_content)

    def test_agent_adjust_preprompt_and_decay(self):
        employee_dict = build_employee_dict()
        se = employee_dict["SE"]
        se.runtime_role_name = "SE"
        se.name = "SE"
        self.store.upsert_agent("SE", "SoftwareEngineer")

        # Fake the active tool caller
        with patch("robits.core.tool_functions._m.active_tool_caller", se), \
             patch("robits.core.tool_functions._m.active_tool_caller_name", "SE"):
            
            from robits.core.tool_functions import agent_adjust_preprompt
            res = agent_adjust_preprompt(employee_dict, "Write more comments", turns=3, mode="work")
            self.assertIn("Successfully applied temporary work adjustment", res)

            # Assert it was written to db
            metadata = self.store.get_agent_metadata("SE")
            self.assertEqual(metadata.get("work_adjustment"), "Write more comments")
            self.assertEqual(metadata.get("work_adjustment_turns"), 3)

            # Setup session
            session = Session(participants=employee_dict, run_id="test-session")
            session.runtime_clock_state = "on"

            # 1st step decay
            session._decay_agent_adjustments("SE")
            metadata = self.store.get_agent_metadata("SE")
            self.assertEqual(metadata.get("work_adjustment_turns"), 2)
            self.assertEqual(metadata.get("work_adjustment"), "Write more comments")

            # 2nd step decay
            session._decay_agent_adjustments("SE")
            metadata = self.store.get_agent_metadata("SE")
            self.assertEqual(metadata.get("work_adjustment_turns"), 1)

            # 3rd step decay should clear it
            session._decay_agent_adjustments("SE")
            metadata = self.store.get_agent_metadata("SE")
            self.assertIsNone(metadata.get("work_adjustment"))
            self.assertIsNone(metadata.get("work_adjustment_turns"))

    def test_clock_transition_clears_opposing_adjustments(self):
        employee_dict = build_employee_dict()
        se = employee_dict["SE"]
        se.name = "SE"
        self.store.upsert_agent("SE", "SoftwareEngineer")

        # Set work adjustment in db
        self.store.update_agent_metadata("SE", {
            "work_adjustment": "Work hard",
            "work_adjustment_turns": 5
        })

        session = Session(participants=employee_dict, run_id="test-session")
        # Clock shifts off-clock
        with patch.object(session, "_effective_clock_state", return_value="off"):
            session._decay_agent_adjustments("SE")
            metadata = self.store.get_agent_metadata("SE")
            # Work adjustment should be cleared immediately since we are off-clock
            self.assertIsNone(metadata.get("work_adjustment"))
            self.assertIsNone(metadata.get("work_adjustment_turns"))

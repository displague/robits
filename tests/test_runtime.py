import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import main


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


if __name__ == "__main__":
    unittest.main()

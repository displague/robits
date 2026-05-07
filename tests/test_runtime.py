import json
import unittest
from contextlib import redirect_stdout
from io import StringIO

import main


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        main.escape_codes.clear()

    def test_parse_escape_code_handles_surrounding_text(self):
        response = 'Ops should run this:\n{"exec": "create_role", "args": {"role_name": "QA"}}\nDone.'

        self.assertEqual(
            json.loads(main.parse_escape_code(response)),
            {"exec": "create_role", "args": {"role_name": "QA"}},
        )

    def test_parse_escape_code_handles_multiline_json(self):
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
            json.loads(main.parse_escape_code(response)),
            {
                "exec": "create_role",
                "args": {
                    "role_name": "QA",
                    "role_description": "Tests the organization",
                },
            },
        )

    def test_preloaded_create_role_executes_against_employee_dict(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.preload(system)

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
            main.preload(system)

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

    def test_escape_code_reports_missing_args(self):
        employee_dict = main.build_employee_dict()
        system = main.System(employee_dict)
        with redirect_stdout(StringIO()):
            main.preload(system)

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

    def test_untrusted_escape_code_definition_is_rejected(self):
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

        self.assertIn("trusted preload", response)
        self.assertNotIn("surprise", main.escape_codes)

    def test_escape_code_builtins_are_restricted(self):
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

        self.assertIn("Stored escape code", response)
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


if __name__ == "__main__":
    unittest.main()

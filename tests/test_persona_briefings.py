"""Persona directories and on-demand briefings (#141); no model server required."""
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.persona import load_personas
from robits.core.providers import ChatCompletionsProvider
from robits.core.roles import format_briefing_index
from tests.test_persona_tools import chat_client, chat_message

STYLE_BODY = "Lead with the user-visible change. One line per fix. Link the issue."


def write_roster(root: Path):
    (root / "briefings").mkdir()
    (root / "briefings" / "release-notes.md").write_text(
        "---\nsummary: How we write release notes\n---\n# Release notes\n\n" + STYLE_BODY + "\n")
    (root / "briefings" / "tracker.md").write_text("# Using the task tracker\n\nClaim before you start.\n")
    (root / "riley.yaml").write_text(
        "full_name: Riley Park\nrole: Ops\n"
        "tools: [agent.think, memory.search]\ntool_budget: 1\n"
        "briefings: [tracker]\n")
    (root / "sam.yml").write_text(
        "username: sam_writer\nfull_name: Sam Ortiz\nrole: SE\n"
        "briefings:\n  - release-notes\n"
        "  - {name: on-call, summary: Who to page and when, text: Page the on-call engineer for outages only.}\n"
        "memories:\n  - {kind: digest, digest_type: identity, content: Sam maintains the docs., relationship_type: work}\n")
    (root / "notes.txt").write_text("not a persona")


class BriefingTestCase(unittest.TestCase):
    def setUp(self):
        main.tool_registry.clear()
        scope = patch.multiple(config, memory_store=None, active_tool_caller=None, mcp_config_path="")
        scope.start()
        self.addCleanup(scope.stop)
        self.addCleanup(main.tool_registry.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def build(self, path):
        employee_dict = main.build_employee_dict(load_personas(str(path)), headless=True)
        with redirect_stdout(StringIO()):
            main.load_tools(main.System(employee_dict))
        return employee_dict


class PersonaDirectoryTests(BriefingTestCase):
    def test_directory_loads_one_persona_per_file(self):
        write_roster(self.root)
        personas = load_personas(str(self.root))
        self.assertEqual(sorted(personas), ["riley", "sam_writer"], "username defaults to the file name")
        self.assertEqual(personas["sam_writer"]["full_name"], "Sam Ortiz")
        self.assertEqual(len(personas["sam_writer"]["entries"]), 1)
        self.assertEqual(personas["riley"]["entries"], [], "memories are optional")

    def test_briefings_resolve_from_directory_and_inline(self):
        write_roster(self.root)
        sam = self.build(self.root)["sam_writer"]
        self.assertEqual(list(sam.briefings), ["release-notes", "on-call"])
        self.assertEqual(sam.briefings["release-notes"]["summary"], "How we write release notes")
        self.assertIn(STYLE_BODY, sam.briefings["release-notes"]["text"])
        self.assertEqual(sam.briefings["on-call"]["text"], "Page the on-call engineer for outages only.")
        riley = self.build(self.root)["riley"]
        self.assertEqual(riley.briefings["tracker"]["summary"], "Using the task tracker",
                         "without front matter the first line is the summary")

    def test_single_file_reads_briefings_next_to_it(self):
        write_roster(self.root)
        roster = self.root / "personas.yaml"
        roster.write_text("- username: solo\n  role: SE\n  briefings: [release-notes]\n  memories: []\n")
        solo = self.build(roster)["solo"]
        self.assertIn(STYLE_BODY, solo.briefings["release-notes"]["text"])

    def test_bad_briefings_skip_the_persona(self):
        write_roster(self.root)
        for bad in ("[missing]", "['../riley']", "release-notes", "[{name: x}]", "[tracker, tracker]"):
            (self.root / "riley.yaml").write_text(f"role: Ops\nbriefings: {bad}\n")
            with self.subTest(bad=bad), self.assertLogs(level="WARNING") as logs:
                employee_dict = main.build_employee_dict(load_personas(str(self.root)), headless=True)
            self.assertNotIn("riley", employee_dict)
            self.assertIn("sam_writer", employee_dict)
            self.assertTrue(any("invalid briefings" in line for line in logs.output))


class BriefingPromptTests(BriefingTestCase):
    def test_only_the_index_is_in_the_prompt_until_the_persona_reads_a_briefing(self):
        write_roster(self.root)
        sam = self.build(self.root)["sam_writer"]
        client, calls = chat_client([
            chat_message(calls=[("agent__read_briefing", json.dumps({"name": "release-notes"}))]),
            chat_message("Notes drafted."),
        ])
        provider = ChatCompletionsProvider(client, main.tool_registry)
        with patch.object(config, "model_provider", provider), redirect_stdout(StringIO()):
            reply = sam.interact("riley", "Draft the release notes for 2.3.")
        self.assertEqual(reply, "Notes drafted.")

        system = calls[0]["messages"][0]["content"]
        self.assertIn("=== Briefings ===", system)
        self.assertIn("- release-notes: How we write release notes", system)
        self.assertIn("- on-call: Who to page and when", system)
        self.assertNotIn(STYLE_BODY, json.dumps(calls[0]["messages"]), "body not sent before it is read")
        tool_results = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn(STYLE_BODY, tool_results[-1]["content"])
        self.assertNotIn(STYLE_BODY, calls[1]["messages"][0]["content"], "system prompt keeps only the index")

    def test_no_briefings_no_index(self):
        write_roster(self.root)
        (self.root / "riley.yaml").write_text("role: Ops\n")
        self.assertEqual(format_briefing_index(self.build(self.root)["riley"]), "")


class BriefingToolTests(BriefingTestCase):
    def test_read_briefing_is_offered_first_within_the_tool_budget(self):
        write_roster(self.root)
        riley = self.build(self.root)["riley"]
        self.assertEqual(riley.tool_ceiling[0], "agent.read_briefing")
        self.assertEqual([t.name for t in main.tool_registry.offered_tools(riley)], ["agent.read_briefing"])

    def test_unknown_briefing_lists_the_persona_s_briefings(self):
        write_roster(self.root)
        employee_dict = self.build(self.root)
        sam = employee_dict["sam_writer"]
        result = main.tool_registry.execute("agent.read_briefing", {"name": "nope"}, employee_dict, caller=sam)
        self.assertIn("release-notes, on-call", result)
        (self.root / "riley.yaml").write_text("role: Ops\n")
        employee_dict = self.build(self.root)
        result = main.tool_registry.execute("agent.read_briefing", {"name": "release-notes"},
                                            employee_dict, caller=employee_dict["riley"])
        self.assertTrue(result.startswith("Error:"), "a persona can read only its own briefings")


if __name__ == "__main__":
    unittest.main()

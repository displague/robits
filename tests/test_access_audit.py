"""Accountable cross-persona workspace access (#140); no model server required."""
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import main
from robits.core import access_audit
from robits.core.config import _config as config
from robits.runtime.workspace import AgentWorkspaceStore
from tests.test_headless import FakeRole, HeadlessTestCase

SECRET = "private diary: thinking about leaving"


class AuditTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.audit_file = self.tmp / "audit" / "access.jsonl"
        main.tool_registry.clear()
        self.addCleanup(main.tool_registry.clear)
        scope = patch.multiple(config, memory_store=None, active_tool_caller=None,
                               agent_workspace_store=AgentWorkspaceStore(self.tmp / "agents"),
                               access_audit_file=str(self.audit_file), access_notice_mode="immediate",
                               access_notices={})
        scope.start()
        self.addCleanup(scope.stop)
        self.org = main.build_employee_dict()
        with redirect_stdout(StringIO()):
            main.load_tools(main.System(self.org))
        config.agent_workspace_store.write("SE", "notes/diary.md", SECRET)

    def call(self, caller, tool, **args):
        return main.tool_registry.execute(tool, args, self.org, caller=self.org[caller])

    def entries(self):
        if not self.audit_file.exists():
            return []
        return [json.loads(line) for line in self.audit_file.read_text().splitlines()]


class CrossPersonaAccessTests(AuditTestCase):
    def test_hr_read_is_audited_once_without_contents(self):
        result = self.call("HR", "agent.files_read", agent_name="SE", path="notes/diary.md",
                           reason="wellbeing check requested after missed check-ins")
        self.assertIn(SECRET, result, "the access itself still works")
        entries = self.entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual({k: entry[k] for k in ("caller", "owner", "operation", "path")},
                         {"caller": "HR", "owner": "SE", "operation": "read", "path": "notes/diary.md"})
        self.assertEqual(entry["reason"], "wellbeing check requested after missed check-ins")
        self.assertIn("ts", entry)
        self.assertNotIn(SECRET, self.audit_file.read_text())

    def test_missing_reason_is_refused(self):
        for reason in (None, "", "   "):
            args = {"agent_name": "SE", "path": "notes/diary.md"}
            if reason is not None:
                args["reason"] = reason
            result = self.call("HR", "agent.files_read", **args)
            self.assertTrue(result.startswith("Error:"), result)
            self.assertIn("reason", result)
            self.assertNotIn(SECRET, result)
        self.assertEqual(self.entries(), [])

    def test_owner_access_is_not_audited(self):
        result = self.call("SE", "agent.files_read", agent_name="SE", path="notes/diary.md")
        self.assertIn(SECRET, result)
        self.assertEqual(self.entries(), [])

    def test_every_workspace_operation_is_covered(self):
        reason = "offboarding review"
        self.org["HR"].allowed_tools.add("builtin.file_search")
        self.call("HR", "agent.files_list", agent_name="SE", reason=reason)
        self.call("HR", "builtin.file_search", agent_name="SE", query="diary", reason=reason)
        self.call("HR", "agent.files_write", agent_name="SE", path="handoff.md", content="x", reason=reason)
        self.call("HR", "agent.files_delete", agent_name="SE", path="handoff.md", reason=reason)
        self.assertEqual([e["operation"] for e in self.entries()], ["list", "search", "write", "delete"])

    def test_unauthorised_caller_still_denied(self):
        result = self.call("SE", "agent.files_read", agent_name="HR", path="x", reason="curious")
        self.assertIn("cannot access workspace", result)
        self.assertEqual(self.entries(), [])


class OwnerNoticeTests(AuditTestCase):
    def read_twice(self):
        for path in ("notes/diary.md", "notes/diary.md"):
            self.call("HR", "agent.files_read", agent_name="SE", path=path, reason="wellbeing check")

    def test_immediate_notice_names_caller_path_and_reason(self):
        self.read_twice()
        notice = access_audit.pop_notice("SE")
        self.assertEqual(notice.count("[Notice:"), 2)
        self.assertIn("HR read a file in your workspace (notes/diary.md)", notice)
        self.assertIn("Reason given: wellbeing check", notice)
        self.assertEqual(access_audit.pop_notice("SE"), "", "a notice is delivered once")

    def test_digest_notice_counts_only(self):
        config.access_notice_mode = "digest"
        self.read_twice()
        notice = access_audit.pop_notice("SE")
        self.assertIn("HR read a file in it 2 times", notice)
        self.assertNotIn("diary", notice)
        self.assertNotIn("wellbeing", notice)

    def test_no_notice_but_still_audited(self):
        config.access_notice_mode = "none"
        self.read_twice()
        self.assertEqual(access_audit.pop_notice("SE"), "")
        self.assertEqual(len(self.entries()), 2)


class ReviewToolTests(AuditTestCase):
    def test_review_is_operator_only(self):
        self.call("HR", "agent.files_read", agent_name="SE", path="notes/diary.md", reason="wellbeing check")
        denied = self.call("SE", "org.access_audit")
        self.assertIn("not allowed", denied)
        denied_hr = self.call("HR", "org.access_audit")
        self.assertIn("not allowed", denied_hr, "the auditor of record is not the audited role")
        review = json.loads(self.call("Ops", "org.access_audit", owner="SE"))
        self.assertEqual(review["count"], 1)
        self.assertEqual(review["entries"][0]["caller"], "HR")

    def test_access_duty_note_only_for_hr_and_operator(self):
        self.assertIn("logged for review", access_audit.access_duty_note(self.org["HR"]))
        self.assertIn("logged for review", access_audit.access_duty_note(self.org["Ops"]))
        self.assertEqual(access_audit.access_duty_note(self.org["SE"]), "")


class SessionNoticeTests(HeadlessTestCase):
    def test_owner_sees_notice_on_its_next_turn(self):
        with patch.object(config, "access_notices", {}), patch.object(config, "access_notice_mode", "immediate"):
            config.access_notices["A"] = [{"ts": "2026-01-05T09:00:00+00:00", "caller": "HR", "owner": "A",
                                           "operation": "list", "path": "", "reason": "audit"}]
            a = FakeRole("A", ["ok"])
            session = self.session({"A": a})
            self.run_quiet(session, initial_message="A: status?", max_turns=1)
            self.assertIn("[Notice: HR listed files in your workspace", a.received[0][1])
            self.assertIn("status?", a.received[0][1])
            self.assertEqual(config.access_notices, {})


if __name__ == "__main__":
    unittest.main()

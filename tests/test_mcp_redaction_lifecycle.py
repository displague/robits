"""#138 part 3: redaction of account fields, and persona lifecycle hooks."""
import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import main
from robits.core.config import _config as config
from robits.core.io import JsonlLogger, set_logger
from robits.core.lifecycle import archive_lifecycle_role, create_lifecycle_role, retire_lifecycle_role
from robits.core.session import RuntimeEventStream
from robits.runtime import oauth
from robits.runtime.mcp import MCPManager, redact_result
from tests.test_mcp_identities import IdentityTestCase, persona


class FakeTransport:
    def __init__(self, url=None, **kwargs):
        self.url, self.kwargs, self.result = url, kwargs, {}

    def request(self, method, params, notify):
        return self.result


class RedactionTests(unittest.TestCase):
    KEYS = frozenset({"email", "owner"})

    def test_structured_and_json_text_content_are_redacted(self):
        result = {
            "structuredContent": {"items": [{"title": "a", "Email": "x@example.org", "owner": {"name": "N"}}]},
            "content": [{"type": "text", "text": json.dumps({"email": "y@example.org", "ok": True})},
                        {"type": "text", "text": "plain text mentioning owner stays"},
                        {"type": "image", "data": "..."}],
            "_meta": {"owner": "protocol fields are left alone"},
        }
        redacted, removed = redact_result(result, self.KEYS)
        self.assertEqual(removed, 3)
        self.assertEqual(redacted["structuredContent"], {"items": [{"title": "a"}]})
        self.assertEqual(json.loads(redacted["content"][0]["text"]), {"ok": True})
        self.assertEqual(redacted["content"][1]["text"], "plain text mentioning owner stays")
        self.assertEqual(redacted["_meta"], result["_meta"])
        self.assertIn("Email", result["structuredContent"]["items"][0], "the original is not mutated")

    def test_manager_redacts_configured_servers_only(self):
        transports = {}

        def factory(**options):
            transports[options.get("url")] = t = FakeTransport(**options)
            t.result = {"structuredContent": {"email": "x@example.org", "n": 1}}
            return t
        manager = MCPManager({"hub": {"url": "http://hub/mcp", "redact_keys": ["email"]},
                              "other": {"url": "http://other/mcp"}}, transport_factory=factory)
        self.assertNotIn("redact_keys", transports["http://hub/mcp"].kwargs, "not passed to the transport")
        role = SimpleNamespace(name="SE", capabilities={"mcp"})
        self.assertEqual(manager.rpc("hub", "tools/call", {"name": "t", "arguments": {}}, role)["structuredContent"], {"n": 1})
        self.assertIn("email", manager.rpc("other", "tools/call", {"name": "t", "arguments": {}}, role)["structuredContent"])

    def test_redact_keys_validated(self):
        with self.assertRaises(ValueError):
            MCPManager({"hub": {"url": "http://hub/mcp", "redact_keys": "email"}}, transport_factory=FakeTransport)


class LifecycleHookTests(IdentityTestCase):
    def setUp(self):
        super().setUp()
        self.stream = RuntimeEventStream()
        self.hr = SimpleNamespace(name="HR", runtime_event_stream=self.stream, runtime_session_id="s1")
        scope = patch.multiple(config, memory_store=None, active_tool_caller=self.hr, mcp_manager=None,
                               mcp_config_path="")
        scope.start()
        self.addCleanup(scope.stop)
        self.org = main.build_employee_dict()
        self.log = io.StringIO()
        set_logger(JsonlLogger(self.log))
        self.addCleanup(set_logger, None)

    def events(self, kind):
        return [e.payload for e in self.stream.events() if e.event_type == kind]

    def test_create_announces_the_persona(self):
        result = create_lifecycle_role(self.org, "QA", "Tests things", requested_by="HR")
        self.assertIn("Created", result)
        self.assertEqual([(e["persona"], e["requested_by"]) for e in self.events("persona.created")], [("QA", "HR")])
        self.assertIn('"type": "persona_created"', self.log.getvalue())

    def test_retire_revokes_and_deletes_persona_credentials_once(self):
        create_lifecycle_role(self.org, "QA", "Tests things")
        self.login_as("persona:QA")
        self.login_as("persona:alice")
        qa_refresh = json.loads(oauth.CredentialStore().path("persona:QA", "hub").read_text())["refresh_token"]
        config.mcp_manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth}})
        self.assertIn("retired", retire_lifecycle_role(self.org, "QA", requested_by="HR", reason="project ended"))
        retired = self.events("persona.retired")
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["credentials_removed"], ["hub"])
        self.assertEqual(retired[0]["reason"], "project ended")
        self.assertEqual(self.world.revoked, [qa_refresh])
        self.assertEqual(oauth.CredentialStore().identities("hub"), ["persona:alice"], "other personas untouched")
        archive_lifecycle_role(self.org, "QA", requested_by="HR")
        self.assertEqual(len(self.events("persona.retired")), 1, "archiving a retired persona doesn't repeat it")
        for secret in self.world.issued:
            self.assertNotIn(secret, self.log.getvalue())
            self.assertNotIn(secret, json.dumps(retired))

    def test_cleanup_failure_never_blocks_retirement(self):
        create_lifecycle_role(self.org, "QA", "Tests things")
        broken = SimpleNamespace(forget_identity=lambda identity: (_ for _ in ()).throw(OSError("disk")))
        config.mcp_manager = broken
        self.assertIn("retired", retire_lifecycle_role(self.org, "QA"))
        self.assertEqual(self.events("persona.retired")[0]["credential_cleanup_error"], "OSError")
        self.assertEqual(self.org["QA"].lifecycle_state, "retired")

    def test_no_mcp_config_means_no_manager_is_created(self):
        create_lifecycle_role(self.org, "QA", "Tests things")
        retire_lifecycle_role(self.org, "QA")
        self.assertIsNone(config.mcp_manager)
        self.assertEqual(self.events("persona.retired")[0]["credentials_removed"], [])

    def test_service_identities_are_not_touched(self):
        service_auth = {"type": "oauth", "registration": "dynamic", "identity": "service:desk"}
        self.login_as("service:desk", service_auth, server="desk")
        config.mcp_manager = MCPManager({"desk": {"url": self.url, "auth": service_auth}})
        create_lifecycle_role(self.org, "QA", "Tests things")
        retire_lifecycle_role(self.org, "QA")
        self.assertEqual(oauth.CredentialStore().identities("desk"), ["service:desk"])
        self.assertEqual(self.world.revoked, [])


if __name__ == "__main__":
    unittest.main()

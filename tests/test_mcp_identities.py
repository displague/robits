"""Per-persona and service identities for OAuth MCP servers (#138 part 2)."""
import io
import json
import os
import stat
import unittest
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from robits import auth as auth_cli
from robits.core.io import JsonlLogger, set_logger
from robits.core.session import RuntimeEventStream
from robits.core.tools import ToolRegistry
from robits.runtime import oauth
from robits.runtime.mcp import MCPManager, registry_tool_name
from tests.test_mcp_oauth import OAuthTestCase

SCHEMA = {"type": "object", "properties": {}}


def persona(name, parent=None):
    return SimpleNamespace(name=name, runtime_role_name=parent, capabilities={"mcp"},
                           runtime_event_stream=RuntimeEventStream(), runtime_session_id="s1")


class IdentityTestCase(OAuthTestCase):
    def setUp(self):
        super().setUp()
        self.persona_auth = {"type": "oauth", "registration": "dynamic", "identity": "persona"}

    def login_as(self, identity, auth=None, server="hub"):
        client = oauth.OAuthClient(server, self.url, {**(auth or self.persona_auth), "identity": identity})
        return self.login(client)

    def token_of(self, identity):
        return json.loads(oauth.CredentialStore().path(identity, "hub").read_text())["access_token"]


class PersonaIdentityTests(IdentityTestCase):
    def test_two_personas_carry_their_own_tokens(self):
        self.login_as("persona:alice")
        self.login_as("persona:bob")
        manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth}})
        manager.rpc("hub", "tools/call", {"name": "echo", "arguments": {}}, persona("alice"))
        manager.rpc("hub", "tools/call", {"name": "echo", "arguments": {}}, persona("bob"))
        self.assertEqual(self.world.mcp_tokens, [self.token_of("persona:alice"), self.token_of("persona:bob")])
        self.assertNotEqual(self.token_of("persona:alice"), self.token_of("persona:bob"))

    def test_subtask_acts_as_its_parent_persona(self):
        self.login_as("persona:alice")
        manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth}})
        manager.rpc("hub", "tools/call", {"name": "echo", "arguments": {}}, persona("subtask-7", parent="alice"))
        self.assertEqual(self.world.mcp_tokens, [self.token_of("persona:alice")])

    def test_credentials_are_laid_out_per_kind_and_private(self):
        self.login_as("persona:alex_chen")
        path = oauth.CredentialStore().path("persona:alex_chen", "hub")
        self.assertEqual(path, self.creds / "persona" / "alex_chen" / "hub.json")
        for directory in (self.creds, self.creds / "persona", self.creds / "persona" / "alex_chen"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700, directory)
        self.assertEqual(oauth.CredentialStore().identities("hub"), ["persona:alex_chen"])
        for bad in ("persona", "persona:", "persona:../x", "team", "service:a/b"):
            with self.subTest(bad=bad), self.assertRaises(oauth.OAuthError):
                oauth.identity_parts(bad)

    def test_persona_not_logged_in_skips_the_server_without_failing(self):
        self.login_as("persona:alice")
        registry = ToolRegistry()
        manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth}})
        bob = persona("bob")
        manager.register(registry, bob)   # must not raise
        self.assertNotIn(registry_tool_name("hub", "echo"), registry)
        relogin = [e.payload for e in bob.runtime_event_stream.events() if e.event_type == "auth.relogin_required"]
        self.assertEqual(relogin, [{"server": "hub", "identity": "persona:bob"}])
        manager.register(registry, persona("alice"))
        self.assertIn(registry_tool_name("hub", "echo"), registry)
        with self.assertRaisesRegex(oauth.ReloginRequired, "persona:bob"):
            manager.call("hub", "echo", {}, bob)


class DiscoveryTests(IdentityTestCase):
    def test_each_identity_is_offered_only_what_it_listed(self):
        self.login_as("persona:alice")
        self.login_as("persona:bob")
        alice_token = self.token_of("persona:alice")
        self.world.tools_for = lambda token: (
            [{"name": "echo", "inputSchema": SCHEMA}, {"name": "publish", "inputSchema": SCHEMA}]
            if token == alice_token else [{"name": "echo", "inputSchema": SCHEMA}])
        registry = ToolRegistry()
        manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth}})
        alice, bob = persona("alice"), persona("bob")
        manager.register(registry, alice)
        manager.register(registry, bob)   # same server again: no collision error
        self.assertIn(registry_tool_name("hub", "publish"), registry)
        self.assertEqual(manager.identity_tools[("hub", "persona:bob")], {"echo"})
        manager.call("hub", "publish", {}, alice)
        with self.assertRaisesRegex(PermissionError, "not offered to persona:bob"):
            manager.call("hub", "publish", {}, bob)


class ServiceIdentityTests(IdentityTestCase):
    def test_service_tools_limit_what_runs_under_the_shared_identity(self):
        service_auth = {"type": "oauth", "registration": "dynamic", "identity": "service:desk",
                        "service_tools": ["post_update"]}
        self.login_as("service:desk", service_auth, server="desk")
        self.world.tools_for = lambda token: [{"name": "post_update", "inputSchema": SCHEMA},
                                              {"name": "delete_everything", "inputSchema": SCHEMA}]
        registry = ToolRegistry()
        manager = MCPManager({"desk": {"url": self.url, "auth": service_auth}})
        manager.register(registry, persona("alice"))
        self.assertIn(registry_tool_name("desk", "post_update"), registry)
        self.assertNotIn(registry_tool_name("desk", "delete_everything"), registry)
        manager.call("desk", "post_update", {}, persona("alice"))
        manager.call("desk", "post_update", {}, persona("bob"))
        token = json.loads(oauth.CredentialStore().path("service:desk", "desk").read_text())["access_token"]
        self.assertEqual(self.world.mcp_tokens, [token] * 3, "discovery and both personas use the service identity")
        with self.assertRaisesRegex(PermissionError, "not one of the tools allowed under service:desk"):
            manager.call("desk", "delete_everything", {}, persona("alice"))

    def test_service_tools_need_a_service_identity(self):
        with self.assertRaises(ValueError):
            MCPManager({"hub": {"url": self.url, "auth": {**self.persona_auth, "service_tools": ["echo"]}}})

    def test_persona_and_service_entries_can_share_one_url(self):
        self.login_as("persona:alice")
        service_auth = {"type": "oauth", "registration": "dynamic", "identity": "service:desk",
                        "service_tools": ["echo"]}
        oauth.OAuthClient("desk", self.url, service_auth)  # separate server key, separate files
        self.login(oauth.OAuthClient("desk", self.url, service_auth))
        manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth},
                              "desk": {"url": self.url, "auth": service_auth}})
        registry = ToolRegistry()
        manager.register(registry, persona("alice"))
        self.assertIn(registry_tool_name("hub", "echo"), registry)
        self.assertIn(registry_tool_name("desk", "echo"), registry)


class IdentityLoggingTests(IdentityTestCase):
    def test_every_call_logs_its_identity_and_no_token(self):
        self.login_as("persona:alice")
        log = io.StringIO()
        set_logger(JsonlLogger(log))
        self.addCleanup(set_logger, None)
        alice = persona("alice")
        manager = MCPManager({"hub": {"url": self.url, "auth": self.persona_auth}})
        manager.call("hub", "echo", {}, alice)
        events = [e.payload for e in alice.runtime_event_stream.events() if e.event_type == "mcp.call"]
        self.assertEqual(events, [{"server": "hub", "method": "tools/call", "tool": "echo",
                                   "identity": "persona:alice", "agent": "alice"}])
        logged = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertTrue(any(e.get("type") == "mcp_call" and e.get("identity") == "persona:alice" for e in logged))
        for secret in self.world.issued:
            self.assertNotIn(secret, log.getvalue())
            self.assertNotIn(secret, json.dumps(events))


class CliIdentityTests(IdentityTestCase):
    def config(self):
        path = self.creds.parent / "mcp.json"
        path.write_text(json.dumps({"hub": {"url": self.url, "auth": self.persona_auth}}))
        return str(path)

    def paste_login(self, identity):
        out = io.StringIO()
        world = self.world

        class Browser:
            def readline(self):
                url = next(l for l in out.getvalue().splitlines() if l.startswith(world.base + "/as/authorize"))
                _, headers, _ = oauth.http_request("GET", url)
                return headers["Location"] + "\n"
        code = auth_cli.main(["--config", self.config(), "login", "hub", "--as", identity, "--paste"],
                             stdin=Browser(), out=out)
        return code, out.getvalue()

    def test_login_per_persona_and_status_lists_them(self):
        self.assertEqual(self.paste_login("persona:alice")[0], 0)
        self.assertEqual(self.paste_login("persona:bob")[0], 0)
        out = io.StringIO()
        auth_cli.main(["--config", self.config(), "status"], out=out)
        statuses = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([(s["identity"], s["status"]) for s in statuses],
                         [("persona:alice", "ok"), ("persona:bob", "ok")])
        auth_cli.main(["--config", self.config(), "logout", "hub", "--as", "persona:bob"], out=io.StringIO())
        self.assertEqual(oauth.CredentialStore().identities("hub"), ["persona:alice"])

    def test_persona_server_requires_as(self):
        with self.assertRaises(SystemExit):
            auth_cli.main(["--config", self.config(), "login", "hub", "--paste"], stdin=io.StringIO(), out=io.StringIO())


if __name__ == "__main__":
    unittest.main()

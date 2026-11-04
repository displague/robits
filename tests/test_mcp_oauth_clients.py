"""Confidential dynamic clients and one client per identity (servers that treat each client as an agent)."""
import io
import json
import os
import stat
import unittest

from robits import auth as auth_cli
from robits.core.io import JsonlLogger, set_logger
from robits.runtime import oauth
from robits.runtime.mcp import MCPManager
from tests.test_mcp_identities import IdentityTestCase, persona


class ConfidentialClientTests(IdentityTestCase):
    def test_server_without_public_clients_gets_client_secret_post(self):
        self.world.auth_methods = ["client_secret_post", "client_secret_basic"]
        client = self.login()
        registered = next(iter(self.world.clients.values()))
        self.assertEqual(registered["method"], "client_secret_post")
        stored = json.loads(client.path.read_text())
        self.assertEqual(stored["auth_method"], "client_secret_post")
        self.assertEqual(stat.S_IMODE((self.creds / "_clients" / "tracker.json").stat().st_mode), 0o600)
        self.expire(client)
        client.access_token()              # refresh must authenticate too, or the fake server refuses it
        self.assertEqual(self.world.refresh_calls, 1)
        client.logout()                    # and so must revocation
        self.assertEqual(len(self.world.revoked), 1)

    def test_basic_auth_when_configured(self):
        self.world.auth_methods = ["client_secret_basic"]
        client = self.login(self.client(token_endpoint_auth_method="client_secret_basic"))
        self.expire(client)
        client.access_token()
        self.assertEqual(self.world.refresh_calls, 1)

    def test_wrong_secret_is_refused(self):
        self.world.auth_methods = ["client_secret_post"]
        client = self.login()
        path = self.creds / "_clients" / "tracker.json"
        record = json.loads(client.path.read_text())
        record["client_secret"] = "not-the-secret"
        client.store.save(client.path, record)
        self.expire(client)
        with self.assertRaises(oauth.OAuthError):
            client.access_token()

    def test_secret_never_appears_in_status_or_logs(self):
        self.world.auth_methods = ["client_secret_post"]
        self.login()
        log = io.StringIO()
        set_logger(JsonlLogger(log))
        self.addCleanup(set_logger, None)
        out = io.StringIO()
        config = self.creds.parent / "mcp.json"
        config.write_text(json.dumps({"tracker": {"url": self.url, "auth": self.auth}}))
        auth_cli.main(["--config", str(config), "status"], out=out)
        secret = next(iter(self.world.clients.values()))["secret"]
        self.assertNotIn(secret, out.getvalue() + log.getvalue())

    def test_bad_configured_method(self):
        with self.assertRaises(oauth.OAuthError):
            self.login(self.client(token_endpoint_auth_method="private_key_jwt"))


class ClientPerIdentityTests(IdentityTestCase):
    def test_each_persona_registers_its_own_named_client(self):
        auth = {**self.persona_auth, "client_per_identity": True, "client_name": "team"}
        self.login_as("persona:alice", auth)
        self.login_as("persona:bob", auth)
        self.login_as("persona:alice", auth)   # reuses alice's client
        names = sorted(c["name"] for c in self.world.clients.values())
        self.assertEqual(names, ["team (persona alice)", "team (persona bob)"])
        alice = json.loads(oauth.CredentialStore().path("persona:alice", "hub").read_text())
        bob = json.loads(oauth.CredentialStore().path("persona:bob", "hub").read_text())
        self.assertNotEqual(alice["client_id"], bob["client_id"])
        self.assertTrue((self.creds / "_clients" / "hub" / "persona" / "alice.json").exists())
        manager = MCPManager({"hub": {"url": self.url, "auth": auth}})
        manager.rpc("hub", "tools/call", {"name": "echo", "arguments": {}}, persona("bob"))
        self.assertEqual(self.world.mcp_tokens[-1], bob["access_token"])

    def test_client_name_uses_a_conservative_charset(self):
        auth = {**self.persona_auth, "client_per_identity": True, "client_name": "ops: team #1"}
        self.login_as("persona:alice", auth)
        name = next(iter(self.world.clients.values()))["name"]
        self.assertEqual(name, "ops team 1 (persona alice)")
        self.assertRegex(name, r"^[\w ()._-]{2,128}$")

    def test_shared_client_by_default(self):
        self.login_as("persona:alice")
        self.login_as("persona:bob")
        self.assertEqual(self.world.registrations, 1)


if __name__ == "__main__":
    unittest.main()

"""OAuth for HTTP MCP servers (#138 part 1) against in-process fake servers."""
import base64
import hashlib
import io
import json
import os
import secrets
import stat
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

from robits import auth as auth_cli
from robits.core.io import JsonlLogger, set_logger
from robits.core.session import RuntimeEventStream
from robits.runtime import oauth
from robits.runtime.mcp import HttpTransport, MCPManager


class FakeOAuthWorld:
    """An authorization server (/as) and an MCP server (/mcp) on one loopback port."""

    def __init__(self):
        self.lock = threading.Lock()
        self.codes, self.access, self.refresh = {}, {}, {}
        self.issued = []            # every token and code handed out, to check for leaks
        self.authorize_params = []
        self.refresh_calls = 0
        self.registrations = 0
        self.revoked = []
        self.refuse_refresh = False
        self.reject_access_once = set()
        self.mcp_tokens = []
        self.tools_for = lambda token: [{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}]
        world = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body=None, headers=None):
                data = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                if body is not None:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                parts = urlsplit(self.path)
                if parts.path == "/.well-known/oauth-protected-resource/mcp":
                    return self.reply(200, {"resource": world.base + "/mcp", "authorization_servers": [world.base + "/as"]})
                if parts.path == "/.well-known/oauth-authorization-server/as":
                    return self.reply(200, {
                        "issuer": world.base + "/as",
                        "authorization_endpoint": world.base + "/as/authorize",
                        "token_endpoint": world.base + "/as/token",
                        "registration_endpoint": world.base + "/as/register",
                        "revocation_endpoint": world.base + "/as/revoke",
                        "code_challenge_methods_supported": ["S256"]})
                if parts.path == "/as/authorize":
                    params = {k: v[0] for k, v in parse_qs(parts.query).items()}
                    world.authorize_params.append(params)
                    code = world.mint("code")
                    world.codes[code] = params
                    location = params["redirect_uri"] + "?" + urlencode({"code": code, "state": params["state"]})
                    return self.reply(302, headers={"Location": location})
                return self.reply(404, {"error": "not_found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                path = urlsplit(self.path).path
                if path == "/as/register":
                    world.registrations += 1
                    return self.reply(201, {"client_id": f"dyn-{world.registrations}"})
                if path == "/as/revoke":
                    world.revoked.append(parse_qs(raw.decode())["token"][0])
                    return self.reply(200, {})
                if path == "/as/token":
                    return self.reply(*world.token({k: v[0] for k, v in parse_qs(raw.decode()).items()}))
                if path == "/mcp":
                    header = self.headers.get("Authorization", "")
                    token = header[7:] if header.startswith("Bearer ") else None
                    with world.lock:
                        valid = token in world.access and token not in world.reject_access_once
                        world.reject_access_once.discard(token)
                    if not valid:
                        return self.reply(401, {"error": "invalid_token"}, {
                            "WWW-Authenticate": f'Bearer resource_metadata="{world.base}/.well-known/oauth-protected-resource/mcp"'})
                    world.mcp_tokens.append(token)
                    message = json.loads(raw)
                    result = ({"tools": world.tools_for(token)} if message["method"] == "tools/list"
                              else {"content": [{"type": "text", "text": "ok"}]})
                    return self.reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": result})
                return self.reply(404, {"error": "not_found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def mint(self, kind):
        value = f"{kind}-{secrets.token_hex(8)}"
        self.issued.append(value)
        return value

    def tokens(self):
        access, refresh = self.mint("at"), self.mint("rt")
        self.access[access] = True
        self.refresh[refresh] = "live"
        return 200, {"access_token": access, "refresh_token": refresh, "token_type": "Bearer", "expires_in": 3600}

    def token(self, form):
        with self.lock:
            if form.get("resource") != self.base + "/mcp":
                return 400, {"error": "invalid_target"}
            if form.get("grant_type") == "authorization_code":
                params = self.codes.pop(form.get("code"), None)
                if params is None or form.get("redirect_uri") != params["redirect_uri"]:
                    return 400, {"error": "invalid_grant"}
                challenge = base64.urlsafe_b64encode(
                    hashlib.sha256(form.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
                if challenge != params["code_challenge"]:
                    return 400, {"error": "invalid_grant"}
                return self.tokens()
            if form.get("grant_type") == "refresh_token":
                self.refresh_calls += 1
                if self.refuse_refresh or self.refresh.get(form.get("refresh_token")) != "live":
                    return 400, {"error": "invalid_grant"}
                self.refresh[form["refresh_token"]] = "used"   # rotation: single use
                return self.tokens()
        return 400, {"error": "unsupported_grant_type"}


class OAuthTestCase(unittest.TestCase):
    def setUp(self):
        self.world = FakeOAuthWorld()
        self.addCleanup(self.world.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.creds = Path(tmp.name) / "credentials"
        env = patch.dict(os.environ, {"ROBITS_CREDENTIALS_DIR": str(self.creds)})
        env.start()
        self.addCleanup(env.stop)
        self.url = self.world.base + "/mcp"
        self.auth = {"type": "oauth", "registration": "dynamic", "scopes": ["mcp"], "identity": "service:team"}

    def client(self, **overrides):
        return oauth.OAuthClient("tracker", self.url, {**self.auth, **overrides})

    def login(self, client=None):
        client = client or self.client()
        auth_url, pending = client.begin_login("http://127.0.0.1:8765/callback")
        status, headers, _ = oauth.http_request("GET", auth_url)
        self.assertEqual(status, 302)
        query = parse_qs(urlsplit(headers["Location"]).query)
        client.complete_login(pending, query["code"][0], query["state"][0])
        return client

    def expire(self, client):
        record = json.loads(client.path.read_text())
        record["expires_at"] = time.time() - 1
        client.store.save(client.path, record)


class LoginTests(OAuthTestCase):
    def test_login_uses_pkce_s256_and_resource(self):
        client = self.login()
        params = self.world.authorize_params[0]
        self.assertEqual(params["code_challenge_method"], "S256")
        self.assertEqual(params["resource"], self.url)
        self.assertEqual(params["scope"], "mcp")
        self.assertEqual(params["client_id"], "dyn-1")
        self.assertEqual(client.status()["status"], "ok")
        self.assertEqual(stat.S_IMODE(client.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(client.path.parent.stat().st_mode), 0o700)
        self.assertEqual(client.path, self.creds / "service" / "team" / "tracker.json")

    def test_wrong_verifier_or_state_is_refused(self):
        client = self.client()
        auth_url, pending = client.begin_login("http://127.0.0.1:8765/callback")
        _, headers, _ = oauth.http_request("GET", auth_url)
        query = parse_qs(urlsplit(headers["Location"]).query)
        with self.assertRaises(oauth.OAuthError):
            client.complete_login(pending, query["code"][0], "forged-state")
        with self.assertRaisesRegex(oauth.OAuthError, "invalid_grant"):
            client.complete_login({**pending, "verifier": "not-the-verifier"}, query["code"][0], query["state"][0])

    def test_dynamic_client_registered_once_per_server(self):
        self.login()
        self.login()
        self.assertEqual(self.world.registrations, 1)

    def test_metadata_document_client_id_needs_no_registration(self):
        self.login(self.client(registration=None, client_id="https://example.org/robits/client.json"))
        self.assertEqual(self.world.registrations, 0)
        self.assertEqual(self.world.authorize_params[0]["client_id"], "https://example.org/robits/client.json")

    def test_configuration_errors(self):
        for bad in ({"type": "basic"}, {"type": "oauth"}, {"type": "oauth", "client_id": "http://example.org/c.json"},
                    {"type": "oauth", "registration": "dynamic", "identity": "persona"},
                    {"type": "oauth", "registration": "dynamic", "identity": "persona:"},
                    {"type": "oauth", "registration": "dynamic", "identity": "../x"}):
            with self.subTest(bad=bad), self.assertRaises(oauth.OAuthError):
                oauth.OAuthClient("tracker", self.url, bad)
        with self.assertRaises(ValueError):
            MCPManager({"local": {"command": "/bin/true", "auth": self.auth}})


class RefreshTests(OAuthTestCase):
    def test_refresh_rotates_and_the_old_refresh_token_is_rejected(self):
        client = self.login()
        old = json.loads(client.path.read_text())
        self.expire(client)
        token = client.access_token()
        new = json.loads(client.path.read_text())
        self.assertNotEqual(token, old["access_token"])
        self.assertEqual(token, new["access_token"])
        self.assertNotEqual(new["refresh_token"], old["refresh_token"], "rotated refresh token saved")
        status, _, body = oauth.http_request("POST", self.world.base + "/as/token", form={
            "grant_type": "refresh_token", "refresh_token": old["refresh_token"],
            "client_id": "dyn-1", "resource": self.url})
        self.assertEqual((status, json.loads(body)["error"]), (400, "invalid_grant"))

    def test_concurrent_refresh_spends_the_refresh_token_once(self):
        client = self.login()
        self.expire(client)
        clients = [self.client() for _ in range(8)]   # separate instances, shared store
        results, errors = [], []

        def worker(c):
            try:
                results.append(c.access_token())
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(c,)) for c in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.world.refresh_calls, 1)
        self.assertEqual(len(set(results)), 1)

    def test_rejected_refresh_requires_relogin_without_retrying(self):
        client = self.login()
        self.world.refuse_refresh = True
        self.expire(client)
        with self.assertRaisesRegex(oauth.ReloginRequired, "needs a new login for service:team"):
            client.access_token()
        with self.assertRaises(oauth.ReloginRequired):
            client.access_token()
        self.assertEqual(self.world.refresh_calls, 1, "no retry loop after the first refusal")
        status = client.status()
        self.assertEqual(status["status"], "relogin_required")
        self.assertFalse(status["has_refresh_token"])
        self.login(client)
        self.assertEqual(client.status()["status"], "ok", "a new login clears the state")


class TransportTests(OAuthTestCase):
    def transport(self):
        return HttpTransport(self.url, auth={**self.auth, "server": "tracker"})

    def test_requests_carry_the_bearer_token(self):
        self.login()
        result = self.transport().request("tools/list", {}, lambda message: None)
        self.assertEqual(result["tools"][0]["name"], "echo")
        stored = json.loads((self.creds / "service" / "team" / "tracker.json").read_text())
        self.assertEqual(self.world.mcp_tokens, [stored["access_token"]])

    def test_401_forces_one_refresh_and_retry(self):
        client = self.login()
        self.world.reject_access_once.add(json.loads(client.path.read_text())["access_token"])
        result = self.transport().request("tools/call", {"name": "echo", "arguments": {}}, lambda m: None)
        self.assertEqual(result["content"][0]["text"], "ok")
        self.assertEqual(self.world.refresh_calls, 1)

    def test_not_logged_in_is_a_relogin_state(self):
        with self.assertRaisesRegex(oauth.ReloginRequired, "python -m robits.auth login tracker"):
            self.transport().request("tools/list", {}, lambda m: None)

    def test_relogin_event_and_no_secrets_anywhere(self):
        self.login()
        log = io.StringIO()
        set_logger(JsonlLogger(log))
        self.addCleanup(set_logger, None)
        stream = RuntimeEventStream()
        role = SimpleNamespace(name="SE", runtime_event_stream=stream, runtime_session_id="s1",
                               capabilities={"mcp"})
        manager = MCPManager({"tracker": {"url": self.url, "auth": self.auth}})
        manager.rpc("tracker", "tools/list", {}, role)
        self.world.refuse_refresh = True
        self.expire(manager.servers["tracker"].oauth)
        with self.assertRaises(oauth.ReloginRequired) as caught:
            manager.rpc("tracker", "tools/list", {}, role)
        events = [e for e in stream.events() if e.event_type == "auth.relogin_required"]
        self.assertEqual([e.payload for e in events], [{"server": "tracker", "identity": "service:team"}])
        out = io.StringIO()
        auth_cli.main(["--config", self.config_file(), "status"], out=out)
        surfaces = "\n".join([log.getvalue(), json.dumps([e.payload for e in stream.events()]),
                              str(caught.exception), out.getvalue()])
        for secret in self.world.issued:
            self.assertNotIn(secret, surfaces)

    def config_file(self):
        path = self.creds.parent / "mcp.json"
        path.write_text(json.dumps({"tracker": {"url": self.url, "auth": self.auth}}))
        return str(path)


class CliTests(OAuthTestCase):
    def test_paste_login_and_logout(self):
        config = self.creds.parent / "mcp.json"
        config.write_text(json.dumps({"tracker": {"url": self.url, "auth": self.auth}}))
        out = io.StringIO()
        world = self.world

        class Browser:
            """Reads the printed URL, 'signs in', and pastes back where the browser was sent."""
            def readline(self):
                url = next(line for line in out.getvalue().splitlines() if line.startswith(world.base + "/as/authorize"))
                _, headers, _ = oauth.http_request("GET", url)
                return headers["Location"] + "\n"

        self.assertEqual(auth_cli.main(["--config", str(config), "login", "tracker", "--paste"],
                                       stdin=Browser(), out=out), 0)
        self.assertIn('"status": "ok"', out.getvalue())
        self.assertTrue((self.creds / "service" / "team" / "tracker.json").exists())
        refresh_token = json.loads((self.creds / "service" / "team" / "tracker.json").read_text())["refresh_token"]
        self.assertEqual(auth_cli.main(["--config", str(config), "logout", "tracker"], out=io.StringIO()), 0)
        self.assertEqual(self.world.revoked, [refresh_token])
        self.assertFalse((self.creds / "service" / "team" / "tracker.json").exists())

    def test_loopback_login(self):
        client = self.client()
        out = io.StringIO()

        def browser():
            for _ in range(100):
                line = next((l for l in out.getvalue().splitlines() if l.startswith(self.world.base + "/as/authorize")), None)
                if line:
                    _, headers, _ = oauth.http_request("GET", line)
                    oauth.http_request("GET", headers["Location"])
                    return
                time.sleep(0.05)
        threading.Thread(target=browser, daemon=True).start()
        auth_cli.login(client, self.url, open_browser=False, out=out, timeout=10)
        self.assertEqual(client.status()["status"], "ok")


if __name__ == "__main__":
    unittest.main()

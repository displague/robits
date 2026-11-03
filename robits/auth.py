"""Operator CLI for MCP server OAuth logins (#138). Models never start a login.

    python -m robits.auth login <server> [--as IDENTITY] [--paste] [--port N] [--no-browser]
    python -m robits.auth status [<server>]
    python -m robits.auth logout <server> [--as IDENTITY]

For a server with "identity": "persona", each persona logs in separately:
``--as persona:<username>``. Otherwise the configured identity is used.

Servers and their ``auth`` blocks come from ROBITS_MCP_CONFIG (or --config).
``login`` opens the authorization URL and waits for the loopback redirect on
127.0.0.1. ``--paste`` is for hosts without a browser: open the printed URL
anywhere, then paste the address the browser was redirected to (the page itself
will fail to load; that is expected). Nothing printed contains a token.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from robits.runtime.oauth import CredentialStore, OAuthClient, OAuthError, http_request

_DONE_PAGE = b"<!doctype html><title>robits</title><p>Login complete. You can close this tab.</p>"


def load_servers(path):
    if not path:
        raise SystemExit("Set ROBITS_MCP_CONFIG or pass --config")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def client_for(servers, name, store=None, identity=None):
    options = servers.get(name)
    if not isinstance(options, dict) or "url" not in options:
        raise SystemExit(f"No HTTP MCP server named {name!r} in the config")
    if not isinstance(options.get("auth"), dict):
        raise SystemExit(f"Server {name!r} has no auth block")
    auth = dict(options["auth"])
    mode = str(auth.get("identity") or "default")
    if identity:
        if mode == "persona" and not identity.startswith("persona:"):
            raise SystemExit(f"{name} acts as each persona: use --as persona:<username>")
        if mode != "persona" and identity != mode:
            raise SystemExit(f"{name} always uses {mode}; --as {identity} doesn't apply")
        auth["identity"] = identity
    elif mode == "persona":
        raise SystemExit(f"{name} acts as each persona: pass --as persona:<username>")
    return OAuthClient(name, options["url"], auth, store=store)


def probe(url):
    """Ask the MCP server for a challenge; return its WWW-Authenticate header, if any."""
    status, headers, _ = http_request("POST", url, json_body={
        "jsonrpc": "2.0", "id": "robits-auth-probe", "method": "tools/list", "params": {}},
        headers={"Accept": "application/json, text/event-stream"})
    if status == 401:
        return next((v for k, v in headers.items() if k.lower() == "www-authenticate"), None)
    return None


def _code_from_url(text):
    query = parse_qs(urlsplit(text.strip()).query)
    if "error" in query:
        raise OAuthError(f"Authorization refused: {query['error'][0]}")
    return query.get("code", [None])[0], query.get("state", [None])[0]


def login(client, url, paste=False, port=0, open_browser=True, stdin=sys.stdin, out=sys.stdout, timeout=300):
    challenge = probe(url)
    if paste:
        redirect_uri = f"http://127.0.0.1:{port or 8765}/callback"
        auth_url, pending = client.begin_login(redirect_uri, challenge)
        print("Open this URL in a browser and sign in:\n\n" + auth_url, file=out)
        print("\nThen paste the full address the browser was sent to:", file=out)
        code, state = _code_from_url(stdin.readline())
        client.complete_login(pending, code, state)
        return
    result = {}

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path != "/callback":
                self.send_error(404)
                return
            try:
                result["code"], result["state"] = _code_from_url(self.path)
            except OAuthError as error:
                result["error"] = str(error)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_DONE_PAGE)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Callback)
    try:
        redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
        auth_url, pending = client.begin_login(redirect_uri, challenge)
        print("Sign in at:\n\n" + auth_url + "\n\nWaiting for the browser to come back...", file=out)
        if open_browser:
            import webbrowser
            threading.Thread(target=webbrowser.open, args=(auth_url,), daemon=True).start()
        server.timeout = 1
        deadline = time.monotonic() + timeout
        while not result and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if "error" in result:
        raise OAuthError(result["error"])
    if not result:
        raise OAuthError("Timed out waiting for the login redirect")
    client.complete_login(pending, result["code"], result["state"])


def main(argv=None, stdin=sys.stdin, out=sys.stdout):
    parser = argparse.ArgumentParser(prog="python -m robits.auth", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.environ.get("ROBITS_MCP_CONFIG", ""),
                        help="MCP server config (default: ROBITS_MCP_CONFIG)")
    commands = parser.add_subparsers(dest="command", required=True)
    login_cmd = commands.add_parser("login", help="sign in for one server")
    login_cmd.add_argument("server")
    login_cmd.add_argument("--paste", action="store_true", help="paste the redirect address instead of a loopback listener")
    login_cmd.add_argument("--port", type=int, default=0, help="loopback port (default: random; 8765 with --paste)")
    login_cmd.add_argument("--no-browser", action="store_true")
    login_cmd.add_argument("--as", dest="identity", help="identity to log in as, e.g. persona:alex_chen")
    status_cmd = commands.add_parser("status", help="show stored logins (no secrets)")
    status_cmd.add_argument("server", nargs="?")
    logout_cmd = commands.add_parser("logout", help="revoke and delete a stored login")
    logout_cmd.add_argument("server")
    logout_cmd.add_argument("--as", dest="identity", help="identity to log out, e.g. persona:alex_chen")
    args = parser.parse_args(argv)
    servers = load_servers(args.config)
    store = CredentialStore()
    try:
        if args.command == "login":
            client = client_for(servers, args.server, store, args.identity)
            login(client, servers[args.server]["url"], paste=args.paste, port=args.port,
                  open_browser=not args.no_browser, stdin=stdin, out=out)
            print(json.dumps(client.status(), sort_keys=True), file=out)
        elif args.command == "status":
            names = [args.server] if args.server else [n for n, o in servers.items()
                                                        if isinstance(o, dict) and isinstance(o.get("auth"), dict)]
            for name in names:
                mode = str(servers[name]["auth"].get("identity") or "default")
                identities = store.identities(name) if mode == "persona" else [mode]
                if not identities:
                    print(json.dumps({"server": name, "identity": "persona:*", "status": "not_logged_in"}), file=out)
                for identity in identities:
                    print(json.dumps(client_for(servers, name, store, identity if mode == "persona" else None).status(),
                                     sort_keys=True), file=out)
        else:
            client = client_for(servers, args.server, store, args.identity)
            client.logout()
            print(f"Logged out of {args.server} ({client.identity}).", file=out)
    except OAuthError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

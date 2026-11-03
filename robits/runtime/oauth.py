"""OAuth 2.1 for HTTP MCP servers (#138, part 1).

Discovery follows the MCP authorization flow: protected resource metadata
(RFC 9728), then authorization server metadata (RFC 8414). Logins use the
authorization code flow with PKCE (S256), and every authorization and token
request carries ``resource`` (RFC 8707) so tokens are audience-bound to the MCP
server. The client is identified by a Client ID Metadata Document URL, or is
registered dynamically (RFC 7591) once per server.

Tokens live in a credential store outside every agent workspace: one file per
identity and server, mode 0600, written atomically. Refreshes run under an
exclusive file lock, so a rotating refresh token is spent by one writer only. A
refresh the server rejects puts the identity into ``relogin_required``; nothing
is retried until an operator logs in again (``python -m robits.auth login``).

Tokens never appear in exceptions, logs, events or model context.
"""
import base64
import fcntl
import hashlib
import json
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

REFRESH_MARGIN_SECONDS = 60
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,63}$")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_thread_locks = {}
_thread_locks_guard = threading.Lock()


class OAuthError(Exception):
    """An OAuth step failed. Messages never contain tokens or codes."""


class ReloginRequired(OAuthError):
    def __init__(self, server, identity):
        super().__init__(f"{server} needs a new login for {identity} (operator action: "
                         f"python -m robits.auth login {server})")
        self.server, self.identity = server, identity


def _require_url(url, what):
    parts = urlsplit(str(url or ""))
    if parts.scheme == "https" and parts.hostname:
        return url
    if parts.scheme == "http" and parts.hostname in _LOOPBACK:
        return url
    raise OAuthError(f"{what} must be an https URL (http only for loopback)")


def http_request(method, url, *, form=None, json_body=None, headers=None, timeout=15):
    """One request, no redirects, 1 MiB cap. Returns (status, headers, body bytes)."""
    import urllib.error
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    data, sent = None, {"Accept": "application/json", **(headers or {})}
    if form is not None:
        data = urlencode(form).encode()
        sent["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        data = json.dumps(json_body).encode()
        sent["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=sent, method=method)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read(1 << 20)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read(1 << 20)
    except OSError as error:
        raise OAuthError(f"{method} {urlsplit(url).netloc} failed: {error.__class__.__name__}") from None


def _json(status, body, what):
    try:
        value = json.loads(body or b"{}")
    except ValueError:
        raise OAuthError(f"{what}: response is not JSON (HTTP {status})") from None
    if not isinstance(value, dict):
        raise OAuthError(f"{what}: response is not an object (HTTP {status})")
    return value


def resource_metadata_url(www_authenticate):
    """Return resource_metadata from a WWW-Authenticate header, if present."""
    match = re.search(r'resource_metadata="([^"]+)"', www_authenticate or "")
    return match.group(1) if match else None


def _well_known(url, name):
    """RFC 8414/9728 path insertion: https://h/p -> https://h/.well-known/<name>/p."""
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, f"/.well-known/{name}{path}", "", ""))


def discover(server_url, www_authenticate=None):
    """Find the authorization server for an MCP endpoint; return its metadata plus resource."""
    candidates = [resource_metadata_url(www_authenticate)] if www_authenticate else []
    candidates += [_well_known(server_url, "oauth-protected-resource"),
                   _well_known(urlunsplit(urlsplit(server_url)[:2] + ("", "", "")), "oauth-protected-resource")]
    prm = None
    for candidate in [c for c in candidates if c]:
        _require_url(candidate, "Resource metadata URL")
        status, _, body = http_request("GET", candidate)
        if status == 200:
            prm = _json(status, body, "Protected resource metadata")
            break
    if prm is None or not prm.get("authorization_servers"):
        raise OAuthError("MCP server publishes no protected resource metadata with authorization_servers")
    resource = prm.get("resource") or server_url
    if urlsplit(resource)[:2] != urlsplit(server_url)[:2]:
        raise OAuthError("Protected resource metadata names a different origin than the MCP server")
    issuer = _require_url(prm["authorization_servers"][0], "Authorization server")
    metadata = None
    for candidate in (_well_known(issuer, "oauth-authorization-server"),
                      _well_known(issuer, "openid-configuration"),
                      issuer.rstrip("/") + "/.well-known/openid-configuration"):
        status, _, body = http_request("GET", candidate)
        if status == 200:
            metadata = _json(status, body, "Authorization server metadata")
            break
    if metadata is None:
        raise OAuthError("Authorization server metadata not found")
    for key in ("authorization_endpoint", "token_endpoint"):
        _require_url(metadata.get(key), key)
    methods = metadata.get("code_challenge_methods_supported")
    if methods is not None and "S256" not in methods:
        raise OAuthError("Authorization server does not support PKCE S256")
    return {
        "issuer": metadata.get("issuer", issuer),
        "authorization_endpoint": metadata["authorization_endpoint"],
        "token_endpoint": metadata["token_endpoint"],
        "registration_endpoint": metadata.get("registration_endpoint"),
        "revocation_endpoint": metadata.get("revocation_endpoint"),
        "resource": resource,
    }


class CredentialStore:
    """One JSON file per identity and server, mode 0600, outside agent workspaces."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("ROBITS_CREDENTIALS_DIR", "").strip() or "var/credentials")

    def path(self, identity, server):
        for value, what in ((identity, "identity"), (server, "server")):
            if not _NAME.match(str(value)):
                raise OAuthError(f"Invalid {what} name {value!r}")
        return self.root / identity.replace(":", "_") / f"{server}.json"

    def client_path(self, server):
        """Dynamic registrations, one per server, shared by every identity."""
        self.path("default", server)  # validates the server name
        return self.root / "_clients" / f"{server}.json"

    def load(self, path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def delete(self, path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @contextmanager
    def lock(self, path):
        """Exclusive across threads and processes for one credential file."""
        with _thread_locks_guard:
            thread_lock = _thread_locks.setdefault(str(path), threading.Lock())
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with thread_lock:
            fd = os.open(path.with_name(f".{path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


def _pkce():
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


class OAuthClient:
    """Tokens for one identity on one MCP server."""

    def __init__(self, server, server_url, auth, store=None):
        if not isinstance(auth, dict) or auth.get("type") != "oauth":
            raise OAuthError('auth must be {"type": "oauth", ...}')
        self.server, self.server_url, self.auth = server, server_url, auth
        self.identity = str(auth.get("identity") or "default")
        if self.identity == "persona":
            raise OAuthError('"identity": "persona" is not supported yet; name one identity such as "service:team"')
        self.store = store or CredentialStore()
        self.path = self.store.path(self.identity, server)
        scopes = auth.get("scopes") or []
        if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
            raise OAuthError("auth.scopes must be a list of strings")
        self.scope = " ".join(scopes)
        if auth.get("client_id"):
            _require_url(auth["client_id"], "auth.client_id (a Client ID Metadata Document URL)")
        elif auth.get("registration") != "dynamic":
            raise OAuthError('auth needs client_id (a metadata document URL) or "registration": "dynamic"')

    # --- client identification -------------------------------------------------
    def client_id(self, metadata, redirect_uri):
        if self.auth.get("client_id"):
            return self.auth["client_id"]
        path = self.store.client_path(self.server)
        with self.store.lock(path):
            client = self.store.load(path)
            if client and client.get("issuer") == metadata["issuer"]:
                return client["client_id"]
            endpoint = metadata.get("registration_endpoint")
            if not endpoint:
                raise OAuthError("Authorization server offers no dynamic client registration")
            status, _, body = http_request("POST", _require_url(endpoint, "registration_endpoint"), json_body={
                "client_name": "robits",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            })
            data = _json(status, body, "Client registration")
            if status not in (200, 201) or not data.get("client_id"):
                raise OAuthError(f"Client registration failed (HTTP {status})")
            self.store.save(path, {"issuer": metadata["issuer"], "client_id": data["client_id"]})
            return data["client_id"]

    # --- login ------------------------------------------------------------------
    def begin_login(self, redirect_uri, www_authenticate=None):
        """Return (authorization URL, pending login state) for an operator to complete."""
        metadata = discover(self.server_url, www_authenticate)
        verifier, challenge = _pkce()
        state = secrets.token_urlsafe(24)
        client_id = self.client_id(metadata, redirect_uri)
        params = {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
                  "state": state, "code_challenge": challenge, "code_challenge_method": "S256",
                  "resource": metadata["resource"]}
        if self.scope:
            params["scope"] = self.scope
        separator = "&" if "?" in metadata["authorization_endpoint"] else "?"
        pending = {"metadata": metadata, "client_id": client_id, "verifier": verifier,
                   "state": state, "redirect_uri": redirect_uri}
        return metadata["authorization_endpoint"] + separator + urlencode(params), pending

    def complete_login(self, pending, code, state):
        if not secrets.compare_digest(str(state or ""), pending["state"]):
            raise OAuthError("Login state mismatch; start the login again")
        metadata = pending["metadata"]
        status, _, body = http_request("POST", metadata["token_endpoint"], form={
            "grant_type": "authorization_code", "code": code, "redirect_uri": pending["redirect_uri"],
            "client_id": pending["client_id"], "code_verifier": pending["verifier"],
            "resource": metadata["resource"]})
        tokens = _json(status, body, "Token request")
        if status != 200 or not tokens.get("access_token"):
            raise OAuthError(f"Token request failed: {tokens.get('error', f'HTTP {status}')}")
        with self.store.lock(self.path):
            self.store.save(self.path, self._record(metadata, pending["client_id"], tokens, None))

    def _record(self, metadata, client_id, tokens, previous):
        refresh = tokens.get("refresh_token") or (previous or {}).get("refresh_token")
        expires_in = tokens.get("expires_in")
        return {
            "status": "ok", "server": self.server, "identity": self.identity,
            "issuer": metadata["issuer"], "token_endpoint": metadata["token_endpoint"],
            "revocation_endpoint": metadata.get("revocation_endpoint"),
            "resource": metadata["resource"], "client_id": client_id,
            "access_token": tokens["access_token"], "refresh_token": refresh,
            "expires_at": time.time() + float(expires_in) if expires_in else None,
            "scope": tokens.get("scope", self.scope), "updated_at": time.time(),
        }

    # --- use --------------------------------------------------------------------
    def access_token(self, force_refresh=False, stale=None):
        """Return a usable access token, refreshing under the lock when needed.

        ``stale`` is the token a request was just refused with: when another
        writer has already replaced it, the new one is used without refreshing.
        """
        with self.store.lock(self.path):
            record = self.store.load(self.path)
            if not record:
                raise ReloginRequired(self.server, self.identity)
            if record.get("status") == "relogin_required":
                raise ReloginRequired(self.server, self.identity)
            if stale is not None and record["access_token"] != stale:
                return record["access_token"]
            expires_at = record.get("expires_at")
            fresh = expires_at is None or expires_at - time.time() > REFRESH_MARGIN_SECONDS
            if fresh and not force_refresh:
                return record["access_token"]
            return self._refresh(record)

    def _refresh(self, record):
        if not record.get("refresh_token"):
            self._mark_relogin(record)
        status, _, body = http_request("POST", record["token_endpoint"], form={
            "grant_type": "refresh_token", "refresh_token": record["refresh_token"],
            "client_id": record["client_id"], "resource": record["resource"]})
        try:
            tokens = _json(status, body, "Token refresh")
        except OAuthError:
            tokens = {}
        if status == 200 and tokens.get("access_token"):
            metadata = {k: record.get(k) for k in ("issuer", "token_endpoint", "revocation_endpoint", "resource")}
            updated = self._record(metadata, record["client_id"], tokens, record)
            # Persist a rotated refresh token before anyone uses the new access token.
            self.store.save(self.path, updated)
            return updated["access_token"]
        if status in (400, 401) and tokens.get("error") in {"invalid_grant", "invalid_client", "unauthorized_client"}:
            self._mark_relogin(record)
        raise OAuthError(f"Token refresh failed: {tokens.get('error', f'HTTP {status}')}")

    def _mark_relogin(self, record):
        record = {**record, "status": "relogin_required", "access_token": None, "refresh_token": None,
                  "updated_at": time.time()}
        self.store.save(self.path, record)
        raise ReloginRequired(self.server, self.identity)

    def mark_relogin(self):
        with self.store.lock(self.path):
            record = self.store.load(self.path) or {}
            try:
                self._mark_relogin(record)
            except ReloginRequired:
                pass

    def status(self):
        """Non-secret summary of the stored credential."""
        record = self.store.load(self.path)
        if not record:
            return {"server": self.server, "identity": self.identity, "status": "not_logged_in"}
        return {"server": self.server, "identity": self.identity, "status": record.get("status"),
                "issuer": record.get("issuer"), "scope": record.get("scope"),
                "expires_at": record.get("expires_at"), "updated_at": record.get("updated_at"),
                "has_refresh_token": bool(record.get("refresh_token"))}

    def logout(self):
        """Revoke (when supported) and delete the stored credential."""
        with self.store.lock(self.path):
            record = self.store.load(self.path)
            if record and record.get("revocation_endpoint") and record.get("refresh_token"):
                try:
                    http_request("POST", _require_url(record["revocation_endpoint"], "revocation_endpoint"), form={
                        "token": record["refresh_token"], "token_type_hint": "refresh_token",
                        "client_id": record["client_id"]})
                except OAuthError:
                    pass
            self.store.delete(self.path)

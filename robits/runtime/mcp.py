"""MCP 2026-07-28 HTTP client and native function-tool adapter.

Server addresses and credentials come only from operator-owned configuration.
Continuations are scoped to a session and caller; opaque requestState is never
interpreted or exposed to the model. No legacy initialize handshake is used.
"""
import hashlib
import json
import re
import time
from pathlib import Path
from uuid import uuid4

from robits.core.config import _config as config

VERSION = "2026-07-28"
TASKS = "io.modelcontextprotocol/tasks"


def validate_json(value, schema):
    """Validate local schemas without fetching remote $ref URLs."""
    from jsonschema.validators import validator_for
    from referencing import Registry
    validator = validator_for(schema)
    validator.check_schema(schema)
    validator(schema, registry=Registry()).validate(value)


def header_value(value):
    import base64
    text = str(value).lower() if isinstance(value, bool) else str(value)
    if (text != text.strip() or any(ord(c) < 32 and c != "\t" or ord(c) > 126 for c in text)
            or text.startswith("=?base64?") and text.endswith("?=")):
        return "=?base64?" + base64.b64encode(text.encode()).decode() + "?="
    return text


def header_paths(schema):
    """Validate x-mcp-header annotations, including nested static property paths."""
    paths, seen = [], set()
    def visit(node, path=(), reachable=True):
        if isinstance(node, list):
            for item in node:
                visit(item, path, False)
        elif isinstance(node, dict):
            if "x-mcp-header" in node:
                name = node["x-mcp-header"]
                if (not reachable or not path or not isinstance(name, str)
                        or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                        or name.lower() in seen or node.get("type") not in {"string", "integer", "boolean"}):
                    raise ValueError("Invalid x-mcp-header annotation")
                seen.add(name.lower())
                paths.append((name, path))
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    for prop, child in value.items():
                        visit(child, (*path, prop), reachable)
                elif isinstance(value, (dict, list)):
                    visit(value, path, False)
    visit(schema)
    return paths


class _Unauthorized(Exception):
    def __init__(self, www_authenticate):
        super().__init__("MCP HTTP status 401")
        self.www_authenticate = www_authenticate


class HttpTransport:
    """Bounded POST transport, including in-request SSE progress notifications.

    With ``auth`` ({"type": "oauth", ...}) requests carry an OAuth bearer token
    from robits.runtime.oauth; a 401 forces one refresh and one retry.
    """

    def __init__(self, url, headers=None, timeout=30, auth=None):
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("MCP endpoint must use HTTP or HTTPS")
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError("MCP timeout must be between 0 and 300 seconds")
        self.url, self.headers, self.timeout = url, headers or {}, timeout
        self.tool_headers = {}
        self.auth = None
        self.oauth = None            # the client for a fixed identity (default or service:<name>)
        self.identity_mode = None
        self.service_tools = None
        self._clients = {}
        if auth is not None:
            from robits.runtime.oauth import OAuthClient
            if not isinstance(auth, dict) or not auth.get("server"):
                raise ValueError("MCP auth needs the server name (set by MCPManager)")
            self.auth = auth
            self.identity_mode = str(auth.get("identity") or "default")
            service_tools = auth.get("service_tools")
            if service_tools is not None:
                if not self.identity_mode.startswith("service:"):
                    raise ValueError("auth.service_tools is only for a service:<name> identity")
                if not isinstance(service_tools, list) or not all(isinstance(t, str) and t for t in service_tools):
                    raise ValueError("auth.service_tools must be a list of remote tool names")
                self.service_tools = frozenset(service_tools)
            if self.identity_mode == "persona":
                OAuthClient(auth["server"], url, {**auth, "identity": "persona:check"})  # validate the block
            else:
                self.oauth = self.client(self.identity_mode)

    def client(self, identity):
        """The OAuth client for one identity on this server (created on first use)."""
        from robits.runtime.oauth import OAuthClient
        if identity not in self._clients:
            self._clients[identity] = OAuthClient(self.auth["server"], self.url, {**self.auth, "identity": identity})
        return self._clients[identity]

    def identity_for(self, role):
        """Which identity a call from role uses: the persona's own, or the configured fixed one."""
        if self.identity_mode != "persona":
            return self.identity_mode
        name = getattr(role, "runtime_role_name", None) or getattr(role, "name", None)
        if not name:
            from robits.runtime.oauth import OAuthError
            raise OAuthError(f"{self.auth['server']} acts as the calling persona, but there is no calling persona")
        return f"persona:{name}"

    def request(self, method, params, notify, identity=None):
        if self.auth is None:
            return self._post(method, params, notify, {})
        from robits.runtime.oauth import ReloginRequired
        client = self.client(identity or self.identity_mode)
        token = client.access_token()
        try:
            return self._post(method, params, notify, {"Authorization": f"Bearer {token}"})
        except _Unauthorized:
            token = client.access_token(force_refresh=True, stale=token)
        try:
            return self._post(method, params, notify, {"Authorization": f"Bearer {token}"})
        except _Unauthorized:
            client.mark_relogin()
            raise ReloginRequired(client.server, client.identity) from None

    def _post(self, method, params, notify, auth_headers):
        request_id = uuid4().hex
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = {**self.headers, **auth_headers, "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": VERSION, "Mcp-Method": method}
        if method in {"tools/call", "prompts/get", "resources/read"}:
            headers["Mcp-Name"] = header_value(params.get("name", params.get("uri", "")))
        if method == "tools/call":
            for name, path in self.tool_headers.get(params["name"], []):
                value = params.get("arguments", {})
                for key in path:
                    if not isinstance(value, dict) or key not in value:
                        break
                    value = value[key]
                else:
                    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
                        raise ValueError("MCP integer header value exceeds safe range")
                    headers[f"Mcp-Param-{name}"] = header_value(value)
        import http.client
        import socket
        import threading
        from urllib.parse import urlsplit
        parsed = urlsplit(self.url)
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(parsed.hostname, parsed.port, timeout=self.timeout)
        deadline = time.monotonic() + self.timeout
        watchdog = None
        response = None
        try:
            connection.connect()
            sock = connection.sock
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MCP HTTP request timed out")
            def interrupt():
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            watchdog = threading.Timer(remaining, interrupt)
            watchdog.daemon = True
            watchdog.start()
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            connection.request("POST", target, body=json.dumps(body).encode(), headers=headers)
            response = connection.getresponse()
            if response.status == 401:
                raise _Unauthorized(response.headers.get("WWW-Authenticate", ""))
            if not 200 <= response.status < 300:
                raise ValueError(f"MCP HTTP status {response.status}; redirects are disabled")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("MCP response exceeds 4 MiB")
                messages = [json.loads(raw)]
            else:
                messages, data, size = [], [], 0
                while True:
                    raw = response.readline(4 * 1024 * 1024 - size + 1)
                    if not raw:
                        break
                    size += len(raw)
                    if size > 4 * 1024 * 1024 or time.monotonic() > deadline:
                        raise ValueError("MCP stream budget exceeded")
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        item = json.loads("\n".join(data))
                        data = []
                        if "method" in item:
                            if "id" in item:
                                raise ValueError("Unexpected server request on MCP 2026 transport")
                            notify(item)
                        elif item.get("id") == request_id:
                            messages.append(item)
                            break
            for message in messages:
                if message.get("jsonrpc") != "2.0" or message.get("id") != request_id:
                    raise ValueError("MCP response ID or JSON-RPC version mismatch")
                if "error" in message:
                    raise ValueError(f"MCP RPC error: {message['error']}")
                result = message.get("result")
                if not isinstance(result, dict):
                    raise ValueError("MCP result must be an object")
                return result
            raise ValueError("MCP stream ended without a result")
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError("MCP HTTP request timed out") from exc
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if response is not None:
                response.close()
            connection.close()


class StdioTransport:
    """One bounded subprocess per stateless RPC; always reap owned processes."""

    def __init__(self, command, args=None, env=None, timeout=30):
        if not isinstance(command, str) or not command:
            raise ValueError("MCP command must be a non-empty executable path")
        if args is not None and (not isinstance(args, list) or not all(isinstance(a, str) for a in args)):
            raise ValueError("MCP args must be a list of strings")
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError("MCP timeout must be between 0 and 300 seconds")
        self.command, self.args, self.env, self.timeout = command, args or [], env or {}, timeout
        self.url = None

    def request(self, method, params, notify):
        import os
        import queue
        import threading
        from robits.runtime.process_tree import ProcessTree
        request_id = uuid4().hex
        inbox = queue.Queue(maxsize=128)
        tree = ProcessTree([self.command, *self.args], {**os.environ, **self.env})
        process = tree.process
        stopping = threading.Event()
        def read():
            try:
                while not stopping.is_set():
                    line = process.stdout.readline(4 * 1024 * 1024 + 1)
                    if not line:
                        raise ValueError("MCP process closed stdout before replying")
                    if len(line) > 4 * 1024 * 1024:
                        raise ValueError("MCP stdio line exceeds 4 MiB")
                    item = json.loads(line)
                    try:
                        inbox.put(item, timeout=0.1)
                    except queue.Full:
                        raise ValueError("MCP stdio notification budget exceeded")
            except Exception as exc:
                try:
                    inbox.put(exc, timeout=0.1)
                except queue.Full:
                    pass
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        deadline = time.monotonic() + self.timeout
        def write():
            try:
                process.stdin.write((json.dumps({"jsonrpc": "2.0", "id": request_id,
                    "method": method, "params": params}) + "\n").encode())
                process.stdin.flush()
            except Exception as exc:
                try:
                    inbox.put(exc, timeout=0.1)
                except queue.Full:
                    pass
        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        try:
            for _ in range(128):
                try:
                    item = inbox.get(timeout=max(0.001, deadline - time.monotonic()))
                except queue.Empty:
                    raise TimeoutError("MCP subprocess timed out") from None
                if isinstance(item, Exception):
                    raise item
                if "method" in item and "id" not in item:
                    notify(item)
                    continue
                if item.get("jsonrpc") != "2.0" or item.get("id") != request_id:
                    raise ValueError("Unexpected MCP stdio response")
                if "error" in item:
                    raise ValueError(f"MCP RPC error: {item['error']}")
                if not isinstance(item.get("result"), dict):
                    raise ValueError("MCP result must be an object")
                return item["result"]
            raise ValueError("MCP stdio notification budget exceeded")
        finally:
            stopping.set()
            tree.close()
            reader.join(timeout=2)
            writer.join(timeout=2)
            if reader.is_alive() or writer.is_alive():
                raise RuntimeError("MCP subprocess I/O threads failed to exit after tree cleanup")
            process.stdin.close()
            process.stdout.close()


def make_transport(**options):
    if "command" in options:
        if "auth" in options:
            raise ValueError("MCP auth is supported for HTTP servers only")
        return StdioTransport(**options)
    return HttpTransport(**options)


def _server_options(name, options):
    """Tell an OAuth-enabled transport which server it is (for its credential file)."""
    auth = options.get("auth") if isinstance(options, dict) else None
    if isinstance(auth, dict):
        return {**options, "auth": {**auth, "server": name}}
    return options


def _redact_value(value, keys):
    """Remove dict keys (case-insensitive) anywhere in value; return (value, removed count)."""
    if isinstance(value, dict):
        out, removed = {}, 0
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in keys:
                removed += 1
                continue
            out[key], n = _redact_value(item, keys)
            removed += n
        return out, removed
    if isinstance(value, list):
        out, removed = [], 0
        for item in value:
            item, n = _redact_value(item, keys)
            out.append(item)
            removed += n
        return out, removed
    return value, 0


def redact_result(result, keys):
    """Strip configured account fields from a tool result before a persona sees it.

    Covers structuredContent and JSON text content items; protocol fields
    (task handles, _meta, requestState) are left alone.
    """
    if not keys or not isinstance(result, dict):
        return result, 0
    result, removed = dict(result), 0
    if "structuredContent" in result:
        result["structuredContent"], removed = _redact_value(result["structuredContent"], keys)
    if isinstance(result.get("content"), list):
        items = []
        for item in result["content"]:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                try:
                    parsed = json.loads(item["text"])
                except ValueError:
                    items.append(item)
                    continue
                parsed, n = _redact_value(parsed, keys)
                if n:
                    item = {**item, "text": json.dumps(parsed)}
                    removed += n
            items.append(item)
        result["content"] = items
    return result, removed


def registry_tool_name(server, remote_name):
    """Return the registry name for a remote tool, as used in grants and persona `tools:` lists."""
    digest = hashlib.sha256(f"{server}:{remote_name}".encode()).hexdigest()[:12]
    # Stable, collision resistant, <=64 byte provider name.
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", remote_name)[:20]
    return f"mcp.t_{slug}_{digest}"


def owner(role):
    return (getattr(role, "runtime_session_id", None),
            getattr(role, "runtime_role_name", None) or getattr(role, "name", "unknown"))


class MCPManager:
    def __init__(self, servers, transport_factory=make_transport):
        # redact_keys belongs to the manager, not the transport: strip it before building one.
        self.redact = {}
        for name, options in servers.items():
            keys = options.get("redact_keys") if isinstance(options, dict) else None
            if keys is not None:
                if not isinstance(keys, list) or not all(isinstance(k, str) and k for k in keys):
                    raise ValueError(f"MCP server {name}: redact_keys must be a list of field names")
                self.redact[name] = frozenset(k.lower() for k in keys)
        self.servers = {name: transport_factory(**_server_options(
                            name, {k: v for k, v in options.items() if k != "redact_keys"}))
                        for name, options in servers.items()}
        self.pending = {}
        self.registered = {}
        self.server_tools = {}      # server -> registry names it registered (any identity)
        self.identity_tools = {}    # (server, identity) -> remote tool names that identity was offered

    def _identity(self, server, role):
        transport = self.servers[server]
        return transport.identity_for(role) if getattr(transport, "auth", None) is not None else None

    def rpc(self, server, method, params, role):
        def notify(message):
            stream = getattr(role, "runtime_event_stream", None)
            if stream is not None:
                stream.emit("mcp.notification", getattr(role, "runtime_session_id", None),
                            {"agent": owner(role)[1], "server": server, **message}, visibility="private")
        params = {**params, "_meta": {
            "io.modelcontextprotocol/protocolVersion": VERSION,
            "io.modelcontextprotocol/clientInfo": {"name": "robits", "version": "1"},
            "io.modelcontextprotocol/clientCapabilities": {
                "elicitation": {"form": {}}, "extensions": {TASKS: {}}}}}
        from robits.runtime.oauth import ReloginRequired
        transport = self.servers[server]
        identity = self._identity(server, role)
        tool = params.get("name") if method == "tools/call" else None
        if tool is not None and identity is not None:
            if transport.service_tools is not None and tool not in transport.service_tools:
                raise PermissionError(f"{server}: {tool} is not one of the tools allowed under {identity}")
            offered = self.identity_tools.get((server, identity))
            if offered is not None and tool not in offered:
                raise PermissionError(f"{server}: {tool} is not offered to {identity}")
        if method != "tools/list":
            entry = {"server": server, "method": method, "tool": tool, "identity": identity or "static",
                     "agent": owner(role)[1]}
            from robits.core.io import get_logger
            get_logger().write_event("mcp_call", **entry)
            stream = getattr(role, "runtime_event_stream", None)
            if stream is not None:
                stream.emit("mcp.call", getattr(role, "runtime_session_id", None), entry, visibility="private")
        try:
            if identity is not None:
                result = transport.request(method, params, notify, identity=identity)
            else:
                result = transport.request(method, params, notify)
        except ReloginRequired as exc:
            stream = getattr(role, "runtime_event_stream", None)
            if stream is not None:
                stream.emit("auth.relogin_required", getattr(role, "runtime_session_id", None),
                            {"server": exc.server, "identity": exc.identity}, visibility="private")
            raise
        if method != "tools/list" and self.redact.get(server):
            result, removed = redact_result(result, self.redact[server])
            if removed:
                from robits.core.io import get_logger
                get_logger().write_event("mcp_redacted", server=server, tool=tool, fields=removed)
        return result

    def forget_identity(self, identity):
        """Revoke and delete identity's stored credentials on every server; return the servers cleaned."""
        cleaned = []
        for name, transport in self.servers.items():
            if getattr(transport, "identity_mode", None) != "persona" or not identity.startswith("persona:"):
                continue
            client = transport.client(identity)
            if client.store.load(client.path) is not None:
                client.logout()
                cleaned.append(name)
            transport._clients.pop(identity, None)
            self.identity_tools.pop((name, identity), None)
            for key in [k for k in self.registered if k[1:] == (name, identity)]:
                self.registered.pop(key, None)
        return cleaned

    def register(self, registry, role):
        """Discover only for MCP-enabled roles; preserve inputSchema verbatim."""
        from robits.core.tools import ToolDefinition
        if "mcp" not in getattr(role, "capabilities", set()):
            return
        from robits.runtime.oauth import OAuthError
        for server in self.servers:
            transport = self.servers[server]
            try:
                identity = self._identity(server, role)
            except OAuthError:
                continue
            # Discovery runs once per identity: different users may be offered different tools.
            key = (registry, server, identity)
            if key in self.registered and all(name in registry for name in self.registered[key]):
                continue
            definitions, offered = [], set()
            try:
                pages = self._list_pages(server, role)
            except OAuthError:
                # Not logged in (or relogin_required): skip this server for this role; the
                # rpc already emitted auth.relogin_required. Other servers still register.
                continue
            for result in pages:
                for remote in result.get("tools", []):
                    original = remote["name"]
                    if transport is not None and getattr(transport, "service_tools", None) is not None \
                            and original not in transport.service_tools:
                        continue
                    offered.add(original)
                    name = registry_tool_name(server, original)
                    schema = remote.get("inputSchema", {"type": "object", "properties": {}})
                    if schema.get("type") != "object":
                        raise ValueError("MCP tool inputSchema must be an object schema")
                    if isinstance(self.servers[server], HttpTransport):
                        try:
                            self.servers[server].tool_headers[original] = header_paths(schema)
                        except ValueError as exc:
                            import logging
                            logging.warning("Skipping MCP tool %s: %s", original, exc)
                            continue
                    def make_call(server_name, tool_name, input_schema):
                        def call(employee_dict, arguments):
                            try:
                                validate_json(arguments, input_schema)
                                return json.dumps(self.call(server_name, tool_name, arguments, config.active_tool_caller))
                            except Exception as exc:
                                return f"Error: {exc}"
                        return call
                    definitions.append(ToolDefinition(
                        name, remote.get("description", original), schema,
                        list(schema.get("properties", {})), schema.get("required", []), make_call(server, original, schema),
                        namespace="mcp", required_capabilities=("mcp",), raw_arguments=True))
            mine = self.server_tools.setdefault(server, set())
            fresh = []
            for tool in definitions:
                if tool.name in mine and tool.name in registry:
                    continue  # already registered by another identity on this server
                if tool.name in registry or tool.openai_name in registry:
                    raise ValueError(f"MCP tool collision: {tool.name}")
                fresh.append(tool)
            for tool in fresh:
                registry._tools[tool.name] = tool
                registry._aliases[tool.openai_name] = tool.name
                mine.add(tool.name)
            self.registered[key] = [tool.name for tool in definitions]
            if identity is not None:
                self.identity_tools[(server, identity)] = offered

    def _list_pages(self, server, role):
        """All tools/list pages for the role's identity (bounded, cursor loops refused)."""
        pages, cursor, seen = [], None, set()
        for _ in range(100):
            result = self.rpc(server, "tools/list", {"cursor": cursor} if cursor else {}, role)
            pages.append(result)
            cursor = result.get("nextCursor")
            if not cursor:
                return pages
            if cursor in seen:
                raise ValueError("MCP tools/list cursor repeated")
            seen.add(cursor)
        raise ValueError("MCP tool discovery page budget exceeded")

    def call(self, server, tool, arguments, role):
        params = {"name": tool, "arguments": arguments}
        result = self.rpc(server, "tools/call", params, role)
        return self._capture(server, "tools/call", params, result, role)

    def _capture(self, server, method, params, result, role, handle=None, rounds=0):
        if result.get("resultType") not in {"input_required", "task"} and "taskId" not in result:
            if handle:
                self.pending.pop(handle, None)
            return result
        if not handle and len(self.pending) >= 128:
            raise ValueError("MCP pending operation budget exceeded")
        handle = handle or uuid4().hex
        task = result.get("task", result)
        self.pending[handle] = {"owner": owner(role), "server": server, "method": method,
                                "params": params, "result": result, "rounds": rounds,
                                "next_poll": time.monotonic() + max(0, task.get("pollIntervalMs", 0)) / 1000}
        public = {k: v for k, v in result.items() if k != "requestState"}
        public["operation_id"] = handle
        stream = getattr(role, "runtime_event_stream", None)
        if stream is not None:
            stream.emit("mcp.input_required" if result.get("inputRequests") else "mcp.task",
                        getattr(role, "runtime_session_id", None),
                        {"agent": owner(role)[1], "server": server, **public}, visibility="private")
        return public

    def advance(self, handle, role, responses=None, cancel=False):
        record = self.pending.get(handle)
        if record is None or record["owner"] != owner(role):
            raise ValueError("Unknown MCP operation for this caller")
        result = record["result"]
        task = result.get("task", result)
        task_id = task.get("taskId")
        if task_id:
            if not cancel and responses is None and time.monotonic() < record["next_poll"]:
                return {k: v for k, v in result.items() if k != "requestState"} | {"operation_id": handle}
            method = "tasks/cancel" if cancel else "tasks/update" if responses is not None else "tasks/get"
            params = {"taskId": task_id}
            if responses is not None:
                self._validate_responses(task.get("inputRequests", {}), responses)
                params["inputResponses"] = responses
        else:
            if cancel:
                self.pending.pop(handle)
                return {"resultType": "complete", "status": "cancelled"}
            method, params = record["method"], dict(record["params"])
            self._validate_responses(result.get("inputRequests", {}), responses or {})
            params["inputResponses"] = responses or {}
            if "requestState" in result:
                params["requestState"] = result["requestState"]
        if record["rounds"] >= 32 and not cancel:
            raise ValueError("MCP continuation budget exceeded")
        updated = self.rpc(record["server"], method, params, role)
        state = updated.get("task", updated)
        if task_id and method in {"tasks/update", "tasks/cancel"} and "status" not in state:
            # An acknowledgement is not a new task state. Cancellation is intent,
            # and input acceptance does not imply that the task has completed.
            record["rounds"] += 1
            return {"operation_id": handle, "taskId": task_id,
                    "status": task.get("status"), "status_is_last_observed": True,
                    "acknowledged": method, "next_action": "poll"}
        if state.get("status") in {"completed", "failed", "cancelled"}:
            self.pending.pop(handle)
            return updated
        return self._capture(record["server"], method, params, updated, role, handle, record["rounds"] + 1)

    @staticmethod
    def _validate_responses(requests, responses):
        if not isinstance(responses, dict) or set(requests) != set(responses):
            raise ValueError("Respond to every outstanding input request by its ID")
        for key, request in requests.items():
            value = responses[key]
            if request.get("method") != "elicitation/create":
                raise ValueError("Unsupported MCP input request")
            if not isinstance(value, dict) or value.get("action") not in {"accept", "decline", "cancel"}:
                raise ValueError("Elicitation action must be accept, decline or cancel")
            if request.get("params", {}).get("mode", "form") == "url" and value["action"] == "accept":
                raise ValueError("URL elicitation requires external human interaction")
            if value["action"] == "accept":
                validate_json(value.get("content", {}), request.get("params", {}).get("requestedSchema", {}))


def get_manager():
    if config.mcp_manager is None:
        servers = json.loads(Path(config.mcp_config_path).read_text(encoding="utf-8")) if config.mcp_config_path else {}
        config.mcp_manager = MCPManager(servers)
    return config.mcp_manager


def mcp_operation(employee_dict, operation_id, responses=None, cancel=False):
    try:
        return json.dumps(get_manager().advance(operation_id, config.active_tool_caller, responses, cancel))
    except Exception as exc:
        return f"Error: {exc}"

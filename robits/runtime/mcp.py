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


class HttpTransport:
    """Bounded POST transport, including in-request SSE progress notifications."""

    def __init__(self, url, headers=None, timeout=30):
        from urllib.parse import urlsplit
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("MCP endpoint must use HTTP or HTTPS")
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError("MCP timeout must be between 0 and 300 seconds")
        self.url, self.headers, self.timeout = url, headers or {}, timeout
        self.tool_headers = {}

    def request(self, method, params, notify):
        request_id = uuid4().hex
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = {**self.headers, "Content-Type": "application/json",
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
    return StdioTransport(**options) if "command" in options else HttpTransport(**options)


def owner(role):
    return (getattr(role, "runtime_session_id", None),
            getattr(role, "runtime_role_name", None) or getattr(role, "name", "unknown"))


class MCPManager:
    def __init__(self, servers, transport_factory=make_transport):
        self.servers = {name: transport_factory(**options) for name, options in servers.items()}
        self.pending = {}
        self.registered = {}

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
        return self.servers[server].request(method, params, notify)

    def register(self, registry, role):
        """Discover only for MCP-enabled roles; preserve inputSchema verbatim."""
        from robits.core.tools import ToolDefinition
        if "mcp" not in getattr(role, "capabilities", set()):
            return
        for server in self.servers:
            key = (registry, server)
            if key in self.registered and all(name in registry for name in self.registered[key]):
                continue
            definitions, cursor, seen = [], None, set()
            for _ in range(100):
                result = self.rpc(server, "tools/list", {"cursor": cursor} if cursor else {}, role)
                for remote in result.get("tools", []):
                    original = remote["name"]
                    digest = hashlib.sha256(f"{server}:{original}".encode()).hexdigest()[:12]
                    # Stable, collision resistant, <=64 byte provider name.
                    slug = re.sub(r"[^a-zA-Z0-9_]", "_", original)[:20]
                    name = f"mcp.t_{slug}_{digest}"
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
                cursor = result.get("nextCursor")
                if not cursor:
                    break
                if cursor in seen:
                    raise ValueError("MCP tools/list cursor repeated")
                seen.add(cursor)
            else:
                raise ValueError("MCP tool discovery page budget exceeded")
            for tool in definitions:
                if tool.name in registry or tool.openai_name in registry:
                    raise ValueError(f"MCP tool collision: {tool.name}")
            for tool in definitions:
                registry._tools[tool.name] = tool
                registry._aliases[tool.openai_name] = tool.name
            self.registered[key] = [tool.name for tool in definitions]

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

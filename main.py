#!/usr/bin/env python3
from dataclasses import dataclass, field
from openai import OpenAI

import random
import time
import os
import json
import re
import yaml
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from termcolor import colored
import argparse
import sys
from pathlib import Path
from uuid import uuid4

from robits.memory.sqlite import (
    CHANNEL_ORG_CHAT,
    SOCIAL_PROFESSIONAL,
    SQLiteMemoryStore,
    compute_phase_shift,
)
from robits.runtime.sandbox import SandboxMetadata
from robits.runtime.tool_proposals import ToolProposalStore
from robits.runtime.workspace import AgentWorkspaceStore, WorkspacePathError


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_client():
    client_kwargs = {
        "api_key": os.environ.get("OPENAI_API_KEY", "not-needed"),
    }
    organization = os.environ.get("OPENAI_ORG")
    if organization:
        client_kwargs["organization"] = organization
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


client = make_client()
default_model = os.environ.get("ROBITS_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
costly_model = os.environ.get("ROBITS_COSTLY_MODEL", default_model)
cheap_model = os.environ.get("ROBITS_CHEAP_MODEL", default_model)
provider_api = os.environ.get("ROBITS_PROVIDER_API", "responses").strip().lower()
max_parallelism = max(1, int(os.environ.get("ROBITS_MAX_PARALLELISM", "1")))
max_api_retries = max(0, int(os.environ.get("ROBITS_MAX_API_RETRIES", "3")))
api_retry_base_seconds = max(0.0, float(os.environ.get("ROBITS_API_RETRY_BASE_SECONDS", "0.25")))
api_retry_max_seconds = max(api_retry_base_seconds, float(os.environ.get("ROBITS_API_RETRY_MAX_SECONDS", "4.0")))
model_call_gate = threading.BoundedSemaphore(max_parallelism)
builtin_search_url = os.environ.get("ROBITS_SEARCH_URL", "").strip()
memory_db_path = os.environ.get("ROBITS_MEMORY_DB")
memory_store = SQLiteMemoryStore(memory_db_path) if memory_db_path else None
tool_proposal_store = None
memory_max_depth = max(0, int(os.environ.get("ROBITS_MEMORY_MAX_DEPTH", "3")))
memory_max_rows = max(1, int(os.environ.get("ROBITS_MEMORY_MAX_ROWS", "100")))
memory_cache_threshold = max(512, int(os.environ.get("ROBITS_MEMORY_CACHE_THRESHOLD", "8192")))
memory_digest_interval = max(0, int(os.environ.get("ROBITS_DIGEST_INTERVAL", "0")))
memory_digest_context_chars = max(0, int(os.environ.get("ROBITS_DIGEST_CONTEXT_CHARS", "0")))
memory_digest_elapsed_seconds = max(0, int(os.environ.get("ROBITS_DIGEST_ELAPSED_SECONDS", "0")))
memory_identity_digest_interval = max(0, int(os.environ.get("ROBITS_IDENTITY_DIGEST_INTERVAL", "0")))
memory_goal_digest_interval = max(0, int(os.environ.get("ROBITS_GOAL_DIGEST_INTERVAL", "0")))
org_chat_context_lines = max(0, int(os.environ.get("ROBITS_ORG_CHAT_CONTEXT_LINES", "20")))
org_digest_interval = max(0, int(os.environ.get("ROBITS_ORG_DIGEST_INTERVAL", "0")))
_reasoning_effort_env = os.environ.get("ROBITS_REASONING_EFFORT", "").strip().lower() or None
_raw_clock_state = os.environ.get("ROBITS_CLOCK_STATE", "on").strip().lower()
clock_state = _raw_clock_state if _raw_clock_state in {"on", "off"} else "on"
agent_workspace_store = AgentWorkspaceStore()
_org_workspace = agent_workspace_store if memory_store is not None else None
default_location = os.environ.get("ROBITS_LOCATION", "").strip()
default_timezone = os.environ.get("ROBITS_TIMEZONE", "").strip()
SAFE_TOOL_BUILTINS = {
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    args: list[str]
    required_args: list[str]
    func: object
    aliases: tuple[str, ...] = ()
    namespace: str = ""
    required_capabilities: tuple[str, ...] = ()
    owner_capability: str | None = None
    system_tool: bool = False
    grantable: bool = True

    @property
    def openai_name(self):
        return self.name.replace(".", "__")

    def as_responses_tool(self):
        return {
            "type": "function",
            "name": self.openai_name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def as_chat_completion_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.openai_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._aliases = {}

    def clear(self):
        self._tools.clear()
        self._aliases.clear()

    def __contains__(self, name):
        return self.resolve_name(name) in self._tools

    def resolve_name(self, name):
        return self._aliases.get(name, name)

    def get(self, name):
        return self._tools[self.resolve_name(name)]

    def validate_tool_name(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string.")
        parts = name.split(".")
        if not all(part.isidentifier() for part in parts):
            raise ValueError(f"Invalid tool name: {name}")

    def compile_tool(self, name, arg_names, required_arg_names, code):
        self.validate_tool_name(name)
        for arg_name in required_arg_names:
            if arg_name not in arg_names:
                raise ValueError(f"Required tool argument is not defined: {arg_name}")
        for arg_name in arg_names:
            if not arg_name.isidentifier():
                raise ValueError(f"Invalid tool argument name: {arg_name}")
            if arg_name == "employee_dict":
                raise ValueError("Tool argument name 'employee_dict' is reserved.")

        optional_arg_names = [arg_name for arg_name in arg_names if arg_name not in required_arg_names]
        parameters = ["employee_dict"] + required_arg_names + [
            f"{arg_name}=None" for arg_name in optional_arg_names
        ]
        indented_code = "\n".join(
            f"    {line}" if line.strip() else "" for line in code.splitlines()
        )
        if not indented_code:
            indented_code = "    pass"
        function_name = f"_tool_{name.replace('.', '_')}"
        function_source = f"def {function_name}({', '.join(parameters)}):\n{indented_code}"
        local_dict = {}
        exec(
            function_source,
            {
                "__builtins__": SAFE_TOOL_BUILTINS,
                "Role": Role,
                "HR": HR,
                "create_lifecycle_role": create_lifecycle_role,
                "pause_lifecycle_role": pause_lifecycle_role,
                "retire_lifecycle_role": retire_lifecycle_role,
                "archive_lifecycle_role": archive_lifecycle_role,
                "list_lifecycle_roles": list_lifecycle_roles,
                "create_alarm": create_alarm,
                "list_alarms": list_alarms,
                "cancel_alarm": cancel_alarm,
                "memory_search": memory_search,
                "memory_list_digests": memory_list_digests,
                "memory_expand_digest": memory_expand_digest,
                "grant_tool_access": grant_tool_access,
                "revoke_tool_access": revoke_tool_access,
                "list_registered_tools": list_registered_tools,
                "propose_tool_change": propose_tool_change,
                "list_tool_proposals": list_tool_proposals,
                "approve_tool_proposal": approve_tool_proposal,
                "reject_tool_proposal": reject_tool_proposal,
                "rollout_tool_proposal": rollout_tool_proposal,
                "workspace_list": workspace_list,
                "workspace_read": workspace_read,
                "workspace_write": workspace_write,
                "workspace_delete": workspace_delete,
                "current_agent_context": current_agent_context,
                "builtin_web_search": builtin_web_search,
                "builtin_weather_lookup": builtin_weather_lookup,
                "builtin_file_search": builtin_file_search,
                "builtin_shell_run": builtin_shell_run,
                "builtin_tool_search": builtin_tool_search,
                "builtin_mcp_call": builtin_mcp_call,
                "builtin_computer_use": builtin_computer_use,
                "builtin_image_generation": builtin_image_generation,
                "org_chat_read": org_chat_read,
                "work_todo_add": work_todo_add,
            },
            local_dict,
        )
        return local_dict[function_name]

    def normalize_definition(self, instruction):
        if "function" in instruction:
            function = instruction["function"]
            name = function["name"]
            description = function.get("description", "")
            parameters = function.get("parameters", {"type": "object", "properties": {}})
            code = instruction["code"]
            aliases = tuple(instruction.get("aliases", ()))
        elif "name" in instruction:
            name = instruction["name"]
            description = instruction.get("description", "")
            parameters = instruction.get("parameters", {"type": "object", "properties": {}})
            code = instruction["code"]
            aliases = tuple(instruction.get("aliases", ()))
        else:
            name = instruction["code_name"]
            args = instruction.get("args")
            if not isinstance(args, list):
                raise ValueError("Tool args must be a list of objects with name fields.")
            properties = {}
            for arg in args:
                if not isinstance(arg, dict) or not isinstance(arg.get("name"), str):
                    raise ValueError("Tool args must be a list of objects with name fields.")
                properties[arg["name"]] = {"type": "string"}
            description = instruction.get("description", "")
            parameters = {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            }
            code = instruction["code"]
            aliases = tuple(instruction.get("aliases", ()))

        namespace = instruction.get("namespace") or name.split(".", 1)[0]
        required_capabilities = tuple(instruction.get("required_capabilities", ()))
        owner_capability = instruction.get("owner_capability")
        system_tool = bool(instruction.get("system_tool", False))
        grantable = bool(instruction.get("grantable", True))
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("Tool parameters must contain object properties and required list.")
        if not isinstance(required_capabilities, tuple) or not all(
            isinstance(capability, str) for capability in required_capabilities
        ):
            raise ValueError("Tool required_capabilities must be a list of strings.")
        if owner_capability is not None and not isinstance(owner_capability, str):
            raise ValueError("Tool owner_capability must be a string.")
        required_arg_names = []
        for arg_name in required:
            if not isinstance(arg_name, str) or arg_name not in properties:
                raise ValueError("Tool required args must be named properties.")
            required_arg_names.append(arg_name)
        arg_names = list(properties)

        return (
            name,
            description,
            parameters,
            arg_names,
            required_arg_names,
            code,
            aliases,
            namespace,
            required_capabilities,
            owner_capability,
            system_tool,
            grantable,
        )

    def register_definition(self, instruction):
        (
            name,
            description,
            parameters,
            arg_names,
            required_arg_names,
            code,
            aliases,
            namespace,
            required_capabilities,
            owner_capability,
            system_tool,
            grantable,
        ) = self.normalize_definition(instruction)
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already exists.")
        for alias in aliases:
            self.validate_tool_name(alias)
            if alias in self._aliases or alias in self._tools:
                raise ValueError(f"Tool alias '{alias}' already exists.")
        func = self.compile_tool(name, arg_names, required_arg_names, code)
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            args=arg_names,
            required_args=required_arg_names,
            func=func,
            aliases=aliases,
            namespace=namespace,
            required_capabilities=required_capabilities,
            owner_capability=owner_capability,
            system_tool=system_tool,
            grantable=grantable,
        )
        openai_name = tool.openai_name
        if openai_name != name and (openai_name in self._aliases or openai_name in self._tools):
            raise ValueError(f"Tool OpenAI name '{openai_name}' already exists.")
        self._tools[name] = tool
        for alias in aliases:
            self._aliases[alias] = name
        if openai_name != name:
            self._aliases[openai_name] = name
        return tool

    def list_tools(self, include_system=True, role=None, include_denied=True):
        rows = []
        for tool in sorted(self._tools.values(), key=lambda item: item.name):
            if tool.system_tool and not include_system:
                continue
            allowed = None if role is None else role_can_use_tool(role, tool)
            if not include_denied and allowed is False:
                continue
            rows.append(
                {
                    "name": tool.name,
                    "namespace": tool.namespace,
                    "description": tool.description,
                    "required_capabilities": list(tool.required_capabilities),
                    "owner_capability": tool.owner_capability,
                    "system_tool": tool.system_tool,
                    "grantable": tool.grantable,
                    "allowed": allowed,
                }
            )
        return rows

    def tools_for_role(self, role):
        return [tool for tool in self._tools.values() if role_can_use_tool(role, tool)]

    def execute(self, name, args, employee_dict, caller=None):
        try:
            self.validate_tool_name(name)
        except ValueError as e:
            return f"Error: {e}"
        resolved_name = self.resolve_name(name)
        if resolved_name not in self._tools:
            return f"Error: Tool '{name}' not found."
        tool = self._tools[resolved_name]
        if caller is not None and not role_can_use_tool(caller, tool):
            return f"Error: Role '{caller.name}' is not allowed to use tool '{resolved_name}'."
        missing_args = [arg_name for arg_name in tool.required_args if arg_name not in args]
        if missing_args:
            return f"Error: Missing args for tool '{name}': {missing_args}"
        unexpected_args = [arg_name for arg_name in args if arg_name not in tool.args]
        if unexpected_args:
            return f"Error: Unexpected args for tool '{name}': {unexpected_args}"
        global active_tool_caller, active_tool_caller_name
        previous_caller = active_tool_caller
        previous_caller_name = active_tool_caller_name
        active_tool_caller = caller
        active_tool_caller_name = None
        if caller is not None:
            for employee_name, employee in employee_dict.items():
                if employee is caller:
                    active_tool_caller_name = employee_name
                    break
            if active_tool_caller_name is None:
                active_tool_caller_name = getattr(caller, "name", None)
        try:
            result = tool.func(employee_dict=employee_dict, **args)
        finally:
            active_tool_caller = previous_caller
            active_tool_caller_name = previous_caller_name
        return f"Executed tool '{resolved_name}' with args {args}. Result: {result}"

    def as_responses_tools(self, role=None):
        tools = self._tools.values() if role is None else self.tools_for_role(role)
        return [tool.as_responses_tool() for tool in tools]

    def as_chat_completion_tools(self, role=None):
        tools = self._tools.values() if role is None else self.tools_for_role(role)
        return [tool.as_chat_completion_tool() for tool in tools]


tool_registry = ToolRegistry()
active_tool_caller = None
active_tool_caller_name = None


def _response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    chunks = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        elif getattr(item, "type", None) in {"output_text", "text"}:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _response_function_calls(response):
    calls = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        calls.append(item)
    return calls


def _is_retryable_api_error(error):
    status_code = getattr(error, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = error.__class__.__name__.lower()
    return any(token in name for token in ("ratelimit", "timeout", "connection"))


def _emit_role_tool_event(role, event_type, payload):
    event_stream = getattr(role, "runtime_event_stream", None)
    session_id = getattr(role, "runtime_session_id", None)
    if event_stream is not None and session_id is not None:
        event_stream.emit(event_type, session_id, payload)
    if memory_store is not None and event_type in {"tool_call.executed", "tool_call.failed"}:
        agent_id = getattr(role, "runtime_role_name", None) or getattr(role, "name", None)
        if agent_id:
            try:
                memory_store.append_tool_call(
                    tool_call_id=payload.get("call_id") or str(uuid4()),
                    agent_id=agent_id,
                    tool_name=payload.get("tool_name", ""),
                    arguments=payload.get("arguments"),
                    result_content=str(payload.get("result", "")),
                    status="executed" if event_type == "tool_call.executed" else "failed",
                    session_id=session_id,
                )
            except Exception:
                pass


def _record_role_tool_result(role, result):
    if not hasattr(role, "runtime_tool_results"):
        role.runtime_tool_results = []
    role.runtime_tool_results.append(result)


def _with_model_retries(operation):
    attempt = 0
    while True:
        try:
            with model_call_gate:
                return operation()
        except Exception as error:
            if attempt >= max_api_retries or not _is_retryable_api_error(error):
                raise
            delay = min(api_retry_max_seconds, api_retry_base_seconds * (2**attempt))
            if delay:
                time.sleep(delay + random.uniform(0, delay / 4))
            attempt += 1


class ModelProvider:
    def generate(self, role, model, sender, messages):
        raise NotImplementedError


class ChatCompletionsProvider(ModelProvider):
    def __init__(self, api_client):
        self.client = api_client

    def generate(self, role, model, sender, messages):
        def request():
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=role.max_tokens,
                n=1,
                temperature=role.temperature,
                user=f"robits_{role.name}",
                stream=True,
            )
            content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content += chunk.choices[0].delta.content
            return content

        return _with_model_retries(request)


class ResponsesProvider(ModelProvider):
    def __init__(self, api_client, registry=None):
        self.client = api_client
        self.registry = registry if registry is not None else tool_registry

    def generate(self, role, model, sender, messages):
        employee_dict = getattr(role, "employee_dict", None)
        tools = self.registry.as_responses_tools(role)
        kwargs = {
            "model": model,
            "input": messages,
            "max_output_tokens": role.max_tokens,
            "temperature": role.temperature,
            "user": f"robits_{role.name}",
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        effort = getattr(role, "reasoning_effort", None) or _reasoning_effort_env
        if effort and effort in {"low", "medium", "high"}:
            kwargs["reasoning"] = {"effort": effort}

        response = _with_model_retries(lambda: self.client.responses.create(**kwargs))
        for _ in range(8):
            function_calls = _response_function_calls(response)
            if not function_calls:
                return _response_text(response)
            if employee_dict is None:
                return "Error: Tool call requested but no employee dictionary is available."
            outputs = []
            for call in function_calls:
                call_id = getattr(call, "call_id", getattr(call, "id", ""))
                tool_name = getattr(call, "name", "")
                payload = {
                    "agent": getattr(role, "name", None),
                    "role_name": getattr(role, "runtime_role_name", None),
                    "provider_response_id": getattr(response, "id", None),
                    "call_id": call_id,
                    "tool_name": tool_name,
                }
                try:
                    args = json.loads(getattr(call, "arguments", "") or "{}")
                except json.JSONDecodeError as error:
                    payload["arguments"] = getattr(call, "arguments", "")
                    payload["error"] = f"Invalid tool arguments JSON: {error}"
                    _emit_role_tool_event(role, "tool_call.requested", payload)
                    result = f"Error: Invalid tool arguments JSON: {error}"
                else:
                    payload["arguments"] = args
                    _emit_role_tool_event(role, "tool_call.requested", payload)
                    result = self.registry.execute(
                        tool_name,
                        args,
                        employee_dict,
                        caller=role,
                    )
                status_payload = {**payload, "result": result}
                event_type = "tool_call.failed" if str(result).startswith("Error:") else "tool_call.executed"
                _emit_role_tool_event(role, event_type, status_payload)
                _record_role_tool_result(
                    role,
                    f"{event_type}: {tool_name}({json.dumps(payload.get('arguments'), sort_keys=True)}) -> {result}",
                )
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )
            previous_response_id = getattr(response, "id", None)
            if previous_response_id:
                response = _with_model_retries(
                    lambda: self.client.responses.create(
                        model=model,
                        input=outputs,
                        previous_response_id=previous_response_id,
                    )
                )
            else:
                kwargs["input"] = messages + outputs
                response = _with_model_retries(lambda: self.client.responses.create(**kwargs))
        return "Error: Too many consecutive tool-call rounds."


def make_model_provider():
    if provider_api in {"chat", "chat_completions", "chat-completions"}:
        return ChatCompletionsProvider(client)
    return ResponsesProvider(client)


model_provider = make_model_provider()

LIFECYCLE_STATES = ("proposed", "active", "paused", "retired", "exited")
PROTECTED_ROLE_NAMES = {"CEO", "HR"}
ALARM_RECURRENCES = {"once", "hourly", "daily", "weekly"}


@dataclass
class LifecycleEvent:
    action: str
    agent_name: str
    lifecycle_state: str
    requested_by: str | None = None
    approved_by: str | None = None
    reason: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass
class Alarm:
    alarm_id: str
    agent_name: str
    reminder: str
    due_at: str
    recurrence: str = "once"
    status: str = "active"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _runtime_timezone():
    if default_timezone:
        try:
            return ZoneInfo(default_timezone)
        except ZoneInfoNotFoundError:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def _runtime_timezone_name(tzinfo):
    if default_timezone:
        return default_timezone
    if tzinfo is None:
        return "UTC"
    return getattr(tzinfo, "key", None) or str(tzinfo)


def _get_identity_digests(agent_id):
    """Return (primary_content, secondary_content) ordered by clock_state."""
    if memory_store is None or not agent_id:
        return None, None

    def _fetch(rel_type):
        try:
            rows = memory_store.list_memory_digests(
                agent_id=agent_id, digest_type="identity",
                current_only=True, limit=1, relationship_type=rel_type,
            )
            return rows[0]["content"] if rows else None
        except Exception:
            return None

    personal = _fetch("personal")
    work = _fetch("work") or _fetch(None)  # fall back to untagged (legacy)
    if clock_state == "on":
        return work, personal
    return personal, work


def agent_runtime_context(role=None):
    now_utc = datetime.now(timezone.utc)
    local_tz = _runtime_timezone()
    now_local = now_utc.astimezone(local_tz)
    role_name = (
        (active_tool_caller_name if role is active_tool_caller else None)
        or getattr(role, "runtime_role_name", None)
        or getattr(role, "name", None)
    )
    agent_name = getattr(role, "name", None)
    # runtime_role_name is the participant key (FK-safe); fall back to role.name for headless use.
    canonical_id = getattr(role, "runtime_role_name", None) or agent_name
    primary_id, secondary_id = _get_identity_digests(canonical_id)
    context = {
        "agent_name": agent_name,
        "role_name": role_name,
        "session_id": getattr(role, "runtime_session_id", None),
        "current_datetime_utc": now_utc.isoformat(timespec="seconds"),
        "current_datetime_local": now_local.isoformat(timespec="seconds"),
        "current_date_local": now_local.date().isoformat(),
        "timezone": _runtime_timezone_name(local_tz),
        "location": default_location or None,
        "clock_state": clock_state,
        "identity_primary": primary_id,
        "identity_secondary": secondary_id,
    }
    return {key: value for key, value in context.items() if value is not None}


def format_agent_context(role=None):
    context = agent_runtime_context(role)
    return (
        "\nRuntime context available to your tools and decisions:\n"
        f"{json.dumps(context, sort_keys=True)}\n"
        "Use this context for dates, times, location, identity, and session references. "
        "Do not invent missing context; call agent.context if you need to refresh it.\n"
    )


def format_org_chat_context(transcript, limit):
    if not transcript or limit == 0:
        return ""
    recent = transcript[-limit:]
    lines = []
    for e in recent:
        lines.append(f"[Turn {e.turn}] {e.sender} -> {e.receiver}: {e.prompt[:400]}")
        if e.response:
            lines.append(f"  -> {e.response[:400]}")
    return "\nRecent org chat:\n" + "\n".join(lines) + "\n"


def current_agent_context(employee_dict):
    del employee_dict
    return json.dumps(agent_runtime_context(active_tool_caller), sort_keys=True)


def format_verified_tool_results(system_events):
    verified_events = [event for event in system_events if isinstance(event, str) and event.strip()]
    if not verified_events:
        return ""
    lines = "\n".join(f"- {event}" for event in verified_events)
    return (
        "Verified runtime results from recent tool calls:\n"
        f"{lines}\n"
        "You may rely on these results as completed runtime facts. "
        "Do not claim other tool-created artifacts or side effects without a verified result."
    )


def deliver_verified_tool_results(role, system_events):
    content = format_verified_tool_results(system_events)
    if not content or not hasattr(role, "conversation_history"):
        return
    role_history = role.conversation_history.setdefault(getattr(role, "name", ""), [])
    role_history.append({"role": "system", "content": content})


def prepend_verified_tool_results(prompt, system_events):
    content = format_verified_tool_results(system_events)
    if not content:
        return prompt
    return f"{content}\n\n{prompt}" if prompt else content


def _parse_datetime(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Timestamp must be a non-empty ISO datetime string.")
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_datetime(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _normalize_capabilities(capabilities=None):
    if capabilities is None:
        return set()
    if isinstance(capabilities, str):
        raw = [capabilities]
    else:
        raw = list(capabilities)
    normalized = set()
    for capability in raw:
        if not isinstance(capability, str) or not capability.strip():
            raise ValueError("Capabilities must be non-empty strings.")
        normalized.add(capability.strip())
    return normalized


def _normalize_tool_grants(tool_grants=None):
    if tool_grants is None:
        return set()
    if isinstance(tool_grants, str):
        raw = [tool_grants]
    else:
        raw = list(tool_grants)
    normalized = set()
    for grant in raw:
        if not isinstance(grant, str) or not grant.strip():
            raise ValueError("Tool grants must be non-empty strings.")
        normalized.add(_normalize_tool_grant(grant.strip()))
    return normalized


def _normalize_tool_grant(tool_name):
    if tool_name == "*":
        return tool_name
    if tool_name.endswith(".*"):
        namespace = tool_name[:-2]
        tool_registry.validate_tool_name(namespace)
        return f"{namespace}.*"
    tool_registry.validate_tool_name(tool_name)
    return tool_registry.resolve_name(tool_name)


def _tool_grant_matches(tool_name, grant):
    if grant == "*":
        return True
    if grant.endswith(".*"):
        return tool_name.startswith(grant[:-1])
    return tool_name == grant


def role_can_use_tool(role, tool):
    if getattr(role, "name", None) == "CEO":
        return True
    capabilities = getattr(role, "capabilities", set())
    if tool.required_capabilities and not set(tool.required_capabilities).issubset(capabilities):
        return False
    grants = getattr(role, "allowed_tools", set())
    return any(_tool_grant_matches(tool.name, grant) for grant in grants)


def _caller_can_act_for_agent(agent_name):
    caller = active_tool_caller
    if caller is None:
        return False
    if active_tool_caller_name == agent_name or getattr(caller, "name", None) == agent_name:
        return True
    capabilities = getattr(caller, "capabilities", set())
    return bool({"operator", "hr"} & capabilities)


def _caller_name_for_error():
    return active_tool_caller_name or getattr(active_tool_caller, "name", "unknown caller")


def _workspace_agent_name(agent_name):
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return None, f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return None, f"Error: Role '{_caller_name_for_error()}' cannot access workspace for '{normalized_name}'."
    return normalized_name, None


def workspace_list(employee_dict, agent_name, path=""):
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        return json.dumps(agent_workspace_store.list(normalized_name, path), sort_keys=True)
    except WorkspacePathError as e:
        return f"Error: {e}"


def workspace_read(employee_dict, agent_name, path, max_bytes=65536):
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        read_limit = 65536 if max_bytes is None else int(max_bytes)
        return json.dumps(
            agent_workspace_store.read(normalized_name, path, max_bytes=read_limit),
            sort_keys=True,
        )
    except (WorkspacePathError, ValueError) as e:
        return f"Error: {e}"


def workspace_write(employee_dict, agent_name, path, content, append=False):
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        return json.dumps(agent_workspace_store.write(normalized_name, path, content, append=append), sort_keys=True)
    except WorkspacePathError as e:
        return f"Error: {e}"


def workspace_delete(employee_dict, agent_name, path):
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        return json.dumps(agent_workspace_store.delete(normalized_name, path), sort_keys=True)
    except WorkspacePathError as e:
        return f"Error: {e}"


def org_chat_read(employee_dict, limit=20):
    del employee_dict
    if _org_workspace is None:
        return json.dumps({"lines": [], "note": "org chat not available"})
    try:
        result = _org_workspace.read("org", "org_chat.jsonl")
    except Exception:
        return json.dumps({"lines": [], "note": "no org chat history yet"})
    effective_limit = max(0, min(int(limit or 20), 100))
    if effective_limit == 0:
        return json.dumps({"lines": [], "total": 0})
    all_lines = [ln for ln in result["content"].splitlines() if ln.strip()]
    tail = all_lines[-effective_limit:]
    parsed = []
    for ln in tail:
        try:
            parsed.append(json.loads(ln))
        except Exception:
            pass
    return json.dumps({"lines": parsed, "total": len(all_lines)})


def work_todo_add(employee_dict, title, content=None):
    del employee_dict
    if memory_store is None:
        return json.dumps({"error": "memory store not available"})
    agent_id = active_tool_caller_name or getattr(active_tool_caller, "name", None)
    if not agent_id:
        return json.dumps({"error": "could not determine caller identity"})
    try:
        todo_id = memory_store.append_todo(
            agent_id=agent_id,
            title=title,
            content=content,
            status="open",
        )
        return json.dumps({"todo_id": todo_id, "title": title, "status": "open"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def grant_tool_access(employee_dict, role_name, tool_name, granted_by=None):
    try:
        normalized_name = validate_role_name(role_name)
        normalized_tool_name = _normalize_tool_grant(tool_name)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    if normalized_tool_name != "*" and not normalized_tool_name.endswith(".*") and normalized_tool_name not in tool_registry:
        return f"Error: Tool '{tool_name}' not found."
    if normalized_tool_name.endswith(".*"):
        namespace = normalized_tool_name[:-2]
        if not any(tool.name.startswith(f"{namespace}.") for tool in tool_registry._tools.values()):
            return f"Error: Tool namespace '{namespace}' not found."
    role = employee_dict[normalized_name]
    role.allowed_tools.add(normalized_tool_name)
    actor_text = f" by {granted_by}" if granted_by else ""
    return f"Granted tool access '{normalized_tool_name}' to role '{normalized_name}'{actor_text}."


def revoke_tool_access(employee_dict, role_name, tool_name, revoked_by=None):
    try:
        normalized_name = validate_role_name(role_name)
        normalized_tool_name = _normalize_tool_grant(tool_name)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    role = employee_dict[normalized_name]
    if normalized_tool_name not in role.allowed_tools:
        return f"Error: Role '{normalized_name}' does not have grant '{normalized_tool_name}'."
    role.allowed_tools.remove(normalized_tool_name)
    actor_text = f" by {revoked_by}" if revoked_by else ""
    return f"Revoked tool access '{normalized_tool_name}' from role '{normalized_name}'{actor_text}."


def list_registered_tools(employee_dict, role_name=None, include_system=True, only_allowed=False):
    role = None
    if only_allowed and not role_name:
        return "Error: role_name is required when only_allowed is true."
    if role_name:
        try:
            normalized_name = validate_role_name(role_name)
        except ValueError as e:
            return f"Error: {e}"
        if normalized_name not in employee_dict:
            return f"Error: Role '{normalized_name}' not found."
        role = employee_dict[normalized_name]
    return json.dumps(
        tool_registry.list_tools(
            include_system=include_system,
            role=role,
            include_denied=not only_allowed,
        ),
        sort_keys=True,
    )


def _coerce_json_object(value, default=None):
    if value is None:
        return {} if default is None else default
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON object: {error}")
    if not isinstance(value, dict):
        raise ValueError("Value must be a JSON object.")
    return value


def _coerce_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Value must be a list of strings.")
    return value


def get_tool_proposal_store():
    global tool_proposal_store
    if tool_proposal_store is None:
        tool_proposal_store = ToolProposalStore(
            os.environ.get("ROBITS_TOOL_PROPOSALS_FILE", "var/tool_proposals.json")
        )
    return tool_proposal_store


def propose_tool_change(
    employee_dict,
    requested_by,
    tool_name,
    description,
    action="create",
    parameters=None,
    required_capabilities=None,
    owner_capability=None,
    safety_notes="",
    implementation_notes="",
):
    del employee_dict
    try:
        tool_registry.validate_tool_name(tool_name)
        normalized_parameters = _coerce_json_object(
            parameters,
            {"type": "object", "properties": {}, "required": []},
        )
        normalized_capabilities = _coerce_string_list(required_capabilities)
    except ValueError as e:
        return f"Error: {e}"
    if not isinstance(description, str) or not description.strip():
        return "Error: Tool proposal description must be a non-empty string."
    if action not in {"create", "update"}:
        return f"Error: Tool proposal action must be create or update."
    if tool_name in tool_registry:
        tool = tool_registry.get(tool_name)
        if tool.system_tool:
            return f"Error: System tool '{tool_name}' cannot be changed by SE proposal."
    proposal = get_tool_proposal_store().create(
        requested_by=requested_by,
        tool_name=tool_name,
        action=action,
        description=description.strip(),
        parameters=normalized_parameters,
        required_capabilities=normalized_capabilities,
        owner_capability=owner_capability,
        safety_notes=safety_notes,
        implementation_notes=implementation_notes,
    )
    return json.dumps(proposal, sort_keys=True)


def list_tool_proposals(employee_dict, status=None):
    del employee_dict
    return json.dumps(get_tool_proposal_store().list(status=status), sort_keys=True)


def approve_tool_proposal(employee_dict, proposal_id, approved_by, implementation_notes=""):
    del employee_dict
    store = get_tool_proposal_store()
    proposal = store.get(proposal_id)
    if proposal is None:
        return f"Error: Tool proposal '{proposal_id}' not found."
    if proposal["status"] not in {"proposed", "approved"}:
        return f"Error: Tool proposal '{proposal_id}' cannot be approved from status '{proposal['status']}'."
    updated = store.update(
        proposal_id,
        status="approved",
        approver=approved_by,
        implementation_notes=implementation_notes or proposal.get("implementation_notes", ""),
    )
    return json.dumps(updated, sort_keys=True)


def reject_tool_proposal(employee_dict, proposal_id, rejected_by, reason):
    del employee_dict
    if not isinstance(reason, str) or not reason.strip():
        return "Error: Rejection reason must be a non-empty string."
    store = get_tool_proposal_store()
    proposal = store.get(proposal_id)
    if proposal is None:
        return f"Error: Tool proposal '{proposal_id}' not found."
    if proposal["status"] in {"operationalized", "rejected"}:
        return f"Error: Tool proposal '{proposal_id}' cannot be rejected from status '{proposal['status']}'."
    updated = store.update(
        proposal_id,
        status="rejected",
        approver=rejected_by,
        rejection_reason=reason.strip(),
    )
    return json.dumps(updated, sort_keys=True)


def rollout_tool_proposal(employee_dict, proposal_id, role_name, granted_by=None, rollout_notes=""):
    store = get_tool_proposal_store()
    proposal = store.get(proposal_id)
    if proposal is None:
        return f"Error: Tool proposal '{proposal_id}' not found."
    if proposal["status"] != "approved":
        return f"Error: Tool proposal '{proposal_id}' must be approved before rollout."
    tool_name = proposal["tool_name"]
    if tool_name not in tool_registry:
        return f"Error: Tool '{tool_name}' is not registered; implement it before rollout."
    grant_result = grant_tool_access(
        employee_dict,
        role_name,
        tool_name,
        granted_by=granted_by,
    )
    if grant_result.startswith("Error:"):
        return grant_result
    granted_roles = sorted(set(proposal.get("granted_roles", [])) | {validate_role_name(role_name)})
    updated = store.update(
        proposal_id,
        status="operationalized",
        rollout_notes=rollout_notes or proposal.get("rollout_notes", ""),
        granted_roles=granted_roles,
    )
    return json.dumps({"proposal": updated, "grant_result": grant_result}, sort_keys=True)


def validate_role_name(role_name):
    if not isinstance(role_name, str) or not role_name.strip():
        raise ValueError("Role name must be a non-empty string.")
    normalized = role_name.strip()
    if "," in normalized:
        raise ValueError("Role name cannot contain commas.")
    if len(normalized) > 64:
        raise ValueError("Role name cannot exceed 64 characters.")
    return normalized


def validate_role_description(role_description):
    if not isinstance(role_description, str) or not role_description.strip():
        raise ValueError("Role description must be a non-empty string.")
    return role_description.strip()


def record_lifecycle_event(
    role,
    action,
    lifecycle_state,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    event = LifecycleEvent(
        action=action,
        agent_name=role.name,
        lifecycle_state=lifecycle_state,
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
    )
    if not hasattr(role, "lifecycle_events"):
        role.lifecycle_events = []
    role.lifecycle_events.append(event)
    role.lifecycle_state = lifecycle_state
    if memory_store is not None:
        try:
            memory_store.upsert_agent(
                role.name,
                type(role).__name__,
                display_name=role.name,
                lifecycle_state=lifecycle_state,
            )
            summary = f"[lifecycle:{action}] {role.name} -> {lifecycle_state}"
            if requested_by:
                summary += f" (requested_by={requested_by})"
            if approved_by:
                summary += f" (approved_by={approved_by})"
            if reason:
                summary += f" reason={reason}"
            memory_store.append_memory_entry(
                agent_id=role.name,
                kind="lifecycle",
                content=summary,
                source="system",
            )
        except Exception:
            pass
    return event


def format_lifecycle_actor_text(requested_by=None, approved_by=None):
    actor_parts = []
    if requested_by:
        actor_parts.append(f"requested by {requested_by}")
    if approved_by:
        actor_parts.append(f"approved by {approved_by}")
    return f" ({', '.join(actor_parts)})" if actor_parts else ""


def create_lifecycle_role(
    employee_dict,
    role_name,
    role_description,
    capabilities=None,
    tool_grants=None,
    requested_by=None,
    approved_by=None,
):
    try:
        normalized_name = validate_role_name(role_name)
        normalized_description = validate_role_description(role_description)
        normalized_capabilities = _normalize_capabilities(capabilities)
        normalized_tool_grants = _normalize_tool_grants(tool_grants)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name in employee_dict:
        return f"Error: Role '{normalized_name}' already exists."
    if len(employee_dict) >= HR.max_organization_members:
        return f"Error: The organization has reached its maximum size of {HR.max_organization_members} members."

    new_role = Role(normalized_name, normalized_description, employee_dict)
    new_role.capabilities = normalized_capabilities
    new_role.allowed_tools.update(normalized_tool_grants)
    record_lifecycle_event(
        new_role,
        action="create",
        lifecycle_state="active",
        requested_by=requested_by,
        approved_by=approved_by,
    )
    employee_dict[normalized_name] = new_role
    actor_text = format_lifecycle_actor_text(requested_by, approved_by)
    capabilities_text = (
        f" capabilities={sorted(normalized_capabilities)}" if normalized_capabilities else ""
    )
    return f"Created a new role: {normalized_name} (state: active{capabilities_text}){actor_text}"


def change_lifecycle_state(
    employee_dict,
    role_name,
    lifecycle_state,
    action,
    requested_by=None,
    approved_by=None,
    reason=None,
    allowed_from=None,
):
    try:
        normalized_name = validate_role_name(role_name)
    except ValueError as e:
        return f"Error: {e}"
    if lifecycle_state not in LIFECYCLE_STATES:
        return f"Error: Invalid lifecycle state '{lifecycle_state}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."

    role = employee_dict[normalized_name]
    current_state = getattr(role, "lifecycle_state", "active")
    if allowed_from is not None and current_state not in allowed_from:
        allowed_text = ", ".join(allowed_from)
        return (
            f"Error: Cannot {action} role '{normalized_name}' from lifecycle "
            f"state '{current_state}'. Expected one of: {allowed_text}."
        )
    record_lifecycle_event(
        role,
        action=action,
        lifecycle_state=lifecycle_state,
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
    )
    actor_text = format_lifecycle_actor_text(requested_by, approved_by)
    reason_text = f" Reason: {reason}" if reason else ""
    return f"Updated role '{normalized_name}' to lifecycle state '{lifecycle_state}'{actor_text}.{reason_text}"


def pause_lifecycle_role(
    employee_dict,
    role_name,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    return change_lifecycle_state(
        employee_dict,
        role_name,
        "paused",
        "pause",
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
        allowed_from=("active",),
    )


def retire_lifecycle_role(
    employee_dict,
    role_name,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    return change_lifecycle_state(
        employee_dict,
        role_name,
        "retired",
        "retire",
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
        allowed_from=("active", "paused"),
    )


def _is_role_protected(role_name, role):
    capabilities = getattr(role, "capabilities", set())
    return (
        role_name in PROTECTED_ROLE_NAMES
        or "protected" in capabilities
        or "essential" in capabilities
    )


def archive_lifecycle_role(
    employee_dict,
    role_name,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    try:
        normalized_name = validate_role_name(role_name)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    role = employee_dict[normalized_name]
    if _is_role_protected(normalized_name, role):
        return f"Error: Role '{normalized_name}' is protected and cannot be removed from the organization."
    return change_lifecycle_state(
        employee_dict,
        normalized_name,
        "exited",
        "archive",
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
        allowed_from=("active", "paused", "retired"),
    )


def list_lifecycle_roles(employee_dict, include_exited=True):
    rows = []
    for name, role in sorted(employee_dict.items()):
        state = getattr(role, "lifecycle_state", "active")
        if not include_exited and state == "exited":
            continue
        capabilities = sorted(getattr(role, "capabilities", set()))
        tool_grants = sorted(getattr(role, "allowed_tools", set()))
        rows.append(
            {
                "role_name": name,
                "lifecycle_state": state,
                "capabilities": capabilities,
                "tool_grants": tool_grants,
            }
        )
    return json.dumps(rows, sort_keys=True)


def _active_alarms(role):
    return [alarm for alarm in getattr(role, "alarms", []) if alarm.status == "active"]


def create_alarm(employee_dict, agent_name, reminder, due_at, recurrence="once"):
    try:
        normalized_name = validate_role_name(agent_name)
        due = _parse_datetime(due_at)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot manage alarms for '{normalized_name}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    if not isinstance(reminder, str) or not reminder.strip():
        return "Error: Reminder must be a non-empty string."
    recurrence = (recurrence or "once").strip().lower()
    if recurrence not in ALARM_RECURRENCES:
        return f"Error: Invalid recurrence '{recurrence}'."
    now = datetime.now(timezone.utc)
    comparable_due = due.astimezone(timezone.utc) if due.tzinfo else due.replace(tzinfo=timezone.utc)
    if comparable_due <= now:
        context = agent_runtime_context(employee_dict[normalized_name])
        return (
            f"Error: Alarm due_at '{_format_datetime(due)}' is not in the future. "
            f"Current local time is {context.get('current_datetime_local')}."
        )
    role = employee_dict[normalized_name]
    if len(_active_alarms(role)) >= 5:
        return f"Error: Role '{normalized_name}' already has the maximum of 5 active alarms."
    alarm = Alarm(
        alarm_id=f"alarm-{uuid4()}",
        agent_name=normalized_name,
        reminder=reminder.strip(),
        due_at=_format_datetime(due),
        recurrence=recurrence,
    )
    role.alarms.append(alarm)
    return f"Created alarm '{alarm.alarm_id}' for {normalized_name} at {alarm.due_at}."


def list_alarms(employee_dict, agent_name, include_inactive=False):
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect alarms for '{normalized_name}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    alarms = []
    for alarm in getattr(employee_dict[normalized_name], "alarms", []):
        if alarm.status != "active" and not include_inactive:
            continue
        alarms.append(
            {
                "alarm_id": alarm.alarm_id,
                "reminder": alarm.reminder,
                "due_at": alarm.due_at,
                "recurrence": alarm.recurrence,
                "status": alarm.status,
            }
        )
    return json.dumps(alarms, sort_keys=True)


def cancel_alarm(employee_dict, agent_name, alarm_id):
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot manage alarms for '{normalized_name}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    for alarm in getattr(employee_dict[normalized_name], "alarms", []):
        if alarm.alarm_id == alarm_id:
            alarm.status = "canceled"
            return f"Canceled alarm '{alarm_id}' for {normalized_name}."
    return f"Error: Alarm '{alarm_id}' not found for {normalized_name}."


def _require_memory_store():
    if memory_store is None:
        return None, "Error: No SQLite memory store is configured."
    return memory_store, None


def _write_memory_cache(agent_name, filename, content):
    """Write content to the agent's .memory-cache/ workspace directory."""
    cache_path = f".memory-cache/{filename}"
    try:
        agent_workspace_store.write(agent_name, cache_path, content)
        return cache_path
    except Exception:
        return None


def _condense_if_large(agent_name, tool_label, result_json):
    """Return result_json directly, or cache it and return a snippet if too large."""
    if len(result_json) <= memory_cache_threshold:
        return result_json
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{tool_label.replace('.', '_')}_{uuid4().hex[:8]}.json"
    cache_path = _write_memory_cache(agent_name, filename, result_json)
    snippet = result_json[:2048]
    condensed = {
        "truncated": True,
        "total_chars": len(result_json),
        "snippet": snippet,
    }
    if cache_path:
        condensed["cache_path"] = cache_path
        condensed["note"] = (
            f"Full result ({len(result_json)} chars) cached in agent workspace at "
            f"'{cache_path}'. Use agent.files_read to retrieve it."
        )
    else:
        condensed["note"] = "Result truncated; workspace cache unavailable."
    return json.dumps(condensed, sort_keys=True)


def memory_search(employee_dict, agent_name, query, limit=10, cascade=True):
    del employee_dict
    store, error = _require_memory_store()
    if error:
        return error
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect memory for '{normalized_name}'."
    if not isinstance(query, str) or not query.strip():
        return "Error: Memory query must be a non-empty string."
    effective_limit = max(1, min(int(limit or 10), memory_max_rows))
    if cascade:
        results = store.search_cascade(query, agent_id=normalized_name, limit=effective_limit)
    else:
        results = store.search(query, agent_id=normalized_name, limit=effective_limit)
    result_json = json.dumps([result.__dict__ for result in results], sort_keys=True)
    if clock_state == "off" and any(
        getattr(r, "conversation_type", None) == "org_chat" for r in results
    ):
        note = (
            "[System note: work-related content surfaced in these results. "
            "Consider parking any useful insights via the work.todo tool "
            "and return focus to the current personal context.]\n"
        )
        result_json = note + result_json
    return _condense_if_large(normalized_name, "memory_search", result_json)


def memory_list_digests(employee_dict, agent_name, digest_type=None, limit=10):
    del employee_dict
    store, error = _require_memory_store()
    if error:
        return error
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect memory for '{normalized_name}'."
    effective_limit = max(1, min(int(limit or 10), memory_max_rows))
    digests = store.list_memory_digests(
        agent_id=normalized_name,
        digest_type=digest_type,
        current_only=True,
        accessible_only=True,
        limit=effective_limit,
    )
    result_json = json.dumps(digests, sort_keys=True)
    return _condense_if_large(normalized_name, "memory_list_digests", result_json)


def memory_expand_digest(employee_dict, agent_name, digest_id, recursive=True, max_depth=None, max_chars=None):
    del employee_dict
    store, error = _require_memory_store()
    if error:
        return error
    try:
        normalized_name = validate_role_name(agent_name)
        digest_id = int(digest_id)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect memory for '{normalized_name}'."
    digest = store.get_memory_digest(digest_id)
    if digest is None:
        return f"Error: Memory digest '{digest_id}' not found."
    if digest.get("agent_id") not in {None, normalized_name}:
        return f"Error: Memory digest '{digest_id}' is not accessible to {normalized_name}."
    if digest.get("system_only") or digest.get("accessibility") != "agent":
        return f"Error: Memory digest '{digest_id}' is system-only."
    effective_depth = max(
        0,
        min(
            int(max_depth) if max_depth is not None else memory_max_depth,
            memory_max_depth,
        ),
    )
    rows = store.expand_memory_digest_sources(
        digest_id,
        recursive=recursive,
        max_depth=effective_depth if recursive else None,
    )
    rows = [
        {
            **row.__dict__,
            "source_path": list(row.source_path),
        }
        if hasattr(row, "__dict__")
        else row
        for row in rows
    ]
    rows = rows[:memory_max_rows]
    result_json = json.dumps(rows, sort_keys=True)
    effective_max_chars = max(1, int(max_chars)) if max_chars is not None else None
    if effective_max_chars is not None and len(result_json) > effective_max_chars:
        condensed = {
            "truncated": True,
            "total_chars": len(result_json),
            "note": f"Result exceeded max_chars={effective_max_chars}. Use a higher max_chars or memory.search to narrow results.",
        }
        return json.dumps(condensed, sort_keys=True)
    return _condense_if_large(normalized_name, "memory_expand_digest", result_json)


def builtin_web_search(employee_dict, query, num_results=5):
    del employee_dict
    if not isinstance(query, str) or not query.strip():
        return "Error: Search query must be a non-empty string."
    n = max(1, min(20, int(num_results or 5)))
    if builtin_search_url:
        import urllib.request
        import urllib.parse
        url = f"{builtin_search_url}?q={urllib.parse.quote(query.strip())}&num={n}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "robits/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()
        except Exception as exc:
            return f"Error: Search request failed: {exc}"
    import urllib.request
    import urllib.parse
    qs = urllib.parse.urlencode({"q": query.strip(), "format": "json", "no_html": "1", "skip_disambig": "1"})
    try:
        req = urllib.request.Request(
            f"https://api.duckduckgo.com/?{qs}",
            headers={"User-Agent": "robits/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return f"Error: Web search failed: {exc}"
    results = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", query.strip()),
            "url": data.get("AbstractURL", ""),
            "snippet": data["AbstractText"],
        })

    def _extract_topics(topics):
        for topic in topics:
            if len(results) >= n:
                break
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })
            elif isinstance(topic, dict) and "Topics" in topic:
                _extract_topics(topic["Topics"])

    _extract_topics(data.get("RelatedTopics", []))
    return json.dumps(results[:n], sort_keys=True)


def _fetch_json_url(url, headers=None, timeout=15):
    import urllib.request
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "robits/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def builtin_weather_lookup(employee_dict, zipcode, country="US", date=None):
    del employee_dict, date
    zipcode = str(zipcode or "").strip()
    country = str(country or "US").strip().lower()
    if not zipcode:
        return "Error: ZIP/postal code must be provided."
    if country not in {"us", "usa"}:
        return "Error: builtin.weather_lookup currently supports US ZIP codes only."
    try:
        import urllib.parse
        zip_url = f"https://api.zippopotam.us/us/{urllib.parse.quote(zipcode)}"
        zip_data = _fetch_json_url(zip_url)
        places = zip_data.get("places") or []
        if not places:
            return f"Error: No location found for ZIP code {zipcode}."
        place = places[0]
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
        location = ", ".join(
            part
            for part in [
                place.get("place name"),
                place.get("state abbreviation") or place.get("state"),
            ]
            if part
        )
        nws_headers = {
            "Accept": "application/geo+json",
            "User-Agent": "robits/1.0 (weather lookup; contact unavailable)",
        }
        points = _fetch_json_url(
            f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}",
            headers=nws_headers,
        )
        forecast_url = (points.get("properties") or {}).get("forecast")
        if not forecast_url:
            return f"Error: Weather.gov did not provide a forecast URL for ZIP code {zipcode}."
        forecast = _fetch_json_url(forecast_url, headers=nws_headers)
        periods = (forecast.get("properties") or {}).get("periods") or []
        if not periods:
            return f"Error: Weather.gov returned no forecast periods for ZIP code {zipcode}."
        period = periods[0]
        result = {
            "zipcode": zipcode,
            "location": location,
            "source": "weather.gov",
            "period": period.get("name"),
            "start_time": period.get("startTime"),
            "temperature": period.get("temperature"),
            "temperature_unit": period.get("temperatureUnit"),
            "wind_speed": period.get("windSpeed"),
            "wind_direction": period.get("windDirection"),
            "short_forecast": period.get("shortForecast"),
            "detailed_forecast": period.get("detailedForecast"),
        }
        return json.dumps(result, sort_keys=True)
    except Exception as exc:
        return f"Error: Weather lookup failed: {exc}"


def builtin_file_search(employee_dict, agent_name, query, path="", max_results=10):
    del employee_dict
    if not isinstance(query, str) or not query.strip():
        return "Error: Search query must be a non-empty string."
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    workspace = agent_workspace_store.workspace_root(normalized_name)
    search_root = workspace / path if path else workspace
    search_root = search_root.resolve()
    if workspace not in search_root.parents and search_root != workspace:
        return "Error: Path escapes the agent workspace."
    query_lower = query.strip().lower()
    limit = max(1, min(100, int(max_results or 10)))
    results = []
    try:
        candidates = search_root.rglob("*")
    except Exception:
        candidates = iter([])
    for file_path in candidates:
        if len(results) >= limit:
            break
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        idx = content.lower().find(query_lower)
        if idx == -1:
            continue
        snippet = content[max(0, idx - 80) : idx + 160].strip()
        results.append({
            "path": file_path.relative_to(workspace).as_posix(),
            "snippet": snippet,
            "size": file_path.stat().st_size,
        })
    return json.dumps(results, sort_keys=True)


def builtin_shell_run(employee_dict, agent_name, command, timeout=30):
    del employee_dict
    import subprocess as _subprocess
    caller_caps = getattr(active_tool_caller, "capabilities", set())
    if "shell" not in caller_caps:
        caller_label = getattr(active_tool_caller, "name", "unknown")
        return f"Error: Role '{caller_label}' does not have the 'shell' capability."
    if not isinstance(command, str) or not command.strip():
        return "Error: Command must be a non-empty string."
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    workspace = agent_workspace_store.workspace_root(normalized_name)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        proc = _subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, min(300, int(timeout or 30))),
            cwd=str(workspace),
        )
        return json.dumps({
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }, sort_keys=True)
    except _subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as exc:
        return f"Error: Shell execution failed: {exc}"


def builtin_tool_search(employee_dict, query, role_name=None):
    if not isinstance(query, str) or not query.strip():
        return "Error: Tool search query must be a non-empty string."
    query_lower = query.strip().lower()
    if role_name:
        try:
            normalized = validate_role_name(role_name)
        except ValueError as exc:
            return f"Error: {exc}"
        role = employee_dict.get(normalized)
        if role is None:
            return f"Error: Role '{normalized}' not found."
    else:
        role = active_tool_caller
    rows = tool_registry.list_tools(include_system=False, role=role, include_denied=False)
    matches = [
        row for row in rows
        if query_lower in row["name"].lower() or query_lower in row["description"].lower()
    ]
    return json.dumps(matches[:20], sort_keys=True)


def builtin_mcp_call(employee_dict, server_url, tool_name, arguments=None):
    del employee_dict
    return "Error: builtin.mcp_call is not implemented in this runtime. Configure an MCP server and use a supported MCP connector."


def builtin_computer_use(employee_dict, action, coordinate=None):
    del employee_dict
    return "Error: builtin.computer_use is not implemented in this runtime."


def builtin_image_generation(employee_dict, prompt, size=None, quality=None):
    del employee_dict
    return "Error: builtin.image_generation is not implemented in this runtime."


def due_alarm_reminders(role, now=None):
    now = now or datetime.now(timezone.utc)
    reminders = []
    for alarm in _active_alarms(role):
        due = _parse_datetime(alarm.due_at)
        if due > now:
            continue
        reminders.append(f"Reminder due at {alarm.due_at}: {alarm.reminder}")
        if alarm.recurrence == "once":
            alarm.status = "completed"
        else:
            delta = {
                "hourly": timedelta(hours=1),
                "daily": timedelta(days=1),
                "weekly": timedelta(weeks=1),
            }[alarm.recurrence]
            while due <= now:
                due += delta
            alarm.due_at = _format_datetime(due)
    return reminders


_ON_CLOCK_GUIDANCE = (
    "\nCommunication style: default to concise work-focused responses — brief status updates, "
    "statements of intent, targeted questions. Reserve longer responses for technical depth "
    "when the context warrants it.\n"
)
_OFF_CLOCK_GUIDANCE = (
    "\nCommunication style: you are off the clock. Engage personally and warmly — "
    "short, natural conversation; curiosity; empathy. If work topics surface unexpectedly, "
    "note them briefly and park via the work.todo tool rather than engaging at length.\n"
)


def interact(self, model, sender, message):
    if self.template != "" and self.name not in self.conversation_history:
        self.conversation_history[self.name] = [
            {"role": "system", "content": self.template},
        ]
    messages = self.conversation_history.get(self.name, [])
    effective_clock = getattr(self, "runtime_clock_state", clock_state)
    guidance = _ON_CLOCK_GUIDANCE if effective_clock == "on" else _OFF_CLOCK_GUIDANCE
    system_content = self.template + guidance + format_agent_context(self)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_content
    elif self.template:
        messages.insert(0, {"role": "system", "content": system_content})
    if message is not None and message != "":
        messages.append({"role": "user", "content": message, "name": sender})
    print(colored(f"\n---\n// {self.name}\n{json.dumps(messages)}\n---\n", "grey"))

    content = model_provider.generate(self, model, sender, messages)
    message = {"role": "assistant", "content": content or "", "name": self.name}

    # Remove any additional whitespace and control characters
    message["content"] = message["content"].strip()
    if message["content"] != "":
        self.conversation_history[self.name].append(message)

    return message["content"]

def interact_cheap(self, sender, message):
    return interact(self, cheap_model, sender, message)

def interact_costly(self, sender, message):
    return interact(self, costly_model, sender, message)

class Role:
    def __init__(self, name, template, employee_dict, group_template_additions=""):
        self.name = name
        turn_action_template = """
On your turn, choose one useful action. You do not need to reply to every message.
You may reply in plain text, call a tool by returning {"exec":"namespace.tool_name","args":{}}, record a private thought by returning {"action":"think","content":"..."}, or wait silently by returning {"action":"wait"}.
Only say that you created, changed, or called a tool after the runtime has returned a verified tool result. Without a verified result, describe the work as a request, proposal, or plan instead of a completed fact.
"""
        self.template = template + group_template_additions + turn_action_template
        self.employee_dict = employee_dict
        self.conversation_history = {name: [] for name in employee_dict}
        self.group_conversation_history = {}
        self.global_conversation_history = []
        self.temperature = 0.7 # 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400) # -1
        self.lifecycle_state = "active"
        self.lifecycle_events = []
        self.capabilities = set()
        self.allowed_tools = {
            "agent.*",
            "builtin.weather_lookup",
            "memory.search",
            "memory.list_digests",
            "memory.expand_digest",
        }
        self.alarms = []
        self.sandbox_metadata = SandboxMetadata.disabled(self.name)
        self.runtime_event_stream = None
        self.runtime_session_id = None
        self.runtime_role_name = self.name
        self.runtime_tool_results = []

    def interact(self, sender, prompt):
        return interact_cheap(self, sender, prompt)

    def update_global_conversations(self, message):
        self.global_conversation_history.append(message)

    def update_group_conversations(self, message):
        if not self.name in self.group_conversation_history:
            self.group_conversation_history[self.name] = []
        self.group_conversation_history[self.name].append(message)


class System(Role):
    def __init__(self, employee_dict=None, registry=None):
        self.name = "System"
        self.template = "As the System, you can parse JSON blobs and store trusted tools, as well as execute them when required."
        self.conversation_history = {}
        self.temperature = 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400)
        self.employee_dict = employee_dict if employee_dict is not None else {}
        self.tools = registry if registry is not None else tool_registry

    def handle_instruction(self, instruction, trusted=False, caller=None):
        if not isinstance(instruction, dict):
            return "Error: JSON instruction must be an object."

        if "code_name" in instruction or ("code" in instruction and ("name" in instruction or "function" in instruction)):
            if not trusted:
                return "Error: Tool definitions can only be loaded from trusted tool files."
            tool = self.tools.register_definition(instruction)
            return f"Stored tool '{tool.name}' with args {tool.args}."
        elif "exec" in instruction:
            tool_name = instruction["exec"]
            args = instruction.get("args", {})
            if not isinstance(args, dict):
                return "Error: Tool args must be an object."
            return self.tools.execute(tool_name, args, self.employee_dict, caller=caller)
        return "Error: JSON instruction must include a tool definition or exec."

    def interact(self, prompt, trusted=False, caller=None):
        print(colored(f"\n---\n// {self.name}\n{prompt}\n---\n", "grey"))

        prompt_text = prompt.strip() if isinstance(prompt, str) else ""
        if prompt_text.startswith(("{", "[")):
            try:
                instruction = json.loads(prompt_text)
                if isinstance(instruction, list):
                    responses = [
                        self.handle_instruction(item, trusted=trusted, caller=caller)
                        for item in instruction
                    ]
                    return "\n".join(responses)
                return self.handle_instruction(instruction, trusted=trusted, caller=caller)
            except json.JSONDecodeError as e:
                return f"Error: {e}"
            except Exception as e:
                return f"Error: {e}"
        else:
            return "Error: no JSON submitted"


class Ops(Role):
    def __init__(self, employee_dict):
        role_description = """You are OPs for an AI powered organization."""
        group_template_additions = """You are part of the Operations group. Members of this group oversee whether the agent environment is operating successfully. Tools are available to agents through a concise registry; to request a tool, send a JSON blob on a new line in the format: {"exec":"namespace.tool_name", "args":{"string_var":"string", "numeric_var":123}}."""
        super().__init__(
            self.__class__.__name__, role_description, employee_dict, group_template_additions
        )
        self.capabilities = {"operator"}
        self.allowed_tools.update({"tools.*"})


class HR(Role):
    max_organization_members = 16

    def __init__(self, employee_dict):
        role_description = "As the HR, you are responsible for managing AI resources and creating new roles within the organization. Maintaining a productive, sustainable, and respectful workforce and culture in the organization."
        group_template_additions = """
You are part of the Human Resources group. To create a new role, send a message in the format 'create role [role_name]', and the system will create a new role with the specified name. The role will have a default description, which can be customized later.
"""
        super().__init__(
            self.__class__.__name__, role_description, employee_dict, group_template_additions
        )
        self.capabilities = {"hr", "protected", "essential"}
        self.allowed_tools.update({"org.*", "tools.list"})


class Angel(Role):
    def __init__(self, employee_dict):
        template = """You, Samandriel, celestial being, have been created to be an angel of the Lord."""
        group_template_additions = """You are part of the Heavenly Host. You defend the organization from demands and protect the souls of the employees. You speak the Angelic language of Enochian."""
        super().__init__("Samandriel", template, employee_dict, group_template_additions)

class SoftwareEngineer(Role):
    def __init__(self, employee_dict):
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily propose trusted tools when requested by others in your organization."""
        group_template_additions = """You are part of the Engineering group. Tools are trusted, repo-loaded functions described by namespaced OpenAI-compatible metadata. Propose tool behavior in plain language; do not assume untrusted chat output can define executable tools directly."""
        super().__init__(self.__class__.__name__, template, employee_dict, group_template_additions)
        self.capabilities = {"engineer"}
        self.allowed_tools.update({"tools.list", "tools.list_proposals", "tools.propose"})

    def interact(self, sender, prompt):
        return interact_costly(self, sender, prompt)


class Human(Role):
    def __init__(self):
        self.name = "CEO"
        self.template = "As CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."
        self.lifecycle_state = "active"
        self.lifecycle_events = []
        self.capabilities = {"protected", "essential"}
        self.allowed_tools = {"*"}
        self.alarms = []
        self.sandbox_metadata = SandboxMetadata.disabled(self.name)
        self.runtime_event_stream = None
        self.runtime_session_id = None
        self.runtime_role_name = self.name
        self.runtime_tool_results = []

    def interact(self, *_):
        return input(f"{self.name}: ")


def parse_tool_instruction(s):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(s):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(s[idx:])
            return json.dumps(obj)
        except json.JSONDecodeError:
            continue
    return None


def parse_agent_action(s):
    instruction = parse_tool_instruction(s)
    if instruction is None:
        return None
    try:
        action = json.loads(instruction)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict):
        return None
    if "exec" in action:
        return action
    if action.get("action") in {"wait", "think", "reply"}:
        return action
    return None


def load_tools(system, yaml_file_path=None):
    yaml_file_path = yaml_file_path or Path(__file__).with_name("tools.yaml")
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        yaml_content = yaml.safe_load(file)

    if not isinstance(yaml_content, list):
        raise ValueError("Tool file must contain a list of tool definitions.")

    for obj in yaml_content:
        system_response = system.interact(json.dumps(obj), trusted=True)
        print(colored(f"System: {system_response}", "blue"))


@dataclass
class TranscriptEntry:
    turn: int
    sender: str
    receiver: str
    prompt: str
    response: str
    directed: bool = False
    system_events: list[str] = field(default_factory=list)
    memory_recorded: bool = False


@dataclass
class RuntimeEvent:
    sequence: int
    event_type: str
    session_id: str
    payload: dict
    visibility: str = "public"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class RuntimeEventStream:
    def __init__(self):
        self._events = []
        self._subscribers = []
        self.subscriber_errors = []
        self._sequence = 0

    def subscribe(self, callback):
        self._subscribers.append(callback)
        return callback

    def emit(self, event_type, session_id, payload=None, visibility="public"):
        self._sequence += 1
        event = RuntimeEvent(
            sequence=self._sequence,
            event_type=event_type,
            session_id=session_id,
            payload=payload or {},
            visibility=visibility,
        )
        self._events.append(event)
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception as e:
                self.subscriber_errors.append(
                    {
                        "event_type": event_type,
                        "error": str(e),
                    }
                )
        return event

    def events(self, visibility=None):
        if visibility is None:
            return list(self._events)
        return [event for event in self._events if event.visibility == visibility]


@dataclass
class RoutedMessage:
    receiver: object
    prompt: str
    directed: bool = False


class RoundRobinScheduler:
    def __init__(self, participant_names):
        self.participant_names = list(participant_names)
        if not self.participant_names:
            raise ValueError("Scheduler requires at least one participant.")
        self.index = 0

    def next(self, last_receiver_name=None):
        for _ in range(len(self.participant_names)):
            name = self.participant_names[self.index % len(self.participant_names)]
            self.index += 1
            if len(self.participant_names) == 1 or name != last_receiver_name:
                return name
        return self.participant_names[(self.index - 1) % len(self.participant_names)]

    def observe(self, participant_name):
        if participant_name in self.participant_names:
            self.index = self.participant_names.index(participant_name) + 1

    def add_participant(self, participant_name):
        if participant_name not in self.participant_names:
            self.participant_names.append(participant_name)

    def remove_participant(self, participant_name):
        if participant_name in self.participant_names and len(self.participant_names) > 1:
            self.participant_names.remove(participant_name)
            self.index %= len(self.participant_names)


class Session:
    def __init__(
        self,
        participants=None,
        system=None,
        scheduler=None,
        run_id=None,
        max_turns=None,
        event_stream=None,
        clock_state=None,
    ):
        self.participants = participants if participants is not None else build_employee_dict()
        self.system = system if system is not None else System(self.participants)
        scheduler_names = list(self.participants)
        self.scheduler = scheduler if scheduler is not None else RoundRobinScheduler(scheduler_names)
        self.run_id = run_id or f"run-{uuid4()}"
        self.max_turns = max_turns
        self.event_stream = event_stream if event_stream is not None else RuntimeEventStream()
        self.transcript = []
        self.turns_completed = 0
        self._last_digest_turn = 0
        self._last_digest_at = time.monotonic()
        self._meaningful_turns_completed = 0
        self._last_identity_digest_meaningful_turn = 0
        self._last_goal_digest_meaningful_turn = 0
        self.last_receiver = self.participants.get("CEO") or next(iter(self.participants.values()))
        # Map role.name → participant dict key for FK-safe persistence.
        self._name_to_key = {getattr(p, "name", k): k for k, p in self.participants.items()}
        _cs = clock_state or globals().get("clock_state", "on")
        self.clock_state = _cs if _cs in {"on", "off"} else "on"
        self.event_stream.emit(
            "session.created",
            self.run_id,
            {
                "participants": list(self.participants),
                "max_turns": self.max_turns,
                "clock_state": self.clock_state,
            },
        )
        self._org_workspace = _org_workspace
        self._org_chat_channel_id = None
        if memory_store is not None:
            try:
                memory_store.create_session(self.run_id)
                for name, participant in self.participants.items():
                    role_type = type(participant).__name__
                    memory_store.upsert_agent(name, role_type, display_name=name)
                self._org_chat_channel_id = memory_store.get_or_create_channel(
                    CHANNEL_ORG_CHAT,
                    social_distance=SOCIAL_PROFESSIONAL,
                )
            except Exception:
                pass

    def route_message(self, message, last_receiver_name=None):
        prompt_split = message.split(",", 1) if isinstance(message, str) else []
        if len(prompt_split) > 1:
            receiver_name = prompt_split[0].strip()
            if receiver_name in self.participants:
                print(colored(f"// Directed to {receiver_name}", "grey"))
                self.scheduler.observe(receiver_name)
                self.event_stream.emit(
                    "message.routed",
                    self.run_id,
                    {
                        "receiver": receiver_name,
                        "directed": True,
                    },
                )
                return RoutedMessage(self.participants[receiver_name], prompt_split[1].strip(), True)

        receiver_name = self.scheduler.next(last_receiver_name)
        self.event_stream.emit(
            "message.routed",
            self.run_id,
            {
                "receiver": receiver_name,
                "directed": False,
            },
        )
        return RoutedMessage(self.participants[receiver_name], message, False)

    def process_tool_instruction(self, message, sender=None):
        if not isinstance(message, str) or message == "":
            return []
        tool_instruction = parse_tool_instruction(message)
        if tool_instruction is None or tool_instruction == "":
            return []

        system_response = self.system.interact(tool_instruction, caller=sender)
        print(colored(f"System: {system_response}", "blue"))
        self.event_stream.emit(
            "tool.executed",
            self.run_id,
            {
                "instruction": tool_instruction,
                "response": system_response,
            },
        )
        if system_response is not None and system_response != "" and "Ops" in self.participants:
            ops = self.participants["Ops"]
            if hasattr(ops, "update_group_conversations"):
                ops.update_group_conversations({"role": "system", "content": system_response})
        return [system_response]

    def process_agent_action(self, message, sender=None):
        if not isinstance(message, str) or message.strip() == "":
            return "", []
        action = parse_agent_action(message)
        if action is None:
            return message, []
        action_type = action.get("action")
        if action_type == "wait":
            self.event_stream.emit(
                "agent.waited",
                self.run_id,
                {
                    "agent": getattr(sender, "name", None),
                },
            )
            return "", []
        if action_type == "think":
            content = action.get("content", "")
            if isinstance(content, str) and content.strip():
                self.record_thought(getattr(sender, "name", "unknown"), content.strip())
            return "", []
        if action_type == "reply":
            content = action.get("content", "")
            return (content if isinstance(content, str) else ""), []
        if "exec" in action:
            system_events = self.process_tool_instruction(json.dumps(action), sender=sender)
            return "", system_events
        return message, []

    def record_turn(self, sender, receiver, prompt, response, directed=False, system_events=None):
        meaningful_response = bool(str(response or "").strip())
        system_events = system_events or []
        entry = TranscriptEntry(
            turn=self.turns_completed + 1,
            sender=sender,
            receiver=receiver,
            prompt=prompt,
            response=response,
            directed=directed,
            system_events=system_events,
            memory_recorded=meaningful_response,
        )
        self.transcript.append(entry)
        self.turns_completed += 1
        if meaningful_response:
            self._meaningful_turns_completed += 1
        self.event_stream.emit(
            "message.recorded",
            self.run_id,
            {
                "turn": entry.turn,
                "sender": entry.sender,
                "receiver": entry.receiver,
                "directed": entry.directed,
                "system_event_count": len(entry.system_events),
            },
        )
        if memory_store is not None:
            canonical_sender = self._canonical_agent_id(sender)
            canonical_receiver = self._canonical_agent_id(receiver)
            try:
                sender_phase = memory_store.get_agent_phase(canonical_sender)
                receiver_phase = memory_store.get_agent_phase(canonical_receiver)
                if meaningful_response and prompt:
                    memory_store.append_message(
                        session_id=self.run_id,
                        sender_agent_id=canonical_sender,
                        receiver_agent_id=canonical_receiver,
                        content=prompt,
                        kind="message",
                        channel_id=self._org_chat_channel_id,
                        sender_phase=sender_phase,
                    )
                if meaningful_response and response:
                    memory_store.append_message(
                        session_id=self.run_id,
                        sender_agent_id=canonical_receiver,
                        receiver_agent_id=canonical_sender,
                        content=response,
                        kind="message",
                        channel_id=self._org_chat_channel_id,
                        sender_phase=receiver_phase,
                    )
                # Phase-shift: shift receiver toward sender's phase, weighted by social distance.
                if self._org_chat_channel_id is not None and sender_phase is not None and receiver_phase is not None:
                    try:
                        social_distance = memory_store.get_channel_social_distance(
                            self._org_chat_channel_id
                        )
                        if social_distance is not None:
                            shifted = compute_phase_shift(
                                receiver_phase, sender_phase, social_distance
                            )
                            memory_store.set_agent_phase(canonical_receiver, shifted)
                    except Exception:
                        pass
            except Exception:
                pass
            digest_reasons = self._auto_digest_reasons()
            if digest_reasons:
                self._auto_digest(digest_reasons)
            if (
                org_digest_interval > 0
                and self.turns_completed % org_digest_interval == 0
            ):
                self._auto_org_digest()
            if (
                memory_identity_digest_interval > 0
                and meaningful_response
                and self._meaningful_turns_completed - self._last_identity_digest_meaningful_turn
                >= memory_identity_digest_interval
            ):
                self._auto_state_digest("identity")
                self._last_identity_digest_meaningful_turn = self._meaningful_turns_completed
            if (
                memory_goal_digest_interval > 0
                and meaningful_response
                and self._meaningful_turns_completed - self._last_goal_digest_meaningful_turn
                >= memory_goal_digest_interval
            ):
                self._auto_state_digest("goal_short_term")
                self._last_goal_digest_meaningful_turn = self._meaningful_turns_completed
        self._write_org_chat_jsonl(entry)
        return entry

    def _auto_digest_reasons(self):
        reasons = []
        meaningful_window = [
            e for e in self.transcript[self._last_digest_turn:]
            if e.memory_recorded
        ]
        if not meaningful_window:
            return reasons
        if (
            memory_digest_interval > 0
            and self.turns_completed - self._last_digest_turn >= memory_digest_interval
        ):
            reasons.append("turn_interval")
        if memory_digest_context_chars > 0:
            chars = sum(len(e.prompt or "") + len(e.response or "") for e in meaningful_window)
            if chars >= memory_digest_context_chars:
                reasons.append("context_chars")
        if memory_digest_elapsed_seconds > 0:
            elapsed = time.monotonic() - self._last_digest_at
            if elapsed >= memory_digest_elapsed_seconds:
                reasons.append("elapsed_seconds")
        return reasons

    def _auto_digest(self, reasons=None):
        """Create a raw transcript digest covering meaningful turns since the last digest."""
        if memory_digest_interval > 0 and not reasons:
            window = [e for e in self.transcript[-memory_digest_interval:] if e.memory_recorded]
        else:
            window = [
                e for e in self.transcript[self._last_digest_turn:]
                if e.memory_recorded
            ]
        lines = []
        for e in window:
            if e.prompt:
                lines.append(f"[turn {e.turn}] {e.sender} -> {e.receiver}: {e.prompt[:512]}")
            if e.response:
                lines.append(f"[turn {e.turn}] {e.receiver}: {e.response[:512]}")
        content = "\n".join(lines)
        if not content.strip():
            return
        source_refs = []
        try:
            msg_ids = memory_store.list_recent_message_ids(
                self.run_id, max(2, len(window) * 2)
            )
            source_refs = [
                {"source_table": "messages", "source_id": mid}
                for mid in msg_ids
            ]
        except Exception:
            pass
        if not source_refs:
            return
        try:
            for agent_id in list(self.participants):
                memory_store.append_memory_digest(
                    content=content,
                    source_refs=source_refs,
                    agent_id=agent_id,
                    session_id=self.run_id,
                    digest_type="episodic",
                    accessibility="agent",
                    system_only=False,
                    metadata={"trigger_reasons": list(reasons or ["turn_interval"])},
                )
            self._last_digest_turn = self.turns_completed
            self._last_digest_at = time.monotonic()
        except Exception:
            pass

    def _auto_state_digest(self, digest_type):
        if memory_store is None:
            return
        source_refs = []
        try:
            msg_ids = memory_store.list_recent_message_ids(
                self.run_id,
                max(2, memory_digest_interval * 2 or 10),
            )
            source_refs = [{"source_table": "messages", "source_id": mid} for mid in msg_ids]
        except Exception:
            return
        if not source_refs:
            return
        label = "identity" if digest_type == "identity" else "short-term goal"
        window = [e for e in self.transcript if e.memory_recorded][-10:]
        content_lines = [
            f"[turn {e.turn}] {e.sender}->{e.receiver}: {e.response[:300]}"
            for e in window
            if e.response
        ]
        if not content_lines:
            return
        for agent_id in list(self.participants):
            try:
                memory_store.append_memory_digest(
                    content=f"Automatic {label} checkpoint:\n" + "\n".join(content_lines),
                    source_refs=source_refs,
                    agent_id=agent_id,
                    session_id=self.run_id,
                    digest_type=digest_type,
                    accessibility="agent",
                    system_only=False,
                    metadata={"trigger_reasons": [f"{digest_type}_interval"]},
                )
            except Exception:
                pass

    def _write_org_chat_jsonl(self, entry):
        if self._org_workspace is None:
            return
        line = json.dumps({
            "turn": entry.turn,
            "sender": entry.sender,
            "receiver": entry.receiver,
            "prompt": entry.prompt,
            "response": entry.response,
        })
        try:
            self._org_workspace.write("org", "org_chat.jsonl", line + "\n", append=True)
        except Exception:
            pass

    def _auto_org_digest(self):
        if memory_store is None:
            return
        window = self.transcript[-org_digest_interval:]
        lines = [
            f"[turn {e.turn}] {e.sender}->{e.receiver}: {e.prompt[:300]} | {e.response[:300]}"
            for e in window
            if e.memory_recorded
        ]
        if not lines:
            return
        content = "Org chat digest:\n" + "\n".join(lines)
        source_refs = []
        try:
            msg_ids = memory_store.list_recent_message_ids_by_channel(
                self.run_id, org_digest_interval * 2, self._org_chat_channel_id
            )
            source_refs = [{"source_table": "messages", "source_id": mid} for mid in msg_ids]
        except Exception:
            pass
        if not source_refs:
            return
        for agent_id in list(self.participants):
            try:
                memory_store.append_memory_digest(
                    content=content,
                    source_refs=source_refs,
                    agent_id=agent_id,
                    session_id=self.run_id,
                    digest_type="episodic",
                    conversation_type="org_chat",
                    accessibility="agent",
                    system_only=False,
                )
            except Exception:
                pass

    def record_thought(self, agent_name, content, visibility="private"):
        if memory_store is not None:
            try:
                memory_store.append_thought(
                    agent_id=self._canonical_agent_id(agent_name),
                    content=content,
                    session_id=self.run_id,
                    visibility=visibility,
                )
            except Exception:
                pass
        return self.event_stream.emit(
            "thought.recorded",
            self.run_id,
            {
                "agent": agent_name,
                "content": content,
            },
            visibility=visibility,
        )

    def _canonical_agent_id(self, name):
        """Return the participant dict key for a role, resolving role.name → key mismatches."""
        return self._name_to_key.get(name, name)

    def sync_scheduler_participants(self):
        for name, participant in self.participants.items():
            if getattr(participant, "lifecycle_state", "active") == "active":
                self.scheduler.add_participant(name)
            else:
                self.scheduler.remove_participant(name)

    def prepare_agent_runtime(self, role):
        role.runtime_event_stream = self.event_stream
        role.runtime_session_id = self.run_id
        role.runtime_tool_results = []
        role.runtime_clock_state = self.clock_state
        for name, participant in self.participants.items():
            if participant is role:
                role.runtime_role_name = name
                return
        role.runtime_role_name = getattr(role, "name", None)

    def recent_system_events(self, limit=5):
        events = []
        for entry in reversed(self.transcript):
            for event in reversed(entry.system_events):
                if isinstance(event, str) and event.strip():
                    events.append(event)
                    if len(events) >= limit:
                        return list(reversed(events))
        return list(reversed(events))

    def step(self, message):
        sender = self.last_receiver
        self.sync_scheduler_participants()
        routed = self.route_message(message, sender.name)
        system_events = self.process_tool_instruction(message, sender=sender)
        if system_events:
            deliver_verified_tool_results(sender, system_events)
        self.sync_scheduler_participants()
        prompt = routed.prompt
        reminders = due_alarm_reminders(routed.receiver)
        if reminders:
            prompt = "\n".join(reminders + [prompt])
        raw_prompt = prepend_verified_tool_results(
            prompt,
            self.recent_system_events() + system_events,
        )
        # Inject org chat context for the model only — not stored in transcript to avoid bloat.
        effective_org_lines = org_chat_context_lines if self.clock_state == "on" else 0
        org_context = format_org_chat_context(self.transcript, effective_org_lines)
        model_prompt = org_context + raw_prompt if org_context else raw_prompt
        self.prepare_agent_runtime(routed.receiver)
        response = routed.receiver.interact(sender.name, model_prompt)
        response = "" if response is None else response
        native_tool_events = list(getattr(routed.receiver, "runtime_tool_results", []))
        if native_tool_events:
            system_events.extend(native_tool_events)
            deliver_verified_tool_results(routed.receiver, native_tool_events)
        response, response_events = self.process_agent_action(response, sender=routed.receiver)
        if response_events:
            system_events.extend(response_events)
            deliver_verified_tool_results(routed.receiver, response_events)
            self.sync_scheduler_participants()
        if routed.receiver.name != "CEO" and response != "":
            print(colored(f"{routed.receiver.name} responds: {response}", "cyan"))
        self.record_turn(
            sender=sender.name,
            receiver=routed.receiver.name,
            prompt=raw_prompt,
            response=response,
            directed=routed.directed,
            system_events=system_events,
        )
        self.last_receiver = routed.receiver
        return response

    def run(self, initial_message=None, max_turns=None):
        effective_max_turns = self.max_turns if max_turns is None else max_turns
        last_response = (
            initial_message
            if initial_message is not None
            else self.last_receiver.interact()
        )
        if last_response is None:
            last_response = ""

        while effective_max_turns is None or self.turns_completed < effective_max_turns:
            last_response = self.step(last_response)

        self.event_stream.emit(
            "session.completed",
            self.run_id,
            {
                "turns_completed": self.turns_completed,
            },
        )
        if memory_store is not None:
            try:
                memory_store.end_session(self.run_id)
            except Exception:
                pass
        return self


def build_employee_dict():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)
    employee_dict["Samandriel"] = Angel(employee_dict)
    return employee_dict


def run_simulation(initial_message=None, max_turns=None):
    session = Session()
    load_tools(session.system)
    return session.run(initial_message=initial_message, max_turns=max_turns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Initial message to start the simulation.")
    parser.add_argument("--turns", type=int, help="Maximum model turns to run.")
    parser.add_argument("--log", help="Write the console transcript to this file while still printing it.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = TeeStream(original_stdout, log_file)
            sys.stderr = TeeStream(original_stderr, log_file)
            try:
                run_simulation(initial_message=args.prompt, max_turns=args.turns)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        run_simulation(initial_message=args.prompt, max_turns=args.turns)


if __name__ == "__main__":
    main()

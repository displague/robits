"""Tool definition, registry, and helper utilities."""
import ast
import json
from dataclasses import dataclass

import main as _m

SAFE_TOOL_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass
class ToolDefinition:
    """Metadata and callable for a single registered tool."""

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
        """Return the tool name with dots replaced by double-underscores for API compatibility."""
        return self.name.replace(".", "__")

    def as_responses_tool(self):
        """Serialise as an OpenAI Responses API tool dict."""
        return {
            "type": "function",
            "name": self.openai_name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def as_chat_completion_tool(self):
        """Serialise as an OpenAI Chat Completions API tool dict."""
        return {
            "type": "function",
            "function": {
                "name": self.openai_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class _ToolCodeValidator(ast.NodeVisitor):
    """Detect prohibited constructs in agent-proposed tool code."""

    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        self.errors.append("import statements are not allowed in tool code; use the provided sandbox functions")

    def visit_ImportFrom(self, node):
        self.errors.append("import statements are not allowed in tool code; use the provided sandbox functions")

    def visit_FunctionDef(self, node):
        self.errors.append(
            f"Tool code must not define functions ('{node.name}'). "
            "Write a direct return statement instead. "
            f"Example: return f\"{{get_caller_name()}}: {{status}}\""
        )

    def visit_AsyncFunctionDef(self, node):
        self.errors.append(
            f"Tool code must not define async functions ('{node.name}'). "
            "Write a direct return statement instead."
        )

    def visit_ClassDef(self, node):
        self.errors.append(f"Tool code must not define classes ('{node.name}').")


def _validate_tool_code_ast(tree):
    """Run _ToolCodeValidator on an AST tree; return a list of error messages."""
    validator = _ToolCodeValidator()
    validator.visit(tree)
    return validator.errors


def role_can_use_tool(role, tool):
    """Return True if role has the required capabilities and a matching tool grant."""
    if getattr(role, "name", None) == "CEO":
        return True
    capabilities = getattr(role, "capabilities", set())
    if tool.required_capabilities and not set(tool.required_capabilities).issubset(capabilities):
        return False
    grants = getattr(role, "allowed_tools", set())
    return any(_tool_grant_matches(tool.name, grant) for grant in grants)


def _tool_grant_matches(tool_name, grant):
    """Return True if grant covers tool_name (exact, wildcard namespace, or global *)."""
    if grant == "*":
        return True
    if grant.endswith(".*"):
        return tool_name.startswith(grant[:-1])
    return tool_name == grant


def _normalize_tool_grant(tool_name):
    """Validate and canonicalise a single tool grant string; raise ValueError on unknown names."""
    if tool_name == "*":
        return tool_name
    if tool_name.endswith(".*"):
        namespace = tool_name[:-2]
        _m.tool_registry.validate_tool_name(namespace)
        return f"{namespace}.*"
    _m.tool_registry.validate_tool_name(tool_name)
    return _m.tool_registry.resolve_name(tool_name)


def _normalize_capabilities(capabilities=None):
    """Coerce capabilities to a set of stripped non-empty strings; raise ValueError on bad input."""
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
    """Coerce tool_grants to a set of validated canonical grant strings."""
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


def _coerce_json_object(value, default=None):
    """Accept a dict or a JSON string and return a dict; raise ValueError on other types."""
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


def _normalize_tool_parameters(params):
    """Coerce a flat {name: schema} property map into proper JSON Schema object format."""
    if not isinstance(params, dict) or "properties" in params or "type" in params:
        return params
    if params and all(isinstance(v, dict) and "type" in v for v in params.values()):
        return {
            "type": "object",
            "properties": params,
            "required": list(params.keys()),
        }
    return params


def _coerce_string_list(value):
    """Return value as a list of strings, splitting comma-separated strings if needed."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Value must be a list of strings.")
    return value


class ToolRegistry:
    """Central registry for tool definitions with alias resolution and access control."""

    def __init__(self):
        self._tools = {}
        self._aliases = {}

    def clear(self):
        """Remove all registered tools and aliases."""
        self._tools.clear()
        self._aliases.clear()

    def __contains__(self, name):
        """Return True if name (or its alias) is a registered tool."""
        return self.resolve_name(name) in self._tools

    def resolve_name(self, name):
        """Return the canonical tool name, resolving OpenAI double-underscore aliases."""
        return self._aliases.get(name, name)

    def get(self, name):
        """Return the ToolDefinition for name, raising KeyError if not found."""
        return self._tools[self.resolve_name(name)]

    def validate_tool_name(self, name):
        """Raise ValueError if name is not a valid dot-separated identifier string."""
        if not isinstance(name, str) or not name:
            raise ValueError("Tool name must be a non-empty string.")
        parts = name.split(".")
        if not all(part.isidentifier() for part in parts):
            raise ValueError(f"Invalid tool name: {name}")

    def compile_tool(self, name, arg_names, required_arg_names, code):
        """Compile agent-provided Python code into a callable inside the restricted sandbox."""
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
                "Role": _m.Role,
                "HR": _m.HR,
                "create_lifecycle_role": _m.create_lifecycle_role,
                "pause_lifecycle_role": _m.pause_lifecycle_role,
                "retire_lifecycle_role": _m.retire_lifecycle_role,
                "archive_lifecycle_role": _m.archive_lifecycle_role,
                "list_lifecycle_roles": _m.list_lifecycle_roles,
                "create_alarm": _m.create_alarm,
                "list_alarms": _m.list_alarms,
                "cancel_alarm": _m.cancel_alarm,
                "memory_search": _m.memory_search,
                "memory_list_digests": _m.memory_list_digests,
                "memory_expand_digest": _m.memory_expand_digest,
                "grant_tool_access": _m.grant_tool_access,
                "revoke_tool_access": _m.revoke_tool_access,
                "list_registered_tools": _m.list_registered_tools,
                "propose_tool_change": _m.propose_tool_change,
                "list_tool_proposals": _m.list_tool_proposals,
                "approve_tool_proposal": _m.approve_tool_proposal,
                "reject_tool_proposal": _m.reject_tool_proposal,
                "rollout_tool_proposal": _m.rollout_tool_proposal,
                "approve_and_rollout_proposal": _m.approve_and_rollout_proposal,
                "workspace_list": _m.workspace_list,
                "workspace_read": _m.workspace_read,
                "workspace_write": _m.workspace_write,
                "workspace_delete": _m.workspace_delete,
                "current_agent_context": _m.current_agent_context,
                "get_caller_name": lambda: _m.active_tool_caller_name or getattr(_m.active_tool_caller, "name", None) or "unknown",
                "builtin_web_search": _m.builtin_web_search,
                "builtin_file_search": _m.builtin_file_search,
                "builtin_shell_run": _m.builtin_shell_run,
                "builtin_tool_search": _m.builtin_tool_search,
                "builtin_mcp_call": _m.builtin_mcp_call,
                "builtin_computer_use": _m.builtin_computer_use,
                "builtin_image_generation": _m.builtin_image_generation,
                "org_chat_read": _m.org_chat_read,
                "work_todo_add": _m.work_todo_add,
                "agent_think": _m.agent_think,
                "agent_wait": _m.agent_wait,
            },
            local_dict,
        )
        return local_dict[function_name]

    def normalize_definition(self, instruction):
        """Normalise a raw instruction dict into a canonical ToolDefinition-ready form."""
        if "function" in instruction:
            function = instruction["function"]
            name = function["name"]
            description = function.get("description", "")
            parameters = function.get("parameters", {"type": "object", "properties": {}})
            code = instruction["code"]
            raw_aliases = instruction.get("aliases", ())
        elif "name" in instruction:
            name = instruction["name"]
            description = instruction.get("description", "")
            parameters = instruction.get("parameters", {"type": "object", "properties": {}})
            code = instruction["code"]
            raw_aliases = instruction.get("aliases", ())
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
            raw_aliases = instruction.get("aliases", ())

        if isinstance(raw_aliases, str):
            raise ValueError("Tool aliases must be a list of strings, not a bare string.")
        if not isinstance(raw_aliases, (list, tuple, set)):
            raise ValueError("Tool aliases must be a list of strings.")
        aliases = tuple(raw_aliases)

        namespace = instruction.get("namespace") or name.split(".", 1)[0]
        raw_capabilities = instruction.get("required_capabilities", ())
        if isinstance(raw_capabilities, str):
            raise ValueError("Tool required_capabilities must be a list of strings, not a bare string.")
        if not isinstance(raw_capabilities, (list, tuple, set)):
            raise ValueError("Tool required_capabilities must be a list of strings.")
        required_capabilities = tuple(raw_capabilities)
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
        """Compile and register a new tool from an instruction dict; raise on duplicates."""
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

    def replace_definition(self, instruction):
        """Replace a non-system tool in-place (for update proposals)."""
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
        existing = self._tools.get(name)
        if existing is None:
            raise ValueError(f"Tool '{name}' does not exist; use register_definition to create it.")
        if existing.system_tool:
            raise ValueError(f"System tool '{name}' cannot be replaced.")
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
        old_tool = self._tools.get(name)
        effective_aliases = aliases if aliases else (old_tool.aliases if old_tool else ())
        for alias in effective_aliases:
            self.validate_tool_name(alias)
            existing_target = self._aliases.get(alias)
            if existing_target is not None and existing_target != name:
                raise ValueError(f"Tool alias '{alias}' already exists.")
            if alias in self._tools and alias != name:
                raise ValueError(f"Tool alias '{alias}' already exists.")
        if old_tool is not None:
            stale = [k for k, v in self._aliases.items() if v == name]
            for k in stale:
                del self._aliases[k]
        tool = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            args=tool.args,
            required_args=tool.required_args,
            func=tool.func,
            aliases=effective_aliases,
            namespace=tool.namespace,
            required_capabilities=tool.required_capabilities,
            owner_capability=tool.owner_capability,
            system_tool=tool.system_tool,
            grantable=tool.grantable,
        )
        self._tools[name] = tool
        for alias in effective_aliases:
            self._aliases[alias] = name
        openai_name = tool.openai_name
        if openai_name != name:
            existing_target = self._aliases.get(openai_name)
            if existing_target is not None and existing_target != name:
                raise ValueError(f"Tool openai_name '{openai_name}' conflicts with an existing alias.")
            self._aliases[openai_name] = name
        return tool

    def list_tools(self, include_system=True, role=None, include_denied=True):
        """Return a list of tool metadata dicts, optionally filtered by role access."""
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
        """Return all ToolDefinitions that role is permitted to use."""
        return [tool for tool in self._tools.values() if role_can_use_tool(role, tool)]

    def execute(self, name, args, employee_dict, caller=None):
        """Look up and invoke a tool, setting active_tool_caller for the duration."""
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
        previous_caller = _m.active_tool_caller
        previous_caller_name = _m.active_tool_caller_name
        _m.active_tool_caller = caller
        _m.active_tool_caller_name = None
        if caller is not None:
            for employee_name, employee in employee_dict.items():
                if employee is caller:
                    _m.active_tool_caller_name = employee_name
                    break
            if _m.active_tool_caller_name is None:
                _m.active_tool_caller_name = getattr(caller, "name", None)
        try:
            result = tool.func(employee_dict=employee_dict, **args)
        finally:
            _m.active_tool_caller = previous_caller
            _m.active_tool_caller_name = previous_caller_name
        return f"Executed tool '{resolved_name}' with args {args}. Result: {result}"

    def as_responses_tools(self, role=None):
        """Return all (or role-filtered) tools serialised for the Responses API."""
        tools = self._tools.values() if role is None else self.tools_for_role(role)
        return [tool.as_responses_tool() for tool in tools]

    def as_chat_completion_tools(self, role=None):
        """Return all (or role-filtered) tools serialised for the Chat Completions API."""
        tools = self._tools.values() if role is None else self.tools_for_role(role)
        return [tool.as_chat_completion_tool() for tool in tools]

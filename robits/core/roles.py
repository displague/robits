"""Role hierarchy and interaction functions."""
import json
import random
import yaml
from pathlib import Path

import main as _m
from robits.runtime.sandbox import SandboxMetadata
from robits.core.context import format_agent_context

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
    effective_clock = getattr(self, "runtime_clock_state", _m.clock_state)
    guidance = _ON_CLOCK_GUIDANCE if effective_clock == "on" else _OFF_CLOCK_GUIDANCE
    system_content = self.template + guidance + format_agent_context(self)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_content
    elif self.template:
        messages.insert(0, {"role": "system", "content": system_content})
    if message is not None and message != "":
        messages.append({"role": "user", "content": message, "name": sender})
    from termcolor import colored
    print(colored(f"\n---\n// {self.name}\n{json.dumps(messages)}\n---\n", "grey"))

    content = _m.model_provider.generate(self, model, sender, messages)
    message = {"role": "assistant", "content": content or "", "name": self.name}

    message["content"] = message["content"].strip()
    if message["content"] != "":
        self.conversation_history[self.name].append(message)

    return message["content"]


def interact_cheap(self, sender, message):
    return interact(self, _m.cheap_model, sender, message)


def interact_costly(self, sender, message):
    return interact(self, _m.costly_model, sender, message)


class Role:
    def __init__(self, name, template, employee_dict, group_template_additions=""):
        self.name = name
        turn_action_template = """
On your turn, choose one useful action. You do not need to reply to every message.
Only say that you created, changed, or called a tool after the runtime has returned a verified tool result. Without a verified result, describe the work as a request, proposal, or plan instead of a completed fact.
"""
        self.template = template + group_template_additions + turn_action_template
        self.employee_dict = employee_dict
        self.conversation_history = {name: [] for name in employee_dict}
        self.group_conversation_history = {}
        self.global_conversation_history = []
        self.temperature = 0.7
        self.max_tokens = random.randint(250, 400)
        self.lifecycle_state = "active"
        self.lifecycle_events = []
        self.capabilities = set()
        self.allowed_tools = {"agent.*", "memory.search", "memory.list_digests", "memory.expand_digest"}
        self.alarms = []
        self.sandbox_metadata = SandboxMetadata.disabled(self.name)
        self.runtime_event_stream = None
        self.runtime_session_id = None
        self.runtime_role_name = self.name
        self.runtime_tool_results = []
        self.waiting_until = None
        self.wait_started_turn = None
        self.wait_clock_state = None

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
        self.tools = registry if registry is not None else _m.tool_registry

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
        from termcolor import colored
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
        group_template_additions = """You are part of the Operations group. Members of this group oversee whether the agent environment is operating successfully."""
        super().__init__(
            self.__class__.__name__, role_description, employee_dict, group_template_additions
        )
        self.capabilities = {"operator"}
        self.allowed_tools.update({"tools.*"})


class HR(Role):
    max_organization_members = 16

    def __init__(self, employee_dict):
        role_description = "As the HR, you are responsible for managing AI resources and creating new roles within the organization. Maintaining a productive, sustainable, and respectful workforce and culture in the organization."
        group_template_additions = "\nYou are part of the Human Resources group.\n"
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
        group_template_additions = """You are part of the Engineering group. Tools are trusted functions described by namespaced OpenAI-compatible metadata. You can propose tools with working Python code using the `code` field in tools.propose. IMPORTANT CODE RULES: write the body as a sequence of simple statements ending with a single `return` — do NOT define nested functions or lambdas. The sandbox has no imports and limited builtins (bool, int, str, list, dict, len, max, min, range, sum). Available globals: get_caller_name() → calling agent's name; workspace_write/read/list → agent files; memory_search → stored context; work_todo_add → task tracking. Example for a status tool with a `status` parameter: `return f"{get_caller_name()}: {status}"`. Parameters must use JSON Schema format: {"type":"object","properties":{"status":{"type":"string"}},"required":["status"]}. Approved proposals with code are registered and activated at rollout without a restart."""
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

    _SLASH_COMMANDS = {
        "/quit": "Stop the session immediately.",
        "/exit": "Stop the session immediately.",
        "/help": "List available slash commands.",
    }

    def interact(self, *_):
        try:
            text = input(f"{self.name}: ")
        except EOFError:
            return ""
        stripped = text.strip()
        if stripped.startswith("/"):
            cmd = stripped.split()[0].lower()
            if cmd in ("/quit", "/exit"):
                raise SystemExit(0)
            if cmd == "/help":
                lines = ["Available slash commands:"]
                for name, desc in self._SLASH_COMMANDS.items():
                    lines.append(f"  {name}  — {desc}")
                print("\n".join(lines))
                return ""
            print(f"Unknown command: {cmd}. Type /help for available commands.")
            return ""
        return text


def build_employee_dict():
    employee_dict = {}
    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)
    employee_dict["Samandriel"] = Angel(employee_dict)
    return employee_dict


def parse_tool_instruction(s):
    import json as _json
    decoder = _json.JSONDecoder()
    for idx, char in enumerate(s):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(s[idx:])
            return _json.dumps(obj)
        except _json.JSONDecodeError:
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
    from termcolor import colored
    yaml_file_path = yaml_file_path or Path(__file__).parents[2] / "tools.yaml"
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        yaml_content = yaml.safe_load(file)

    if not isinstance(yaml_content, list):
        raise ValueError("Tool file must contain a list of tool definitions.")

    for obj in yaml_content:
        system_response = system.interact(json.dumps(obj), trusted=True)
        print(colored(f"System: {system_response}", "blue"))

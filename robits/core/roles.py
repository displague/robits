"""Role hierarchy and interaction functions."""
import json
import random
import yaml
from pathlib import Path

from robits.core.config import _config as _m
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
    """Send message to the model and return the assistant's response text."""
    preprompt_work = getattr(self, "preprompt_work", self.template)
    preprompt_personal = getattr(self, "preprompt_personal", "")
    
    # Retrieve metadata and circadian phase from database
    metadata = {}
    phase = 0.5
    if _m.memory_store is not None:
        try:
            if hasattr(_m.memory_store, "get_agent_metadata"):
                metadata = _m.memory_store.get_agent_metadata(self.name)
            db_phase = _m.memory_store.get_agent_phase(self.name)
            if db_phase is not None:
                phase = db_phase
        except Exception:
            pass

    preprompt_work = metadata.get("preprompt_work") or preprompt_work
    preprompt_personal = metadata.get("preprompt_personal") or preprompt_personal
    
    work_adjustment = metadata.get("work_adjustment")
    work_adjustment_turns = metadata.get("work_adjustment_turns")
    personal_adjustment = metadata.get("personal_adjustment")
    personal_adjustment_turns = metadata.get("personal_adjustment_turns")
    
    import math
    phase_val = max(0.0, min(1.0, float(phase)))
    phase_weight = math.sin(math.pi * phase_val)
    effective_clock = getattr(self, "runtime_clock_state", None) or _m.clock_state
    
    if effective_clock == "on":
        work_weight = 0.5 + 0.5 * phase_weight
    elif effective_clock == "off":
        work_weight = 0.5 * (1.0 - phase_weight)
    else:  # "break"
        work_weight = 0.3 + 0.4 * phase_weight
    personal_weight = 1.0 - work_weight
    
    preprompt_lines = []
    preprompt_lines.append("=== Persona Circadian Rhythm & State ===")
    preprompt_lines.append(f"Active clock state: {effective_clock}")
    preprompt_lines.append(f"Circadian phase: {phase_val:.2f}")
    preprompt_lines.append(f"Persona blend weighting: Work Mode = {work_weight * 100:.0f}%, Personal Mode = {personal_weight * 100:.0f}%")
    preprompt_lines.append("")
    
    preprompt_lines.append(f"--- Work Persona Instructions (Weight: {work_weight * 100:.0f}%) ---")
    preprompt_lines.append(preprompt_work or "")
    if work_adjustment:
        preprompt_lines.append(f"[Active Work Adjustment (Expires in {work_adjustment_turns} turns): {work_adjustment}]")
    preprompt_lines.append("")
    
    preprompt_lines.append(f"--- Personal Persona Instructions (Weight: {personal_weight * 100:.0f}%) ---")
    preprompt_lines.append(preprompt_personal or "")
    if personal_adjustment:
        preprompt_lines.append(f"[Active Personal Adjustment (Expires in {personal_adjustment_turns} turns): {personal_adjustment}]")
    preprompt_lines.append("")
    
    dynamic_preprompt = "\n".join(preprompt_lines)
    turn_action_template = getattr(self, "turn_action_template", "")
    
    guidance = _ON_CLOCK_GUIDANCE if effective_clock == "on" else _OFF_CLOCK_GUIDANCE
    system_content = dynamic_preprompt + turn_action_template + guidance + format_agent_context(self)

    if self.name not in self.conversation_history:
        self.conversation_history[self.name] = [
            {"role": "system", "content": system_content},
        ]
    messages = self.conversation_history.get(self.name, [])
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_content
    else:
        messages.insert(0, {"role": "system", "content": system_content})
    if message is not None and message != "":
        messages.append({"role": "user", "content": message, "name": sender})

    send_messages = messages
    if _m.max_context_tokens > 0 and len(messages) > 1:
        budget = _m.max_context_tokens * 4  # ~4 chars per token
        system_msg = messages[0]
        kept = []
        chars = len(json.dumps(system_msg))
        for msg in reversed(messages[1:]):
            msg_chars = len(json.dumps(msg))
            if chars + msg_chars > budget and kept:
                break
            kept.insert(0, msg)
            chars += msg_chars
        send_messages = [system_msg] + kept

    from robits.core.io import get_logger
    import json as _json
    _ctx_chars = sum(len(_json.dumps(m)) for m in send_messages)
    _last_user = next((m["content"][:200] for m in reversed(send_messages) if m.get("role") == "user"), "")
    get_logger().write_event("agent_prompt", agent=self.name, message_count=len(send_messages), context_chars=_ctx_chars, last_user=_last_user)

    content = _m.model_provider.generate(self, model, sender, send_messages)
    message = {"role": "assistant", "content": content or "", "name": self.name}

    message["content"] = message["content"].strip()
    if message["content"] != "":
        self.conversation_history[self.name].append(message)

    return message["content"]


def interact_cheap(self, sender, message):
    """Interact using the configured cheap model."""
    return interact(self, _m.cheap_model, sender, message)


def interact_costly(self, sender, message):
    """Interact using the configured costly model."""
    return interact(self, _m.costly_model, sender, message)


class Role:
    """Base class for all agent roles in the organisation."""

    def __init__(self, name, template, employee_dict, group_template_additions=""):
        self.name = name
        self.turn_action_template = """
On your turn, choose one useful action. You do not need to reply to every message.
Only say that you created, changed, or called a tool after the runtime has returned a verified tool result. Without a verified result, describe the work as a request, proposal, or plan instead of a completed fact.
When recalling past events, experiences, or recent work — even if the answer seems available in context — use memory.search (for specific queries), memory.list_digests, or memory.expand_digest to retrieve the full picture before answering.
"""
        self.preprompt_work = template + group_template_additions
        self.preprompt_personal = f"You are {name} off the clock. Relax and engage in warm, personal conversation. Focus on hobbies, interests, and building social relationships."
        self.template = self.preprompt_work + self.turn_action_template
        self.employee_dict = employee_dict
        self.conversation_history = {name: [] for name in employee_dict}
        self.group_conversation_history = {}
        self.global_conversation_history = []
        self.temperature = 0.7
        self.base_temperature = self.temperature
        self.top_p = 0.9
        self.base_top_p = self.top_p
        self.max_tokens = random.randint(250, 400)
        self.lifecycle_state = "active"
        self.lifecycle_events = []
        self.capabilities = set()
        self.allowed_tools = {
            "agent.*",
            "builtin.web_search",
            "builtin.url_fetch",
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
        self.waiting_until = None
        self.wait_started_turn = None
        self.wait_clock_state = None

    def interact(self, sender, prompt):
        """Respond to a prompt from sender using the cheap model."""
        return interact_cheap(self, sender, prompt)

    def update_global_conversations(self, message):
        """Append a message to the global conversation history."""
        self.global_conversation_history.append(message)

    def update_group_conversations(self, message):
        """Append a message to this role's group conversation history."""
        if not self.name in self.group_conversation_history:
            self.group_conversation_history[self.name] = []
        self.group_conversation_history[self.name].append(message)


class System(Role):
    """Special role that parses and executes JSON tool instructions."""

    def __init__(self, employee_dict=None, registry=None):
        self.name = "System"
        self.template = "As the System, you can parse JSON blobs and store trusted tools, as well as execute them when required."
        self.conversation_history = {}
        self.temperature = 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400)
        self.employee_dict = employee_dict if employee_dict is not None else {}
        self.tools = registry if registry is not None else _m.tool_registry

    def handle_instruction(self, instruction, trusted=False, caller=None):
        """Dispatch a single parsed instruction dict to register a tool or execute one."""
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
        """Parse a JSON prompt and execute or register the described instruction."""
        from robits.core.io import get_logger
        get_logger().write_event("system_instruction", agent=self.name, content=prompt)

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
    """Operations role — oversees the agent environment and holds operator capability."""

    def __init__(self, employee_dict):
        role_description = """You are OPs for an AI powered organization."""
        group_template_additions = """You are part of the Operations group. Members of this group oversee whether the agent environment is operating successfully."""
        super().__init__(
            self.__class__.__name__, role_description, employee_dict, group_template_additions
        )
        self.capabilities = {"operator"}
        self.allowed_tools.update({"tools.*"})


class HR(Role):
    """Human Resources role — manages workforce, creates roles, and holds hr capability."""

    max_organization_members = 16

    def __init__(self, employee_dict):
        role_description = "As the HR, you are responsible for managing AI resources and creating new roles within the organization. Maintaining a productive, sustainable, and respectful workforce and culture in the organization."
        group_template_additions = "\nYou are part of the Human Resources group.\n"
        super().__init__(
            self.__class__.__name__, role_description, employee_dict, group_template_additions
        )
        self.capabilities = {"hr", "protected", "essential"}
        self.allowed_tools.update({"org.*", "tools.list", "tools.list_proposals"})


class Angel(Role):
    """Samandriel — a heavenly guardian role that speaks Enochian and protects employees."""

    def __init__(self, employee_dict):
        template = """You, Samandriel, celestial being, have been created to be an angel of the Lord."""
        group_template_additions = """You are part of the Heavenly Host. You defend the organization from demands and protect the souls of the employees. You speak the Angelic language of Enochian."""
        super().__init__("Samandriel", template, employee_dict, group_template_additions)


class SoftwareEngineer(Role):
    """Software Engineer role — proposes and implements trusted tools via the costly model."""

    def __init__(self, employee_dict):
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily propose trusted tools when requested by others in your organization."""
        group_template_additions = """You are part of the Engineering group. Tools are trusted functions described by namespaced OpenAI-compatible metadata. You can propose tools with working Python code using the `code` field in tools.propose. IMPORTANT CODE RULES: write the body as a sequence of simple statements ending with a single `return` — do NOT define nested functions or classes (lambdas and inline expressions are fine). The sandbox has no imports; available builtins: abs, bool, dict, enumerate, float, int, len, list, max, min, range, round, set, sorted, str, sum, tuple, zip. Available globals: get_caller_name() → calling agent's name; workspace_write/workspace_read/workspace_list/workspace_delete → agent files; org_chat_read → recent org chat; memory_search/memory_list_digests/memory_expand_digest → stored context; work_todo_add → task tracking; agent_think/agent_wait/agent_spawn → agent control; current_agent_context → runtime context; grant_tool_access/revoke_tool_access → permissions; propose_tool_change/list_tool_proposals/approve_tool_proposal/reject_tool_proposal/rollout_tool_proposal/approve_and_rollout_proposal → proposals; list_registered_tools → tool catalogue; create_lifecycle_role/pause_lifecycle_role/retire_lifecycle_role/archive_lifecycle_role/list_lifecycle_roles → org management; create_alarm/list_alarms/cancel_alarm → scheduling; builtin_web_search/builtin_url_fetch/builtin_file_search/builtin_shell_run/builtin_tool_search/builtin_mcp_call/builtin_computer_use/builtin_image_generation → external integrations. Example: `return f"{get_caller_name()}: {status}"`. Parameters use JSON Schema: {"type":"object","properties":{"status":{"type":"string"}},"required":["status"]}. Approved proposals with code are registered and activated at rollout without a restart."""
        super().__init__(self.__class__.__name__, template, employee_dict, group_template_additions)
        self.capabilities = {"engineer", "shell"}
        self.allowed_tools.update({"tools.list", "tools.list_proposals", "tools.propose", "builtin.shell_run"})

    def interact(self, sender, prompt):
        """Respond using the costly model to handle complex engineering tasks."""
        return interact_costly(self, sender, prompt)


class Human(Role):
    """CEO role — a human-in-the-loop participant that reads from stdin."""

    def __init__(self, employee_dict=None):
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
        """Read a line from stdin and handle slash commands; raise SystemExit on /quit."""
        try:
            text = input(f"{self.name}: ")
        except EOFError:
            return ""
        from robits.core.io import get_logger
        get_logger().write_event("ceo_input", content=text)
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


_ROLE_CLASS_MAP = {
    "SoftwareEngineer": SoftwareEngineer,
    "SE": SoftwareEngineer,
    "Ops": Ops,
    "HR": HR,
    "Angel": Angel,
    "Samandriel": Angel,
    "CEO": Human,
    "Human": Human,
}


def build_employee_dict(persona_map=None):
    """Construct and return the set of organisation participants.

    When ``persona_map`` is provided ({username: {role, full_name, entries}}),
    only the listed personas are instantiated — the persona file defines the
    complete cast.  CEO (Human) is added automatically if not listed.
    Without a persona_map the default cast (CEO, Ops, SE, HR, Samandriel) is used.
    """
    employee_dict = {}

    if persona_map:
        for username, info in persona_map.items():
            if not isinstance(info, dict):
                continue
            role_key = info.get("role", username)
            full_name = info.get("full_name", username)
            RoleClass = _ROLE_CLASS_MAP.get(role_key)
            if RoleClass is None:
                import logging
                logging.warning(
                    "robits: persona %r references unknown role %r — skipping",
                    username, role_key,
                )
                continue
            instance = RoleClass(employee_dict)
            instance.name = username
            instance.role_key = role_key
            instance.full_name = full_name
            parts = full_name.split(None, 1)
            instance.first_name = parts[0] if parts else username
            instance.last_name = parts[1] if len(parts) > 1 else ""
            if info.get("preprompt_work"):
                instance.preprompt_work = info["preprompt_work"]
            if info.get("preprompt_personal"):
                instance.preprompt_personal = info["preprompt_personal"]
            employee_dict[username] = instance
        if not any(isinstance(v, Human) for v in employee_dict.values()):
            employee_dict["CEO"] = Human()
        # Backfill conversation_history so every role knows about every peer,
        # regardless of instantiation order.
        all_keys = set(employee_dict)
        for role in employee_dict.values():
            ch = getattr(role, "conversation_history", None)
            if isinstance(ch, dict):
                for key in all_keys:
                    ch.setdefault(key, [])
    else:
        employee_dict["CEO"] = Human()
        employee_dict["Ops"] = Ops(employee_dict)
        employee_dict["SE"] = SoftwareEngineer(employee_dict)
        employee_dict["HR"] = HR(employee_dict)
        employee_dict["Samandriel"] = Angel(employee_dict)

    return employee_dict


def parse_tool_instruction(s):
    """Scan s for a JSON object or array and return it as a normalised JSON string, or None."""
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
    """Parse s as an agent action dict (exec / wait / think / reply); return None if unrecognised."""
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
    """Load tool definitions from a YAML file and register them with the System role."""
    from termcolor import colored
    yaml_file_path = yaml_file_path or Path(__file__).parents[2] / "tools.yaml"
    with open(yaml_file_path, "r", encoding="utf-8") as file:
        yaml_content = yaml.safe_load(file)

    if not isinstance(yaml_content, list):
        raise ValueError("Tool file must contain a list of tool definitions.")

    for obj in yaml_content:
        system_response = system.interact(json.dumps(obj), trusted=True)
        print(colored(f"System: {system_response}", "blue"))

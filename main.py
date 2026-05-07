#!/usr/bin/env python3
from dataclasses import dataclass, field
from openai import OpenAI

import random
import time
import os
import json
import re
import yaml
from datetime import datetime
from termcolor import colored
import argparse
from pathlib import Path
from uuid import uuid4


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
            {"__builtins__": SAFE_TOOL_BUILTINS, "Role": Role, "HR": HR},
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

        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("Tool parameters must contain object properties and required list.")
        required_arg_names = []
        for arg_name in required:
            if not isinstance(arg_name, str) or arg_name not in properties:
                raise ValueError("Tool required args must be named properties.")
            required_arg_names.append(arg_name)
        arg_names = list(properties)

        return name, description, parameters, arg_names, required_arg_names, code, aliases

    def register_definition(self, instruction):
        name, description, parameters, arg_names, required_arg_names, code, aliases = self.normalize_definition(instruction)
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

    def execute(self, name, args, employee_dict):
        try:
            self.validate_tool_name(name)
        except ValueError as e:
            return f"Error: {e}"
        resolved_name = self.resolve_name(name)
        if resolved_name not in self._tools:
            return f"Error: Tool '{name}' not found."
        tool = self._tools[resolved_name]
        missing_args = [arg_name for arg_name in tool.required_args if arg_name not in args]
        if missing_args:
            return f"Error: Missing args for tool '{name}': {missing_args}"
        unexpected_args = [arg_name for arg_name in args if arg_name not in tool.args]
        if unexpected_args:
            return f"Error: Unexpected args for tool '{name}': {unexpected_args}"
        result = tool.func(employee_dict=employee_dict, **args)
        return f"Executed tool '{resolved_name}' with args {args}. Result: {result}"

    def as_responses_tools(self):
        return [tool.as_responses_tool() for tool in self._tools.values()]

    def as_chat_completion_tools(self):
        return [tool.as_chat_completion_tool() for tool in self._tools.values()]


tool_registry = ToolRegistry()


def interact(self, model, sender, message):
    if self.template != "" and self.name not in self.conversation_history:
        self.conversation_history[self.name] = [
            {"role": "system", "content": self.template},
        ]
    messages = self.conversation_history.get(self.name, [])
    if message is not None and message != "":
        messages.append({"role": "user", "content": message, "name": sender})
    print(colored(f"\n---\n// {self.name}\n{json.dumps(messages)}\n---\n", "grey"))

    # Start a streaming session
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=self.max_tokens,
        n=1,
        temperature=self.temperature,
        user=f"robits_{self.name}",
        stream=True,
    )
    message = {"role": "assistant", "content": "", "name": self.name}
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            message["content"] += chunk.choices[0].delta.content

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
        self.template = template + group_template_additions
        self.conversation_history = {name: [] for name in employee_dict}
        self.group_conversation_history = {}
        self.global_conversation_history = []
        self.temperature = 0.7 # 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400) # -1

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

    def handle_instruction(self, instruction, trusted=False):
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
            return self.tools.execute(tool_name, args, self.employee_dict)
        return "Error: JSON instruction must include a tool definition or exec."

    def interact(self, prompt, trusted=False):
        print(colored(f"\n---\n// {self.name}\n{prompt}\n---\n", "grey"))

        prompt_text = prompt.strip() if isinstance(prompt, str) else ""
        if prompt_text.startswith(("{", "[")):
            try:
                instruction = json.loads(prompt_text)
                if isinstance(instruction, list):
                    responses = [
                        self.handle_instruction(item, trusted=trusted)
                        for item in instruction
                    ]
                    return "\n".join(responses)
                return self.handle_instruction(instruction, trusted=trusted)
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

    def interact(self, sender, prompt):
        return interact_costly(self, sender, prompt)


class Human(Role):
    def __init__(self):
        self.name = "CEO"
        self.template = "As CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."

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


class Session:
    def __init__(self, participants=None, system=None, scheduler=None, run_id=None, max_turns=None):
        self.participants = participants if participants is not None else build_employee_dict()
        self.system = system if system is not None else System(self.participants)
        scheduler_names = list(self.participants)
        self.scheduler = scheduler if scheduler is not None else RoundRobinScheduler(scheduler_names)
        self.run_id = run_id or f"run-{uuid4()}"
        self.max_turns = max_turns
        self.transcript = []
        self.turns_completed = 0
        self.last_receiver = self.participants.get("CEO") or next(iter(self.participants.values()))

    def route_message(self, message, last_receiver_name=None):
        prompt_split = message.split(",", 1) if isinstance(message, str) else []
        if len(prompt_split) > 1:
            receiver_name = prompt_split[0].strip()
            if receiver_name in self.participants:
                print(colored(f"// Directed to {receiver_name}", "grey"))
                self.scheduler.observe(receiver_name)
                return RoutedMessage(self.participants[receiver_name], prompt_split[1].strip(), True)

        receiver_name = self.scheduler.next(last_receiver_name)
        return RoutedMessage(self.participants[receiver_name], message, False)

    def process_tool_instruction(self, message):
        if not isinstance(message, str) or message == "":
            return []
        tool_instruction = parse_tool_instruction(message)
        if tool_instruction is None or tool_instruction == "":
            return []

        system_response = self.system.interact(tool_instruction)
        print(colored(f"System: {system_response}", "blue"))
        if system_response is not None and system_response != "" and "Ops" in self.participants:
            ops = self.participants["Ops"]
            if hasattr(ops, "update_group_conversations"):
                ops.update_group_conversations({"role": "system", "content": system_response})
        return [system_response]

    def record_turn(self, sender, receiver, prompt, response, directed=False, system_events=None):
        entry = TranscriptEntry(
            turn=self.turns_completed + 1,
            sender=sender,
            receiver=receiver,
            prompt=prompt,
            response=response,
            directed=directed,
            system_events=system_events or [],
        )
        self.transcript.append(entry)
        self.turns_completed += 1
        return entry

    def sync_scheduler_participants(self):
        for name in self.participants:
            self.scheduler.add_participant(name)

    def step(self, message):
        sender = self.last_receiver
        routed = self.route_message(message, sender.name)
        system_events = self.process_tool_instruction(message)
        self.sync_scheduler_participants()
        response = routed.receiver.interact(sender.name, routed.prompt)
        response = "" if response is None else response
        if routed.receiver.name != "CEO" and response != "":
            print(colored(f"{routed.receiver.name} responds: {response}", "cyan"))
        self.record_turn(
            sender=sender.name,
            receiver=routed.receiver.name,
            prompt=routed.prompt,
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_simulation(initial_message=args.prompt, max_turns=args.turns)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
escape_codes = {}
SAFE_ESCAPE_BUILTINS = {
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
    def __init__(self, employee_dict=None):
        self.name = "System"
        self.template = "As the System, you can parse JSON blobs and store escape codes, as well as execute them when required."
        self.conversation_history = {}
        self.temperature = 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400)
        self.employee_dict = employee_dict if employee_dict is not None else {}

    def compile_escape_code(self, code_name, arg_names, code):
        if not code_name.isidentifier():
            raise ValueError(f"Invalid escape code name: {code_name}")
        for arg_name in arg_names:
            if not arg_name.isidentifier():
                raise ValueError(f"Invalid escape code argument name: {arg_name}")

        parameters = arg_names + ["employee_dict"]
        indented_code = "\n".join(
            f"    {line}" if line.strip() else "" for line in code.splitlines()
        )
        if not indented_code:
            indented_code = "    pass"
        function_name = f"_escape_{code_name}"
        function_source = f"def {function_name}({', '.join(parameters)}):\n{indented_code}"
        local_dict = {}
        exec(
            function_source,
            {"__builtins__": SAFE_ESCAPE_BUILTINS, "Role": Role, "HR": HR},
            local_dict,
        )
        return local_dict[function_name]

    def handle_instruction(self, instruction, trusted=False):
        if not isinstance(instruction, dict):
            return "Error: JSON instruction must be an object."

        if "code_name" in instruction:
            if not trusted:
                return "Error: Escape code definitions can only be loaded from trusted preload files."
            code_name = instruction["code_name"]
            args = instruction.get("args")
            if not isinstance(args, list):
                return "Error: Escape code args must be a list of objects with name fields."
            arg_names = []
            for arg in args:
                if not isinstance(arg, dict) or not isinstance(arg.get("name"), str):
                    return "Error: Escape code args must be a list of objects with name fields."
                arg_names.append(arg["name"])
            escape_codes[code_name] = {
                "args": arg_names,
                "func": self.compile_escape_code(
                    code_name, arg_names, instruction["code"]
                ),
            }
            return f"Stored escape code '{code_name}' with args {arg_names}."
        elif "exec" in instruction:
            code_name = instruction["exec"]
            args = instruction.get("args", {})
            if not isinstance(args, dict):
                return "Error: Escape code args must be an object."
            if code_name in escape_codes:
                escape_code = escape_codes[code_name]
                missing_args = [
                    arg_name
                    for arg_name in escape_code["args"]
                    if arg_name not in args
                ]
                if missing_args:
                    return f"Error: Missing args for escape code '{code_name}': {missing_args}"
                result = escape_code["func"](
                    **args,
                    employee_dict=self.employee_dict,
                )
                return f"Executed escape code '{code_name}' with args {args}. Result: {result}"

            return f"Error: Escape code '{code_name}' not found."
        return "Error: JSON instruction must include code_name or exec."

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
        group_template_additions = """You are part of the Operations group.Members of this group recognize when other organization members need escape codes executed and send the appropriate escape code. You can also request new code from the Software Engineer who will create escape codes. To execute code, you send a JSON blob on a new line. You will recognize when other organization members need escape codes executed and will send the appropriate escape code, the format is a JSON object: {"exec":"escape_code_name_here", "args":{"string_var":"string", "numeric_var":123}})"""
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
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily create escape codes when requested by others in your organization."""
        group_template_additions = """You are part of the Engineering group. To create an escape code, on a newline write a JSON object with the fields: code_name, args, and code. The code_name is the name of the escape code, the args are a list of objects which name the parameter the code will receive, the code must be a valid python function that accepts the parameters. For example, to create an escape code that fetches a URL, you may post on a newline a JSON blob like {"code_name": "add_100", "args":[{"name":"value"}], "code":"return 100+value"}"""
        super().__init__(self.__class__.__name__, template, employee_dict, group_template_additions)

    def interact(self, sender, prompt):
        return interact_costly(self, sender, prompt)


class Human(Role):
    def __init__(self):
        self.name = "CEO"
        self.template = "As CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."

    def interact(self, *_):
        return input(f"{self.name}: ")


def parse_escape_code(s):
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


def preload(system, yaml_file_path=None):
    yaml_file_path = yaml_file_path or Path(__file__).with_name("preload.yaml")
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Add escape codes for HR
    for obj in yaml_content:
        system_response = system.interact(json.dumps(obj), trusted=True)
        print(colored(f"System: {system_response}", "blue"))


def build_employee_dict():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)
    employee_dict["Samandriel"] = Angel(employee_dict)
    return employee_dict


def run_simulation(initial_message=None, max_turns=None):
    employee_dict = build_employee_dict()
    system = System(employee_dict)
    preload(system)
    last_receiver = employee_dict["CEO"]
    receiver = last_receiver
    last_response = (
        initial_message
        if initial_message is not None
        else receiver.interact()
    )
    turns = 0

    while max_turns is None or turns < max_turns:
        prompt_split = last_response.split(",", 1)
        if len(prompt_split) > 1 and prompt_split[0] in employee_dict:
            print(colored(f"// Directed to {prompt_split[0]}", "grey"))
            receiver = employee_dict[prompt_split[0].strip()]
            last_response = prompt_split[1].strip()
        else:
            while True:
                new_receiver = employee_dict[random.choice(list(employee_dict))]
                if new_receiver != last_receiver:
                    receiver = new_receiver
                    break

        escape_code = parse_escape_code(last_response)
        if escape_code is not None and escape_code != "":
            try:
                system_response = system.interact(escape_code)
                print(colored(f"System: {system_response}", "blue"))
                if system_response is not None and system_response != "":
                    employee_dict["Ops"].update_group_conversations({"role": "system", "content": system_response})

            except json.JSONDecodeError:
                # Return the original response if the JSON blob cannot be processed
                if system_response.startswith("Error:"):
                    print(colored(f"System responds: {system_response}", "red"))
                    continue

        response = receiver.interact(last_receiver.name, last_response)
        if response is None or response == "":
            continue
        if receiver.name != "CEO":
            print(colored(f"{receiver.name} responds: {response}", "cyan"))
        last_response = response
        last_receiver = receiver
        turns += 1


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

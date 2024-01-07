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
import sys

client = OpenAI(
    organization=os.environ.get("OPENAI_ORG", ""),
    api_key=os.environ.get("OPENAI_API_KEY", "bogus"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"),
)
costly_model = "gpt-3.5-turbo"
cheap_model = "text-davinci-002"
escape_codes = {}


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
            message["content"] +=chunk.choices[0].delta.content

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
    def __init__(self):
        self.name = "System"
        self.template = "As the System, you can parse JSON blobs and store escape codes, as well as execute them when required."
        self.conversation_history = {}
        self.temperature = 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400)

    def interact(self, prompt):
        print(colored(f"\n---\n// {self.name}\n{prompt}\n---\n", "grey"))

        if prompt is not None and prompt.startswith("{"):
            try:
                instruction = json.loads(prompt)

                if "code_name" in instruction:
                    escape_codes[instruction["code_name"]] = {
                        "args": instruction["args"],
                        "code": instruction["code"],
                    }
                    return f"Stored escape code '{instruction['code_name']}' with args {instruction['args']}."
                elif "exec" in instruction:
                    code_name = instruction["exec"]
                    args = instruction["args"]
                    if code_name in escape_codes:
                        code = escape_codes[code_name]["code"]
                        exec(code, args)
                        return f"Executed escape code '{code_name}' with args {args}."

                    else:
                        return f"Error: Escape code '{code_name}' not found."
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
    start_idx = next((idx for idx, c in enumerate(s) if c in "{["), None)
    if start_idx is None:
        return None  # or some other appropriate value
    s = s[start_idx:]
    try:
        return json.dumps(json.loads(s))
    except json.JSONDecodeError as e:
        try:
            return json.dumps(json.loads(s[:e.pos]))
        except json.JSONDecodeError:
            return None


def preload(system):
    yaml_file_path = 'preload.yaml'
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # Add escape codes for HR
    for obj in yaml_content:
        system_response = system.interact(json.dumps(obj))
        print(colored(f"System: {system_response}", "blue"))


def main():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)
    employee_dict["Samandriel"] = Angel(employee_dict)

    system = System()
    preload(system)
    last_receiver = employee_dict["CEO"]
    receiver = last_receiver
    last_response = receiver.interact() # "Welcome to the organization. Start a conversation."

    while True:
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
        # time.sleep(3)


if __name__ == "__main__":
    main()

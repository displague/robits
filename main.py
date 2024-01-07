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


def interact_costly(self, message):
    if not self.name in self.conversation_history:
        self.conversation_history[self.name] = [
            {"role": "system", "content": self.template},
        ]
    messages = self.conversation_history.get(self.name, [])

    messages.append({"role": "user", "content": message})
    print(colored(f"\n---\n// {self.name}\n{json.dumps(messages)}\n---\n", "grey"))

    response = client.chat.completions.create(
        model=costly_model,
        messages=messages,
        max_tokens=self.max_tokens,
        n=1,
        temperature=0.8,
    )
    generated_text = response.choices[0].message.content.strip()
    self.conversation_history[self.name].append(
        {"role": "assistant", "content": generated_text}
    )

    return generated_text


def interact_cheap(self, message):
    if not self.name in self.conversation_history:
        self.conversation_history[self.name] = []

    self.conversation_history[self.name].append(message)
    full_prompt = (
        "\n".join(self.group_conversation_history)
        + "\n"
        + "\n".join(self.conversation_history[self.name])
    )
    print(colored(f"\n---\n// {self.name}\n{full_prompt}\n---\n", "grey"))
    response = client.chat.completions.create(
        model=cheap_model,
        prompt=full_prompt,
        max_tokens=self.max_tokens,
        n=1,
        stop=None,
        temperature=0.8,
    )
    print(response)
    response = response.choices[0].text.strip()
    self.conversation_history[self.name].append(response)
    return response


class Role:
    def __init__(self, name, template, employee_dict, group_template_additions=""):
        self.name = name
        self.template = template + group_template_additions
        self.conversation_history = {name: [] for name in employee_dict}
        self.group_conversation_history = []
        self.global_conversation_history = []
        self.temperature = 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400)

    def interact(self, prompt):
        return interact_cheap(self, prompt)

    def update_global_conversations(self, message):
        self.global_conversation_history.append(message)

    def update_group_conversations(self, message):
        self.group_conversation_history.append(message)


class System(Role):
    def __init__(self):
        self.name = "System"
        self.template = "As the System, you can parse JSON blobs and store escape codes, as well as execute them when required."
        self.conversation_history = {}
        self.temperature = 0.1 * random.randint(1, 9)
        self.max_tokens = random.randint(250, 400)

    def interact(self, prompt):
        print(colored(f"\n---\n// {self.name}\n{prompt}\n---\n", "grey"))

        if prompt.startswith("{"):
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
        else:
            return "Error: no JSON submitted"


class Ops(Role):
    def __init__(self, employee_dict):
        role_description = """You are OPs for an AI powered organization."""
        group_template_additions = """You are part of the Operations group.Members of this group recognize when other organization members need escape codes executed and send the appropriate escape code. You can also request new code from the Software Engineer who will create escape codes. To execute code, you send a JSON blob on a new line. You will recognize when other organization members need escape codes executed and will send the appropriate escape code, the format is a JSON object: {"exec":"escape_code_name_here", "args":{"string_var":"string", "numeric_var":123}})"""
        super().__init__(
            "Ops", role_description, employee_dict, group_template_additions
        )


class HR(Role):
    max_organization_members = 16

    def __init__(self, employee_dict):
        self.employee_dict = employee_dict
        role_description = "As the HR, you are responsible for managing AI resources and creating new roles within the organization. Maintaining a productive, sustainable, and respectful workforce and culture in the organization."
        group_template_additions = """
You are part of the Human Resources group. To create a new role, send a message in the format 'create role [role_name]', and the system will create a new role with the specified name. The role will have a default description, which can be customized later.
"""
        super().__init__(
            "HR", role_description, employee_dict, group_template_additions
        )

    def interact(self, prompt):
        return super().interact(prompt)

    def parse_employee_creation_message(self, message):
        try:
            role_name = message.split("named")[1].split("with")[0].strip()
            role_description = message.split("with the description")[1].strip()
            return role_name, role_description
        except IndexError:
            raise ValueError("Invalid employee creation syntax.")


class SoftwareEngineer(Role):
    def __init__(self, employee_dict):
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily create escape codes when requested by others in your organization."""
        group_template_additions = """You are part of the Engineering group. To create an escape code, on a newline write a JSON object with the fields: code_name, args, and code. The code_name is the name of the escape code, the args are a list of objects which name the parameter the code will receive, the code must be a valid python function that accepts the parameters. For example, to create an escape code that fetches a URL, you may post on a newline a JSON blob like {"code_name": "add_100", "args":[{"name":"value"}], "code":"return 100+value"}"""
        super().__init__("SE", template, employee_dict, group_template_additions)
        self.conversation_history = {name: [] for name in employee_dict}

    def interact(self, prompt):
        return interact_costly(self, prompt)


class Human(Role):
    def __init__(self):
        self.name = "CEO"
        self.template = "As CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."

    def interact(self, prompt):
        return input("Enter message: ")


def parse_escape_code(response):
    # Check if the response contains a JSON blob
    json_blob = ""
    if "{" in response and "}" in response:
        lines = response.split("\n")
        for line in lines:
            if line.strip().startswith("{"):
                json_blob += line.strip()
            elif json_blob:
                json_blob += " " + line.strip()
                if line.strip().endswith("}"):
                    break

    return json_blob


def main():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    # employee_dict["HR"] = HR(employee_dict)

    system = System()

    # Add escape codes for HR
    system.interact(
        json.dumps(
            {
                "code_name": "create_role",
                "args": [{"name": "role_name"}, {"name": "role_description"}],
                "code": """
role_name = args["role_name"]
role_description = args["role_description"]
hr = employee_dict["HR"]

if role_name not in hr.employee_dict:
    if len(hr.employee_dict) < HR.max_organization_members:
        new_role = Role(
            role_name,
            role_description,
            hr.employee_dict
        )
        hr.employee_dict[role_name] = new_role
        response = f"Created a new role: {role_name}"
    else:
        response = f"Error: The organization has reached its maximum size of {HR.max_organization_members} members."
else:
    response = f"Error: Role '{role_name}' already exists."
""",
            }
        )
    )

    last_receiver = employee_dict["CEO"]
    receiver = last_receiver
    last_response = "Welcome to the organization. Start a conversation."

    while True:
        prompt_split = last_response.split(",", 1)
        if len(prompt_split) == 2 and prompt_split[0] in employee_dict:
            receiver = employee_dict[prompt_split[0].strip()]
            last_response = prompt_split[1].strip()
        else:
            while True:
                new_receiver = employee_dict[random.choice(list(employee_dict))]
                if new_receiver != last_receiver:
                    receiver = new_receiver
                    break

        escape_code = parse_escape_code(last_response)
        if escape_code != "":
            try:
                system_response = system.interact(escape_code)
                print(colored(f"System: {system_response}", "blue"))
                employee_dict["Ops"].update_group_conversations(
                    f"System: {system_response}"
                )
            except json.JSONDecodeError:
                # Return the original response if the JSON blob cannot be processed
                if system_response.startswith("Error:"):
                    print(colored(f"System responds: {system_response}", "red"))
                    continue

        response = receiver.interact(
            f"{last_receiver.name}: {last_response}\n{receiver.name}: "
        )
        if receiver.name != "CEO":
            print(colored(f"{receiver.name} responds: {response}", "cyan"))
        last_response = response
        last_receiver = receiver
        time.sleep(3)


if __name__ == "__main__":
    main()

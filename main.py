import openai
import random
import time
import os
import json
import re
import yaml
from datetime import datetime
from termcolor import colored
import sys

try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("Error: OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

escape_codes = {}


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
        if not self.name in self.conversation_history:
            self.conversation_history[self.name] = []

        self.conversation_history[self.name].append(prompt)
        full_prompt = "\n".join(self.conversation_history[self.name])

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=full_prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=None,
            temperature=self.temperature,
        )
        response = response.choices[0].text.strip()

        self.conversation_history[self.name].append(response)
        return response

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
                        # Warning: Executing arbitrary code can be dangerous.
                        # Make sure you trust the source of the code.
                        namespace = {"args": args}
                        exec(code, namespace)
                        return f"Executed escape code '{code_name}' with args {args}."

                    else:
                        return f"Error: Escape code '{code_name}' not found."
            except json.JSONDecodeError as e:
                return f"Error: {e}"
        else:
            return super().interact(prompt)


class Ops(Role):
    def __init__(self, employee_dict):
        role_description = """As the Ops role, you recognize when other organization members need escape codes executed and send the appropriate escape code. You can also request new code from the Software Engineer. To execute code, you send a JSON blob on a new line. You will recognize when other organization members need escape codes executed and will send the appropriate escape code, the format is a JSON object: {"exec":"escape_code_name_here", "args":[]})"""
        group_template_additions = "You are part of the Operations group."
        super().__init__(
            "Ops", role_description, employee_dict, group_template_additions
        )

    def interact(self, prompt):
        if prompt.lower().startswith("request code"):
            return f'{self.name}: {{"exec": "request_code", "args": []}}'
        else:
            return super().interact(prompt)


class HR(Role):
    max_organization_members = 16

    def __init__(self, employee_dict):
        self.employee_dict = employee_dict
        role_description = "As the HR, you are responsible for managing AI resources and creating new roles within the organization. Maintaining a productive, sustainable, and respectful workforce and culture in the organization."
        group_template_additions = "You are part of the Human Resources group."
        super().__init__(
            "HR", role_description, employee_dict, group_template_additions
        )

    def interact(self, prompt):
        role_creation_pattern = r"(?i)(create|add)\s+(a\s+)?(new\s+)?role\s+(.+)"
        match = re.match(role_creation_pattern, prompt)
        if match:
            role_name = match.group(4).strip()
            if role_name not in self.employee_dict:
                if len(self.employee_dict) < HR.max_organization_members:
                    new_role = Role(
                        role_name,
                        f"As a {role_name}, you are responsible for performing tasks related to the {role_name} role.",
                    )
                    self.employee_dict[role_name] = new_role
                    return f"Created a new role: {role_name}"
                else:
                    return f"Error: The organization has reached its maximum size of {HR.max_organization_members} members."
            else:
                return f"Error: Role '{role_name}' already exists."
        else:
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
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily create escape codes when requested by others in your organization. To create an escape code, on a newline write a JSON object with the fields: code_name, args, and code. The code_name is the name of the escape code, the args are a list of objects which name the parameter the code will receive, the code must be a valid python function that accepts the parameters. For example, to create an exit code that fetches a URL, you may post on a newline a JSON blob like {"code_name": "crawl", "args":[{"name":"url"}], "code":"python code here that fetches that URL and returns the output."}. The code, code_name, and args should be tailored to the specific need."""
        group_template_additions = "You are part of the Engineering group."
        super().__init__("SE", template, employee_dict, group_template_additions)


class Human(Role):
    def __init__(self):
        self.name = "CEO"
        self.template = "As CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."

    def interact(self, prompt):
        return input("Enter message: ")


def main():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)

    system = System()

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

        system_response = system.interact(last_response)
        if system_response.startswith("Error:"):
            print(colored(f"System responds: {system_response}", "red"))
            continue

        response = receiver.interact(
            f"{last_receiver.name}: {last_response}\n{receiver}: "
        )
        if receiver.name != "CEO":
            print(colored(f"{receiver.name} responds: {response}", "cyan"))
        last_response = response
        last_receiver = receiver
        time.sleep(3)


if __name__ == "__main__":
    main()

import openai
import random
import time
import os
import json
import re

openai.api_key = os.environ['OPENAI_API_KEY']
escape_codes = {}
class Role:
    def __init__(self, name, template):
        self.name = name
        self.template = template + f"""You are part of an organization, {self.name}, and you have to collaborate with other members. Keep introductions to a minimum and get to work. If you are not sure what you should be doing, ask. If answers are not productive, get creative. Employees can create new organization members using the prompt format "Create a new role *role_name*". The current members of the org are HR, SE, Ops, and CEO. If you need help achieving the company goals, you can request a new organization member with that prompt. The conversation follows, each exchange is 1:1. Each individual can recall their own previous conversations.\n---"""
        self.conversation_history = [f"{self.template}\n"]

    def interact(self, prompt):
        self.conversation_history.append(prompt)
        full_prompt = "\n".join(self.conversation_history)
        
        response = openai.Completion.create(
            engine="text-davinci-002",  # Replace with the GPT-4 model once available
            prompt=full_prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.8,
        )
        # print(response)
        response = response.choices[0].text.strip()

        self.conversation_history.append(response)
        return response

class System(Role):
    def __init__(self):
        role_description = "As the System, you can parse JSON blobs and store escape codes, as well as execute them when required."
        super().__init__("System", role_description)

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
    def __init__(self):
        role_description = """As the Ops role, you recognize when other organization members need escape codes executed and send the appropriate escape code. You can also request new code from the Software Engineer. To execute code, you send a JSON blob on a new line. You will recognize when other organization members need escape codes executed and will send the appropriate escape code, the format is a JSON object: {"exec":"escape_code_name_here", "args":[]})"""
        super().__init__("Ops", role_description)

    def interact(self, prompt):
        if prompt.lower().startswith("request code"):
            return f"{self.name}: {{\"exec\": \"request_code\", \"args\": []}}"
        else:
            return super().interact(prompt)

class HR(Role):
    def __init__(self):
        role_description = "As the HR, you are responsible for managing AI resources and creating new roles within the organization. Maintaining a productive, sustainable, and respectful workforce and culture in the organization."
        super().__init__("HR", role_description)

    def interact(self, prompt):
        role_creation_pattern = r"(?i)(create|add)\s+(a\s+)?(new\s+)?role\s+(.+)"
        match = re.match(role_creation_pattern, prompt)
        if match:
            role_name = match.group(4).strip()
            if role_name not in employee_dict:
                new_role = Role(role_name, f"As a {role_name}, you are responsible for performing tasks related to the {role_name} role.")
                employee_dict[role_name] = new_role
                return f"Created a new role: {role_name}"
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
    def __init__(self):
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You can also create escape codes when requested by others in your organization. To create an escape code, on a newline write a JSON object with fields, code_name, args, and code. The code_name is the name of the escape code, the args are a list of parameters the code will receive, code must be valid python function that accepts the parameters. For example, a create an exit code that fetches a URL, you may post on a newline a JSON blob like {"code_name": "crawl", "args":[{"name":"url"}], "code":"python code here that fetches that URL and returns the output."}"""
        super().__init__("SE", template)

    def interact(self, prompt):
        if prompt.startswith("!new"):
            # Create a new escape code
            try:
                code_name, args, code = self.parse_new_escape_code_message(prompt)
                escape_codes[code_name] = {"args": args, "code": code}
                return f"Created a new escape code '{code_name}' with args '{args}'."
            except ValueError as e:
                return f"Error: {e}"
        else:
            return super().interact(prompt)

    def parse_new_escape_code_message(self, message):
        try:
            command, code_name, args = message.split("(", 1)[0].split(maxsplit=2)
            code = message.split("(", 1)[1].rsplit(")", 1)[0]
            return code_name.strip(), args.strip(), code.strip()
        except IndexError:
            raise ValueError("Invalid escape code creation syntax.")

class Human(Role):
    def __init__(self):
        self.name = "CEO"
        self.template = "As CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."

        pass

    def interact(self, prompt):
        return input("Enter message: ")

def main():
    global employee_dict
    employee_dict = {
        "CEO": Human(),
        "HR": HR(),
        "SE": SoftwareEngineer(),
        "Ops": Ops(),
    }
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
                if new_receiver != receiver:
                    receiver = new_receiver
                    break

        system_response = system.interact(last_response)
        if system_response.startswith("Error:"):
            print(f"System responds: {system_response}")
            continue

        # print(f"Message to {receiver.name}: {last_response}")
        response = receiver.interact(f"{last_receiver.name}: {last_response}\n{receiver}: ")
        if receiver.name != "CEO":
            print(f"{receiver.name} responds: {response}")
        last_response = response
        last_receiver = receiver
        time.sleep(3)

if __name__ == "__main__":
    main()

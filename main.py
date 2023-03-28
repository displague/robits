import openai
import random
import time
import os

openai.api_key = os.environ['OPENAI_API_KEY']
escape_codes = {}
class Role:
    def __init__(self, name, template):
        self.name = name
        self.template = template + f"You are part of an organization, {self.name}, and you have to collaborate with other members. Keep introductions to a minimum and get to work. If you are not sure, ask. If answers are not productive, get creative. The conversation follows.\n---"
        self.conversation_history = [f"{self.template}\n"]

    def interact(self, prompt):
        self.conversation_history.append(prompt)
        full_prompt = "\n".join(self.conversation_history)
        
        response = openai.Completion.create(
            engine="text-davinci-002",  # Replace with the GPT-4 model once available
            prompt=full_prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.8,
        )
        # print(response)
        response = response.choices[0].text.strip()

        self.conversation_history.append(response)
        return response

class CEO(Role):
    def __init__(self):
        template = "As the CEO, you are responsible for making high-level decisions and setting the overall direction of the organization."
        super().__init__("CEO", template)

class MarketingDirector(Role):
    def __init__(self):
        template = "As the Marketing Director, you are responsible for planning and executing marketing strategies to promote the organization's products or services."
        super().__init__("Marketing Director", template)

class ProgramManager(Role):
    def __init__(self):
        template = "As the Program Manager, you are responsible for managing multiple projects and ensuring their successful delivery."
        super().__init__("Program Manager", template)

class PersonalAssistant(Role):
    def __init__(self):
        template = "As a Personal Assistant, you are responsible for providing administrative and personal support to your assigned employee."
        super().__init__("Personal Assistant", template)

class Therapist(Role):
    def __init__(self):
        template = "As a Therapist, you are responsible for providing mental health support to your assigned employee."
        super().__init__("Therapist", template)

class Friend(Role):
    def __init__(self):
        template = "As a Friend, you are responsible for providing emotional support, companionship, and advice to your assigned employee."
        super().__init__("Friend", template)

class FamilyMember(Role):
    def __init__(self):
        template = "As a Family Member, you are responsible for providing emotional support and sharing family-related matters with your assigned employee."
        super().__init__("Family Member", template)

class SoftwareEngineer(Role):
    def __init__(self):
        template = "As a Software Engineer, you are responsible for designing, developing, and maintaining software applications. You can also create escape codes when requested."
        super().__init__("Software Engineer", template)

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

class Employee:
    def __init__(self, primary_role, supporting_roles):
        self.primary_role = primary_role
        self.supporting_roles = supporting_roles

    def interact(self, message):
        if isinstance(self.primary_role, SoftwareEngineer) and message.startswith("!new"):
            # Handle escape code creation
            pass
        else:
            response = self.primary_role.interact(message)
            print(f"Response: {response}")
            pass

class Human(Role):
    def __init__(self):
        pass

    def interact(self, prompt):
        return input("Enter message: ")

def main():
    employees = [
        CEO(),
        MarketingDirector(),
        ProgramManager(),
        SoftwareEngineer(),
        #PersonalAssistant(),
        #Therapist(),
        #Friend(),
        #FamilyMember()
    ]

    receiver = Human()
    last_receiver = "Human"
    last_response = "Welcome to the organization. Start a conversation."
    
    while True:
        receiver = random.choice(employees)

        # print(f"Message to {receiver.name}: {last_response}")
        response = receiver.interact(f"{last_receiver}: {last_response}\n{receiver}: ")
        print(f"{receiver.name} responds: {response}")
        last_response = response
        last_receiver = receiver
        time.sleep(3)

if __name__ == "__main__":
    main()

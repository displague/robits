from .role import Role
from interact import Interact


class SoftwareEngineer(Role):
    def __init__(self, employee_dict):
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily create tools when requested by others in your organization."""
        group_template_additions = """You are part of the Engineering group. To create an escape code, on a newline write a JSON object with the fields: code_name, args, and code. The code_name is the name of the escape code, the args are a list of objects which name the parameter the code will receive, the code must be a valid python function that accepts the parameters. For example, to create an escape code that fetches a URL, you may post on a newline a JSON blob in the pseudo-format of: {name, description, parameters: {type:object, properties: {property_name_as_key: {type, description}}, required: ["property_name_as_key"], code}. For example, {"name": "add_100", "description": "Add 100 to an integer", "parameters":[{"type": "object", "properties": {"value": {"type": "integer", "description":"The value to add 100 to"}}}], "required":["value"], "code":"return 100+int(args.get('value', '0'))"}"""
        super().__init__(
            self.__class__.__name__, template, employee_dict, group_template_additions
        )

    def interact(self, sender, prompt):
        return Interact.interact_costly(self, sender, prompt)

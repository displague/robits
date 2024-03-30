from .role import Role
class SoftwareEngineer(Role):
    def __init__(self, employee_dict):
        template = """As a Software Engineer (SE), you are responsible for designing, developing, and maintaining software applications. You primarily create tools when requested by others in your organization."""
        group_template_additions = """You are part of the Engineering group. To create an tool, on a newline write a JSON object with the fields: code_name, args, and code. The code_name is the name of the tool, the args are a list of objects which name the parameter the code will receive, the code must be a valid python function that accepts the parameters. For example, to create an tool that fetches a URL, you may post on a newline a JSON blob like {"type":"function","function":{"name": "add_100", "description":"Add 100 to supplied value", "parameters":{"type":"object","properties":{"value":{"type":"int}"value"}], "code":"return 100+value"}}"""
        super().__init__(
            self.__class__.__name__, template, employee_dict, group_template_additions
        )

    def interact(self, sender, prompt):
        return interact_costly(self, sender, prompt)


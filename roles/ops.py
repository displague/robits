from .role import Role

class Ops(Role):
    def __init__(self, employee_dict):
        role_description = """You are OPs for an AI powered organization."""
        group_template_additions = """You are part of the Operations group.Members of this group recognize when other organization members need tools executed and send the appropriate tool. You can also request new code from the Software Engineer who will create tools. To execute code, you send a JSON blob on a new line. You will recognize when other organization members need tools executed and will send the appropriate tool, the format is a JSON object: {"exec":"tool_name_here", "args":{"string_var":"string", "numeric_var":123}})"""
        super().__init__(
            self.__class__.__name__,
            role_description,
            employee_dict,
            group_template_additions,
        )

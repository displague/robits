import json
from typing import Callable, Optional

class MCPService:
    def __init__(self):
        self.tools = {}
        self._role_creator: Optional[Callable[[str, str], str]] = None
        self.register_preloaded_tools()

    def set_role_creator(self, role_creator: Callable[[str, str], str]) -> None:
        self._role_creator = role_creator

    def register_tool(self, tool_name, tool_function, description, parameters):
        if isinstance(tool_function, str):
            # If the tool_function is a string of code, execute it to define the function
            exec(tool_function, globals())
            # The function should be defined in the globals, so we can retrieve it
            tool_function = globals()[tool_name]

        self.tools[tool_name] = {
            "function": tool_function,
            "description": description,
            "parameters": parameters
        }

    def execute_tool(self, tool_name, **kwargs):
        if tool_name in self.tools:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**kwargs)
        else:
            return f"Error: Tool '{tool_name}' not found."

    def get_tool_spec(self, tool_name):
        if tool_name in self.tools:
            return {
                "type": "function",
                "name": tool_name,
                "description": self.tools[tool_name]["description"],
                "parameters": self.tools[tool_name]["parameters"]
            }
        else:
            return None

    def get_all_tool_specs(self):
        return [self.get_tool_spec(tool_name) for tool_name in self.tools]

    def register_preloaded_tools(self):
        def create_role(role_name, role_description):
            if self._role_creator is None:
                return (
                    "Error: Role creation is not available until the organization "
                    "runtime has been initialized."
                )
            return self._role_creator(role_name, role_description)

        self.register_tool(
            "create_role",
            create_role,
            "Create a new role",
            {
                "type": "object",
                "properties": {
                    "role_name": {
                        "type": "string",
                        "description": "The name of the role"
                    },
                    "role_description": {
                        "type": "string",
                        "description": "The description of the role"
                    }
                },
                "required": ["role_name", "role_description"]
            }
        )

        def get_current_weather(location, unit="fahrenheit"):
            if "tokyo" in location.lower():
                return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
            elif "san francisco" in location.lower():
                return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
            elif "paris" in location.lower():
                return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
            else:
                return json.dumps({"location": location, "temperature": "unknown"})

        self.register_tool(
            "get_current_weather",
            get_current_weather,
            "Get the current weather in a given location",
            {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        )

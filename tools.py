import yaml
import json

class Tools:
    tools = {}

    @staticmethod
    def parse(s):
        """
        Parses a string `s` and returns a JSON string representation of the parsed data.

        Args:
            s (str): The string to be parsed.

        Returns:
            str: A JSON string representation of the parsed data.

        Raises:
            None

        """
        start_idx = next((idx for idx, c in enumerate(s) if c in "{["), None)
        if start_idx is None:
            return None  # or some other appropriate value
        s = s[start_idx:]
        try:
            return json.dumps(json.loads(s))
        except json.JSONDecodeError as e:
            try:
                return json.dumps(json.loads(s[: e.pos]))
            except json.JSONDecodeError:
                return None

    @staticmethod
    def load(yaml_file_path):
        """
        Load tools from a YAML file.

        Args:
            yaml_file_path (str): The path to the YAML file.

        Returns:
            dict: A dictionary of tools, where the keys are the tool names and the values are the tool objects.
        """
        with open(yaml_file_path, "r") as file:
            tools = yaml.safe_load(file)
        return {tool["function"]["name"]: tool for tool in tools}

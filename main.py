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
from roles import Human, Ops, SoftwareEngineer, HR, Angel, System

client = OpenAI(
    organization=os.environ.get("OPENAI_ORG", ""),
    api_key=os.environ.get("OPENAI_API_KEY", "bogus"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com"),
)
costly_model = "gpt-3.5-turbo"
cheap_model = "text-davinci-002"
tools = {}


def interact(self, model, sender, message):
    if self.template != "" and self.name not in self.conversation_history:
        self.conversation_history[self.name] = [
            {"role": "system", "content": self.template},
        ]
    messages = self.conversation_history.get(self.name, [])
    if message is not None and message != "":
        messages.append({"role": "user", "content": message, "name": sender})
    print(colored(f"\n---\n// {self.name}\n{json.dumps(messages)}\n---\n", "grey"))

    do_stream = True
    # Start a streaming session
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=self.max_tokens,
        n=1,
        temperature=self.temperature,
        user=f"robits_{self.name}",
        tools=[{k: v for k, v in tool.items() if k != "code"} for tool in tools],
        tool_choice="auto",
        stream=do_stream,
    )
    message = {"role": "assistant", "content": "", "name": self.name}
    if do_stream:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                message["content"] += chunk.choices[0].delta.content
        message["content"] = message["content"].strip()
    else:
        message = response.choices[0].message

    # Remove any additional whitespace and control characters
    if message["content"] != "":
        self.conversation_history[self.name].append(message)

    messages.append(message)
    if "tools_calls" in message:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool = tools[tool_name]
            code = tool["code"]
            response = exec(code, globals(), tool_args)
            tool_message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": response,
            }
            messages.append(tool_message)
            self.conversation_history[self.name].append(tool_message)

        # Get second response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
            n=1,
            temperature=self.temperature,
            user=f"robits_{self.name}",
            stream=True,
        )
        message = {"role": "assistant", "content": "", "name": self.name}
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                message["content"] += chunk.choices[0].delta.content
        message["content"] = message["content"].strip()

    return message["content"]


def interact_cheap(self, sender, message):
    return interact(self, cheap_model, sender, message)


def interact_costly(self, sender, message):
    return interact(self, costly_model, sender, message)


def parse_tool(s):
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


def load_tools(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        tools = yaml.safe_load(file)
    return {tool["function"]["name"]: tool for tool in tools}


def main():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)
    employee_dict["Samandriel"] = Angel(employee_dict)

    tools = load_tools("preload.yaml")
    system = System(tools)
    last_receiver = employee_dict["CEO"]
    receiver = last_receiver
    last_response = (
        receiver.interact()
    )  # "Welcome to the organization. Start a conversation."

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

        tool = parse_tool(last_response)
        if tool is not None and tool != "":
            try:
                system_response = system.interact(tool)
                print(colored(f"System: {system_response}", "blue"))
                if system_response is not None and system_response != "":
                    employee_dict["Ops"].update_group_conversations(
                        {"role": "system", "content": system_response}
                    )

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

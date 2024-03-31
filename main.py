#!/usr/bin/env python3

import random
import time
import re
from datetime import datetime
import sys
from roles import Human, Ops, SoftwareEngineer, HR, Angel, System
from tools import Tools


def main():
    employee_dict = {}

    employee_dict["CEO"] = Human()
    employee_dict["Ops"] = Ops(employee_dict)
    employee_dict["SE"] = SoftwareEngineer(employee_dict)
    employee_dict["HR"] = HR(employee_dict)
    employee_dict["Samandriel"] = Angel(employee_dict)

    tools = Tools.load("preload.yaml")
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

        tool = Tools.parse(last_response)
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

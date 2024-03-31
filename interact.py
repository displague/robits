from termcolor import colored
import json
from openai import OpenAI
import os

client = OpenAI(
    organization=os.environ.get("OPENAI_ORG", ""),
    api_key=os.environ.get("OPENAI_API_KEY", "bogus"),
    base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
)

costly_model = "gpt-4-turbo-preview"
cheap_model = "gpt-3.5-turbo"


class Interact:
    @staticmethod
    def interact(role, model, sender, message, tools={}):
        if role.template != "" and role.name not in role.conversation_history:
            role.conversation_history[role.name] = [
                {"role": "system", "content": role.template},
            ]
        messages = role.conversation_history.get(role.name, [])
        if message is not None and message != "":
            messages.append({"role": "user", "content": message, "name": sender})
        print(colored(f"\n---\n// {role.name}\n{json.dumps(messages)}\n---\n", "grey"))

        do_stream = True
        # Start a streaming session
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=role.max_tokens,
            n=1,
            temperature=role.temperature,
            user=f"robits_{role.name}",
            tools=[{k: v for k, v in tool.items() if k != "code"} for tool in tools],
            tool_choice="auto",
            stream=do_stream,
        )
        message = {"role": "assistant", "content": "", "name": role.name}
        if do_stream:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    message["content"] += chunk.choices[0].delta.content
            message["content"] = message["content"].strip()
        else:
            message = response.choices[0].message

        # Remove any additional whitespace and control characters
        if message["content"] != "":
            role.conversation_history[role.name].append(message)

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
                role.conversation_history[role.name].append(tool_message)

            # Get second response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=role.max_tokens,
                n=1,
                temperature=role.temperature,
                user=f"robits_{role.name}",
                stream=True,
            )
            message = {"role": "assistant", "content": "", "name": role.name}
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    message["content"] += chunk.choices[0].delta.content
            message["content"] = message["content"].strip()

        return message["content"]

    @staticmethod
    def interact_cheap(role, sender, message, tools={}):
        return Interact.interact(role, cheap_model, sender, message, tools)

    @staticmethod
    def interact_costly(role, sender, message, tools={}):
        return Interact.interact(role, costly_model, sender, message, tools)

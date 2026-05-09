#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import contextlib
import os
import sys

try:
    from termcolor import colored
except ModuleNotFoundError:  # pragma: no cover - cosmetic fallback
    def colored(text: str, *_args: object, **_kwargs: object) -> str:
        return text

from agents import build_default_agents
from conversation_state import ConversationState
from llm_client import LLMClient
from mcp_service import MCPService
from org_runtime import OrgRuntime
from process_utils import ManagedProcess, maybe_launch_from_env


async def human_input_loop(runtime: OrgRuntime, mcp_service: MCPService, shutdown_event: asyncio.Event) -> None:
    while not shutdown_event.is_set():
        try:
            line = await asyncio.to_thread(input, "CEO> ")
        except EOFError:
            shutdown_event.set()
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"/quit", "/exit"}:
            shutdown_event.set()
            break
        if line.lower() == "/tools":
            tools = mcp_service.get_all_tool_specs()
            if not tools:
                print(colored("No tools registered yet.", "yellow"))
            else:
                print(colored("Registered tools:", "green"))
                for tool in tools:
                    print(colored(f"- {tool['name']}", "green"))
            continue
        if line.lower().startswith("/agenda"):
            parts = line.split()
            if len(parts) == 2:
                agent_name = parts[1]
                agent = runtime.agents.get(agent_name)
                if agent:
                    if not agent.state.agenda:
                        print(colored(f"{agent_name} has no scheduled agenda items.", "yellow"))
                    else:
                        print(colored(f"Agenda for {agent_name}:", "cyan"))
                        for item in agent.state.agenda:
                            print(colored(f"- {item.name}: {item.prompt}", "cyan"))
                else:
                    print(colored(f"Unknown agent '{agent_name}'.", "red"))
            else:
                print(colored("Usage: /agenda <AgentName>", "yellow"))
            continue
        await runtime.emit(
            sender="CEO",
            content=line,
            metadata={"origin": "human"},
        )


async def output_loop(runtime: OrgRuntime, shutdown_event: asyncio.Event) -> None:
    while not shutdown_event.is_set():
        try:
            message = await asyncio.wait_for(runtime.output_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        if message.metadata.get("origin") == "status":
            stage = message.metadata.get("stage")
            agent = message.metadata.get("agent", "Agent")
            if stage == "start":
                symbol = "⏳"
                color = "blue"
            elif stage == "error":
                symbol = "✖"
                color = "red"
            else:
                symbol = "✅"
                color = "green"
            print(colored(f"{agent} {symbol} {message.content}", color))
            continue
        if message.sender == "CEO":
            continue
        color = "cyan"
        if message.sender.startswith("tool/"):
            color = "yellow"
        elif message.sender == "System":
            color = "magenta"
        print(colored(f"{message.sender}: {message.content}", color))


async def _autostop(delay: float, shutdown_event: asyncio.Event) -> None:
    await asyncio.sleep(delay)
    shutdown_event.set()


async def run_simulation() -> None:
    responses_proc: ManagedProcess | None = await maybe_launch_from_env("ROBOTS_RESPONSES")
    conversation = ConversationState()
    llm_client = LLMClient()
    mcp_service = MCPService()
    runtime = OrgRuntime(llm_client, mcp_service, conversation)
    mcp_service.set_role_creator(runtime.create_role)

    for agent in build_default_agents(runtime, llm_client, conversation, mcp_service).values():
        runtime.register_agent(agent)
    runtime.start()

    shutdown_event = asyncio.Event()
    output_task = asyncio.create_task(output_loop(runtime, shutdown_event))
    tasks = [output_task]

    if sys.stdin.isatty():
        tasks.append(asyncio.create_task(human_input_loop(runtime, mcp_service, shutdown_event)))
    else:
        await runtime.emit(
            sender="CEO",
            content="Hello team, I am available asynchronously.",
            metadata={"origin": "script"},
        )

    autostop_seconds = os.getenv("ROBOTS_AUTOSTOP_SECONDS")
    if autostop_seconds:
        try:
            delay = float(autostop_seconds)
        except ValueError:
            delay = 0.0
        if delay > 0:
            tasks.append(
                asyncio.create_task(_autostop(delay, shutdown_event), name="autostop")
            )

    await runtime.emit(
        sender="System",
        content="Office online. Use /tools or /agenda for quick status checks.",
        metadata={"origin": "system"},
    )

    try:
        await shutdown_event.wait()
    finally:
        await runtime.stop()
        if responses_proc:
            await responses_proc.stop()
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def main() -> int:
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        print(colored("\nSimulation terminated by user.", "yellow"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

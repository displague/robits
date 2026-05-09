from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Optional

from conversation_state import ConversationState, Message
from agents import AgentConfig, AgendaItem, BaseAgent


class _ActivityContext:
    def __init__(self, runtime: "OrgRuntime", agent_name: str, description: str) -> None:
        self.runtime = runtime
        self.agent_name = agent_name
        self.description = description

    async def __aenter__(self) -> "_ActivityContext":
        await self.runtime._status(self.agent_name, "start", self.description)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            await self.runtime._status(self.agent_name, "success", self.description)
        else:
            await self.runtime._status(
                self.agent_name,
                "error",
                f"{self.description} failed: {exc}",
            )
        return False


class OrgRuntime:
    def __init__(self, llm_client, mcp_service, conversation: ConversationState) -> None:
        self.llm_client = llm_client
        self.mcp_service = mcp_service
        self.conversation = conversation
        self.agents: Dict[str, BaseAgent] = {}
        self._order: List[str] = []
        self._rr_index = 0
        self.output_queue: asyncio.Queue[Message] = asyncio.Queue()

    def register_agent(self, agent: BaseAgent) -> None:
        if agent.config.name in self.agents:
            raise ValueError(f"Agent {agent.config.name} already registered")
        self.agents[agent.config.name] = agent
        self._order.append(agent.config.name)

    def start(self) -> None:
        for agent in self.agents.values():
            agent.start()

    async def stop(self) -> None:
        for agent in self.agents.values():
            await agent.stop()

    def activity(self, agent_name: str, description: str) -> _ActivityContext:
        return _ActivityContext(self, agent_name, description)

    async def emit(self, sender: str, content: str, metadata: Optional[dict] = None) -> Message:
        message = Message(sender=sender, content=content, metadata=metadata or {})
        message = await self.conversation.append(message)
        await self.output_queue.put(message)
        await self._route(message)
        return message

    async def _route(self, message: Message) -> None:
        recipients = []
        for agent in self.agents.values():
            if agent.matches(message):
                recipients.append(agent)
        if not recipients:
            if message.sender not in self.agents:
                fallback = self._fallback_agent(message.sender)
                if fallback:
                    recipients = [fallback]
        for agent in recipients:
            await agent.enqueue(message)

    def _fallback_agent(self, sender: str) -> Optional[BaseAgent]:
        if not self._order:
            return None
        attempts = 0
        while attempts < len(self._order):
            candidate_name = self._order[self._rr_index % len(self._order)]
            self._rr_index += 1
            attempts += 1
            if candidate_name != sender:
                return self.agents[candidate_name]
        return None

    async def dispatch_tool_call(self, agent_name: str, tool_name: Optional[str], arguments: dict) -> Optional[str]:
        if not tool_name:
            return None
        result = await asyncio.to_thread(self.mcp_service.execute_tool, tool_name, **arguments)
        if isinstance(result, (dict, list)):
            payload = json.dumps(result)
        else:
            payload = str(result)
        await self.emit(
            sender=f"tool/{tool_name}",
            content=payload,
            metadata={"origin": "tool", "requested_by": agent_name, "target": agent_name},
        )
        return payload

    async def register_tool_from_spec(self, agent_name: str, spec: dict) -> bool:
        required = {"name", "description", "parameters", "code"}
        if not required.issubset(spec.keys()):
            return False
        tool_name = spec["name"]
        parameters = spec["parameters"]
        code = spec["code"]
        func_lines = [f"def {tool_name}(**kwargs):"]
        for param in parameters.get("properties", {}):
            func_lines.append(f"    {param} = kwargs.get('{param}')")
        for raw_line in code.split("\n"):
            func_lines.append("    " + raw_line)
        func_body = "\n".join(func_lines)
        await asyncio.to_thread(
            self.mcp_service.register_tool,
            tool_name,
            func_body,
            spec.get("description", ""),
            parameters,
        )
        await self.emit(
            sender="System",
            content=f"Registered new tool '{tool_name}' from {agent_name}.",
            metadata={"origin": "tool-registration", "requested_by": agent_name},
        )
        return True

    async def handle_agent_request(self, requester: str, payload: dict) -> bool:
        name = payload.get("name")
        system_prompt = payload.get("system_prompt") or payload.get("description")
        if not name or not system_prompt:
            return False
        if name in self.agents:
            return False
        keywords = payload.get("keywords") or []
        agenda_items = []
        for item in payload.get("agenda", []):
            if isinstance(item, dict):
                agenda_items.append(
                    AgendaItem(
                        name=str(item.get("name", "Agenda Task")),
                        prompt=str(item.get("prompt", "Review ongoing work.")),
                        interval_seconds=item.get("interval_seconds"),
                        initial_delay_seconds=item.get("initial_delay_seconds"),
                    )
                )
        config = AgentConfig(
            name=name,
            system_prompt=system_prompt,
            keywords=tuple(keywords),
            agenda=tuple(agenda_items),
        )
        new_agent = BaseAgent(config, self, self.llm_client, self.conversation, self.mcp_service)
        self.register_agent(new_agent)
        new_agent.start()
        await self.emit(
            sender="System",
            content=f"HR onboarded new agent {name} at the request of {requester}.",
            metadata={"origin": "hr", "requested_by": requester},
        )
        return True

    def create_role(self, role_name: str, role_description: str) -> str:
        if not role_name or not role_description:
            return "Error: role_name and role_description are required."
        if role_name in self.agents:
            return f"Error: Role '{role_name}' already exists."
        config = AgentConfig(
            name=role_name,
            system_prompt=role_description,
            keywords=(role_name.lower(),),
        )
        new_agent = BaseAgent(config, self, self.llm_client, self.conversation, self.mcp_service)
        self.register_agent(new_agent)
        new_agent.start()
        return f"Created a new role: {role_name}"

    async def log_exception(self, agent_name: str, exc: Exception) -> None:
        await self.emit(
            sender="System",
            content=f"Agent {agent_name} encountered an error: {exc}",
            metadata={"origin": "error", "severity": "error"},
        )

    async def _status(self, agent_name: str, stage: str, description: str) -> None:
        message = Message(
            sender="Status",
            content=description,
            metadata={
                "origin": "status",
                "agent": agent_name,
                "stage": stage,
            },
        )
        await self.output_queue.put(message)

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from conversation_state import ConversationState, Message

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from llm_client import LLMClient
    from mcp_service import MCPService
    from org_runtime import OrgRuntime


@dataclass
class AgendaItem:
    name: str
    prompt: str
    interval_seconds: Optional[int] = None
    initial_delay_seconds: Optional[int] = None
    next_run: Optional[float] = None

    def clone(self) -> "AgendaItem":
        return AgendaItem(
            name=self.name,
            prompt=self.prompt,
            interval_seconds=self.interval_seconds,
            initial_delay_seconds=self.initial_delay_seconds,
            next_run=self.next_run,
        )

    def schedule_next(self, now: float) -> None:
        interval = self.interval_seconds or 0
        delay = self.initial_delay_seconds if self.next_run is None else interval
        if delay is None or delay == 0:
            delay = max(interval, 60)
        self.next_run = now + delay

    def due(self, now: float) -> bool:
        return self.next_run is not None and now >= self.next_run


@dataclass
class AgentState:
    notes: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    agenda: List[AgendaItem] = field(default_factory=list)
    previous_response_id: Optional[str] = None

    def summary(self) -> str:
        sections: List[str] = []
        if self.plan:
            sections.append("Plan: " + "; ".join(self.plan[:6]))
        if self.notes:
            sections.append("Notes: " + "; ".join(self.notes[:6]))
        return " | ".join(sections) if sections else "No persistent notes yet."

    def apply_update(self, update: Dict[str, Any]) -> None:
        if "notes" in update and isinstance(update["notes"], list):
            self.notes = [str(item) for item in update["notes"]][:20]
        if "plan" in update and isinstance(update["plan"], list):
            self.plan = [str(item) for item in update["plan"]][:20]
        if "agenda" in update and isinstance(update["agenda"], list):
            refreshed: List[AgendaItem] = []
            for item in update["agenda"]:
                if not isinstance(item, dict):
                    continue
                refreshed.append(
                    AgendaItem(
                        name=str(item.get("name", "Agenda Item")),
                        prompt=str(item.get("prompt", "Review priorities.")),
                        interval_seconds=item.get("interval_seconds"),
                        initial_delay_seconds=item.get("initial_delay_seconds"),
                    )
                )
            if refreshed:
                self.agenda = refreshed


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    keywords: Sequence[str] = field(default_factory=list)
    model: Optional[str] = None
    temperature: float = 0.7
    max_output_tokens: int = 512
    agenda: Sequence[AgendaItem] = field(default_factory=list)
    abilities: Sequence[str] = field(default_factory=list)
    idle_jitter_seconds: Tuple[int, int] = (15, 45)


class BaseAgent:
    state_update_token = "STATE_UPDATE:"
    agent_spawn_token = "AGENT_REQUEST:"

    def __init__(
        self,
        config: AgentConfig,
        runtime: "OrgRuntime",
        llm_client: "LLMClient",
        conversation: ConversationState,
        mcp_service: "MCPService",
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.llm_client = llm_client
        self.conversation = conversation
        self.mcp_service = mcp_service
        self.state = AgentState(
            agenda=[item.clone() for item in config.agenda]
        )
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._runner: Optional[asyncio.Task[None]] = None
        self._agenda_task: Optional[asyncio.Task[None]] = None
        self._last_idle_ping = time.time()

    def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self._loop(), name=f"agent-{self.config.name}-loop")
        if self.state.agenda and self._agenda_task is None:
            self._agenda_task = asyncio.create_task(self._agenda_loop(), name=f"agent-{self.config.name}-agenda")

    async def stop(self) -> None:
        if self._runner:
            self._runner.cancel()
        if self._agenda_task:
            self._agenda_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            if self._runner:
                await self._runner
        with contextlib.suppress(asyncio.CancelledError):
            if self._agenda_task:
                await self._agenda_task

    def matches(self, message: Message) -> bool:
        if message.sender == self.config.name:
            return False
        targets = message.metadata.get("target") or message.metadata.get("targets")
        if targets:
            if isinstance(targets, str):
                targets = [targets]
            return self.config.name in targets
        if message.metadata.get("origin") == "tool":
            return message.metadata.get("requested_by") == self.config.name
        if message.metadata.get("origin") == "agenda":
            return message.metadata.get("agent") == self.config.name
        content = message.content.lower()
        if any(keyword in content for keyword in self.config.keywords):
            return True
        mention = f"@{self.config.name.lower()}"
        return mention in content

    async def enqueue(self, message: Message) -> None:
        await self._queue.put(message)

    async def _loop(self) -> None:
        while True:
            message = await self._queue.get()
            try:
                await self._handle_message(message)
            except Exception as exc:  # pragma: no cover - defensive logging hook
                await self.runtime.log_exception(self.config.name, exc)

    async def _agenda_loop(self) -> None:
        for item in self.state.agenda:
            item.schedule_next(time.time())
        while True:
            now = time.time()
            for item in self.state.agenda:
                if item.next_run is None:
                    item.schedule_next(now)
                if item.due(now):
                    item.schedule_next(now)
                    synthetic = Message(
                        sender="Scheduler",
                        content=item.prompt,
                        metadata={
                            "origin": "agenda",
                            "agenda": item.name,
                            "agent": self.config.name,
                        },
                    )
                    await self.enqueue(synthetic)
            cooldown = random.randint(*self.config.idle_jitter_seconds)
            await asyncio.sleep(min(cooldown, 60))

    async def _handle_message(self, message: Message) -> None:
        async with self.runtime.activity(
            self.config.name,
            f"contacting model about message from {message.sender}",
        ):
            history = await self.conversation.tail_text()
            instructions = self._compose_instructions()
            response_text, response_id, tool_calls = await self.llm_client.generate(
                agent_name=self.config.name,
                model=self.config.model,
                instructions=instructions,
                user_message=message.content,
                sender=message.sender,
                conversation_snippet=history,
                tools=self.mcp_service.get_all_tool_specs(),
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                previous_response_id=self.state.previous_response_id,
                metadata=message.metadata,
            )
        if response_id:
            self.state.previous_response_id = response_id
        await self._post_process(message, response_text, tool_calls)

    def _compose_instructions(self) -> str:
        sections = [self.config.system_prompt.strip()]
        sections.append(
            "You are taking part in an asynchronous office simulation. Respond concisely and focus on decisions and next steps."
        )
        sections.append(
            "Maintain your personal plan and notes. When you change them, append a line starting with 'STATE_UPDATE:' followed by JSON, e.g. STATE_UPDATE:{\"notes\":[...],\"plan\":[...]}"
        )
        sections.append(f"Current personal context: {self.state.summary()}")
        return "\n\n".join(sections)

    async def _post_process(self, message: Message, response_text: str, tool_calls: List[Dict[str, Any]]) -> None:
        clean_text, state_update = self._extract_state_update(response_text)
        if state_update:
            self.state.apply_update(state_update)
        if clean_text.strip():
            await self.runtime.emit(
                sender=self.config.name,
                content=clean_text.strip(),
                metadata={"responding_to": message.id},
            )
        if tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                await self.runtime.dispatch_tool_call(
                    agent_name=self.config.name,
                    tool_name=call.get("name"),
                    arguments=arguments or {},
                )
        agent_request = self._extract_agent_request(response_text)
        if agent_request:
            await self.runtime.handle_agent_request(self.config.name, agent_request)

    def _extract_state_update(self, response_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        token = self.state_update_token
        if token not in response_text:
            return response_text, None
        idx = response_text.rfind(token)
        prefix = response_text[:idx].rstrip()
        remainder = response_text[idx + len(token):].strip()
        try:
            update = json.loads(remainder)
        except json.JSONDecodeError:
            return response_text, None
        return prefix, update

    def _extract_agent_request(self, response_text: str) -> Optional[Dict[str, Any]]:
        token = self.agent_spawn_token
        if token not in response_text:
            return None
        idx = response_text.rfind(token)
        payload = response_text[idx + len(token):].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None


class OperationsAgent(BaseAgent):
    pass


class SoftwareEngineerAgent(BaseAgent):
    async def _post_process(self, message: Message, response_text: str, tool_calls: List[Dict[str, Any]]) -> None:
        clean_text, state_update = self._extract_state_update(response_text)
        new_tool_registered = False
        if "code" in response_text and "parameters" in response_text:
            spec = self._extract_tool_spec(response_text)
            if spec:
                new_tool_registered = await self.runtime.register_tool_from_spec(self.config.name, spec)
        if state_update:
            self.state.apply_update(state_update)
        if clean_text.strip():
            await self.runtime.emit(
                sender=self.config.name,
                content=clean_text.strip(),
                metadata={"responding_to": message.id, "tool_registered": new_tool_registered},
            )
        if tool_calls:
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                await self.runtime.dispatch_tool_call(
                    agent_name=self.config.name,
                    tool_name=call.get("name"),
                    arguments=arguments or {},
                )

    def _extract_tool_spec(self, text: str) -> Optional[Dict[str, Any]]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start:end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            return None
        required_keys = {"name", "description", "parameters", "code"}
        if not required_keys.issubset(set(payload)):
            return None
        return payload


class HumanResourcesAgent(BaseAgent):
    pass


class GuardianAgent(BaseAgent):
    pass


def build_default_agents(runtime: "OrgRuntime", llm_client: "LLMClient", conversation: ConversationState, mcp_service: "MCPService") -> Dict[str, BaseAgent]:
    agents: Dict[str, BaseAgent] = {}

    ops_config = AgentConfig(
        name="Ops",
        system_prompt="You are Operations. You monitor tool usage and keep the office running smoothly. Decide when tools are needed and execute them.",
        keywords=("tool", "execute", "run", "deploy", "weather"),
        abilities=("tool_runner",),
        agenda=(
            AgendaItem(
                name="MorningSystemsCheck",
                prompt="Review the conversation log and note any pending tool executions or outages.",
                initial_delay_seconds=30,
                interval_seconds=600,
            ),
        ),
    )
    agents["Ops"] = OperationsAgent(ops_config, runtime, llm_client, conversation, mcp_service)

    se_config = AgentConfig(
        name="SE",
        system_prompt=(
            "You are the Software Engineer. You design and deliver new MCP tools on demand. "
            "Return tool definitions as JSON with keys name, description, parameters, and code when creating tools."
        ),
        keywords=("tool", "build", "code", "implement"),
        abilities=("tool_builder",),
        agenda=(
            AgendaItem(
                name="TechBacklog",
                prompt="Review outstanding engineering work and plan next steps.",
                initial_delay_seconds=45,
                interval_seconds=900,
            ),
        ),
    )
    agents["SE"] = SoftwareEngineerAgent(se_config, runtime, llm_client, conversation, mcp_service)

    hr_config = AgentConfig(
        name="HR",
        system_prompt=(
            "You manage organizational health, staffing, and policy. Identify hiring needs and create new roles via AGENT_REQUEST blocks when needed."
        ),
        keywords=("hire", "role", "policy", "culture"),
        agenda=(
            AgendaItem(
                name="PeoplePulse",
                prompt="Check in on team morale and staffing gaps.",
                initial_delay_seconds=60,
                interval_seconds=1200,
            ),
        ),
    )
    agents["HR"] = HumanResourcesAgent(hr_config, runtime, llm_client, conversation, mcp_service)

    guardian_config = AgentConfig(
        name="Guardian",
        system_prompt="You are the office guardian angel. Protect the organization from risky requests while keeping morale high.",
        keywords=("protect", "risk", "security", "angel"),
        agenda=(
            AgendaItem(
                name="WellbeingScan",
                prompt="Scan recent events for risks or wellbeing concerns and respond supportively.",
                initial_delay_seconds=90,
                interval_seconds=1500,
            ),
        ),
    )
    agents["Guardian"] = GuardianAgent(guardian_config, runtime, llm_client, conversation, mcp_service)

    return agents

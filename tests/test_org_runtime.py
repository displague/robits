import asyncio

from agents import AgentConfig, BaseAgent
from conversation_state import ConversationState
from mcp_service import MCPService
from org_runtime import OrgRuntime


class DummyLLMClient:
    pass


def test_create_role_registers_and_starts_agent():
    asyncio.run(_run_create_role_registers_and_starts_agent())


async def _run_create_role_registers_and_starts_agent():
    conversation = ConversationState()
    service = MCPService()
    runtime = OrgRuntime(DummyLLMClient(), service, conversation)
    service.set_role_creator(runtime.create_role)

    result = service.execute_tool(
        "create_role",
        role_name="Analyst",
        role_description="Analyze project risks.",
    )

    try:
        assert result == "Created a new role: Analyst"
        assert "Analyst" in runtime.agents
        assert isinstance(runtime.agents["Analyst"], BaseAgent)
        assert runtime.agents["Analyst"].config == AgentConfig(
            name="Analyst",
            system_prompt="Analyze project risks.",
            keywords=("analyst",),
        )
    finally:
        await runtime.stop()

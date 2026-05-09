from mcp_service import MCPService


def test_get_current_weather_tool_spec_and_execution():
    service = MCPService()

    spec = service.get_tool_spec("get_current_weather")

    assert spec["type"] == "function"
    assert spec["name"] == "get_current_weather"
    assert spec["parameters"]["required"] == ["location"]
    assert "San Francisco" in service.execute_tool("get_current_weather", location="San Francisco, CA")


def test_create_role_requires_runtime_creator():
    service = MCPService()

    result = service.execute_tool(
        "create_role",
        role_name="Analyst",
        role_description="Analyze project risks.",
    )

    assert "not available" in result


def test_create_role_delegates_to_runtime_creator():
    service = MCPService()
    calls = []
    service.set_role_creator(lambda name, description: calls.append((name, description)) or f"Created {name}")

    result = service.execute_tool(
        "create_role",
        role_name="Analyst",
        role_description="Analyze project risks.",
    )

    assert result == "Created Analyst"
    assert calls == [("Analyst", "Analyze project risks.")]

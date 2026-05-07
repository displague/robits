from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class SandboxMetadata:
    agent_name: str
    enabled: bool = False
    backend: str = "none"
    private_workspace: str | None = None
    shared_organization_workspace: str | None = None
    policy: str = "disabled"

    @classmethod
    def disabled(cls, agent_name):
        return cls(agent_name=agent_name)

    @classmethod
    def local_process(
        cls,
        agent_name,
        private_workspace,
        shared_organization_workspace,
        policy="agent",
    ):
        return cls(
            agent_name=agent_name,
            enabled=True,
            backend="local_process",
            private_workspace=str(PurePosixPath(private_workspace)),
            shared_organization_workspace=str(PurePosixPath(shared_organization_workspace)),
            policy=policy,
        )


@dataclass(frozen=True)
class SandboxExecutionRequest:
    agent_name: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    private_workspace: str | None = None
    shared_organization_workspace: str | None = None
    policy: str = "agent"


@dataclass(frozen=True)
class SandboxExecutionResult:
    status: str
    output: str
    backend: str
    agent_name: str
    tool_name: str


class FakeSandboxBackend:
    """Deterministic backend for tests and future container adapters."""

    def __init__(self, output="ok", name="fake"):
        self.output = output
        self.name = name
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return SandboxExecutionResult(
            status="completed",
            output=self.output,
            backend=self.name,
            agent_name=request.agent_name,
            tool_name=request.tool_name,
        )


class SandboxRuntime:
    def __init__(self, backend=None):
        self.backend = backend

    def execute_tool(self, sandbox_metadata, tool_name, arguments=None):
        if sandbox_metadata is None or not sandbox_metadata.enabled:
            return SandboxExecutionResult(
                status="skipped",
                output="Sandbox disabled; execute in current runtime.",
                backend="none",
                agent_name=getattr(sandbox_metadata, "agent_name", ""),
                tool_name=tool_name,
            )
        if self.backend is None:
            return SandboxExecutionResult(
                status="error",
                output="Sandbox enabled but no backend is configured.",
                backend="none",
                agent_name=sandbox_metadata.agent_name,
                tool_name=tool_name,
            )
        if sandbox_metadata.backend != self.backend.name:
            return SandboxExecutionResult(
                status="error",
                output=(
                    f"Sandbox backend mismatch: requested '{sandbox_metadata.backend}' "
                    f"but configured '{self.backend.name}'."
                ),
                backend=self.backend.name,
                agent_name=sandbox_metadata.agent_name,
                tool_name=tool_name,
            )
        request = SandboxExecutionRequest(
            agent_name=sandbox_metadata.agent_name,
            tool_name=tool_name,
            arguments=arguments or {},
            private_workspace=sandbox_metadata.private_workspace,
            shared_organization_workspace=sandbox_metadata.shared_organization_workspace,
            policy=sandbox_metadata.policy,
        )
        return self.backend.execute(request)

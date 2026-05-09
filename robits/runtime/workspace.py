from __future__ import annotations

import os
from pathlib import Path, PurePosixPath


class WorkspacePathError(ValueError):
    pass


def safe_agent_workspace_name(agent_name):
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise WorkspacePathError("Agent name must be a non-empty string.")
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in agent_name.strip())


class AgentWorkspaceStore:
    """Private persistent filesystem roots for agents."""

    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("ROBITS_AGENT_WORKSPACE_ROOT", "var/agents"))

    def workspace_root(self, agent_name):
        return self.root / safe_agent_workspace_name(agent_name)

    def _resolve(self, agent_name, relative_path=""):
        workspace = self.workspace_root(agent_name).resolve()
        raw_path = "" if relative_path is None else str(relative_path).strip()
        posix_path = PurePosixPath(raw_path)
        if posix_path.is_absolute() or any(part == ".." for part in posix_path.parts):
            raise WorkspacePathError("Path must be relative and stay inside the agent workspace.")
        target = (workspace / Path(*posix_path.parts)).resolve()
        if target != workspace and workspace not in target.parents:
            raise WorkspacePathError("Path escapes the agent workspace.")
        return workspace, target

    def list(self, agent_name, relative_path=""):
        workspace, target = self._resolve(agent_name, relative_path)
        target.mkdir(parents=True, exist_ok=True)
        entries = []
        for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
            relative = child.relative_to(workspace).as_posix()
            entries.append(
                {
                    "path": relative,
                    "kind": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return entries

    def read(self, agent_name, relative_path, max_bytes=65536):
        workspace, target = self._resolve(agent_name, relative_path)
        if not target.exists() or not target.is_file():
            raise WorkspacePathError("File not found.")
        data = target.read_bytes()
        truncated = len(data) > max_bytes
        content = data[:max_bytes].decode("utf-8", errors="replace")
        return {
            "path": target.relative_to(workspace).as_posix(),
            "content": content,
            "truncated": truncated,
            "size": len(data),
        }

    def write(self, agent_name, relative_path, content, append=False):
        workspace, target = self._resolve(agent_name, relative_path)
        if target == workspace:
            raise WorkspacePathError("Path must name a file.")
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as handle:
            handle.write("" if content is None else str(content))
        return {
            "path": target.relative_to(workspace).as_posix(),
            "size": target.stat().st_size,
            "append": bool(append),
        }

    def delete(self, agent_name, relative_path):
        workspace, target = self._resolve(agent_name, relative_path)
        if target == workspace:
            raise WorkspacePathError("Cannot delete the workspace root.")
        if not target.exists():
            raise WorkspacePathError("Path not found.")
        if target.is_dir():
            target.rmdir()
            kind = "directory"
        else:
            target.unlink()
            kind = "file"
        return {"path": target.relative_to(workspace).as_posix(), "kind": kind}

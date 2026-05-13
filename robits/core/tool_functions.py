"""Tool functions exposed to agents."""
import ast
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from robits.core.config import _config as _m
from robits.core.lifecycle import (
    validate_role_name,
    _caller_can_act_for_agent,
    _caller_name_for_error,
    _workspace_agent_name,
)
from robits.core.tools import (
    _normalize_tool_grant,
    _coerce_json_object,
    _normalize_tool_parameters,
    _coerce_string_list,
    _validate_tool_code_ast,
)
from robits.runtime.tool_proposals import ToolProposalStore


def workspace_list(employee_dict, agent_name, path=""):
    """List files in an agent's workspace directory at path."""
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        return json.dumps(_m.agent_workspace_store.list(normalized_name, path), sort_keys=True)
    except Exception as e:
        return f"Error: {e}"


def workspace_read(employee_dict, agent_name, path, max_bytes=65536):
    """Read a file from an agent's workspace, returning content up to max_bytes."""
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        read_limit = 65536 if max_bytes is None else int(max_bytes)
        return json.dumps(
            _m.agent_workspace_store.read(normalized_name, path, max_bytes=read_limit),
            sort_keys=True,
        )
    except Exception as e:
        return f"Error: {e}"


def workspace_write(employee_dict, agent_name, path, content, append=False):
    """Write (or append) content to a file in an agent's workspace."""
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        return json.dumps(_m.agent_workspace_store.write(normalized_name, path, content, append=append), sort_keys=True)
    except Exception as e:
        return f"Error: {e}"


def workspace_delete(employee_dict, agent_name, path):
    """Delete a file from an agent's workspace."""
    del employee_dict
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    try:
        return json.dumps(_m.agent_workspace_store.delete(normalized_name, path), sort_keys=True)
    except Exception as e:
        return f"Error: {e}"


def org_chat_read(employee_dict, limit=20):
    """Return the most recent `limit` org chat log entries as a JSON object."""
    del employee_dict
    if _m._org_workspace is None:
        return json.dumps({"lines": [], "note": "org chat not available"})
    try:
        result = _m._org_workspace.read("org", "org_chat.jsonl")
    except Exception:
        return json.dumps({"lines": [], "note": "no org chat history yet"})
    effective_limit = max(0, min(int(limit or 20), 100))
    if effective_limit == 0:
        return json.dumps({"lines": [], "total": 0})
    all_lines = [ln for ln in result["content"].splitlines() if ln.strip()]
    tail = all_lines[-effective_limit:]
    parsed = []
    for ln in tail:
        try:
            parsed.append(json.loads(ln))
        except Exception:
            pass
    return json.dumps({"lines": parsed, "total": len(all_lines)})


def work_todo_add(employee_dict, title, content=None):
    """Append a work-todo memory entry for the active agent."""
    del employee_dict
    if _m.memory_store is None:
        return json.dumps({"error": "memory store not available"})
    agent_id = _m.active_tool_caller_name or getattr(_m.active_tool_caller, "name", None)
    if not agent_id:
        return json.dumps({"error": "could not determine caller identity"})
    try:
        todo_id = _m.memory_store.append_todo(
            agent_id=agent_id,
            title=title,
            content=content,
            status="open",
        )
        return json.dumps({"todo_id": todo_id, "title": title, "status": "open"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def grant_tool_access(employee_dict, role_name, tool_name, granted_by=None):
    """Add tool_name to a role's allowed_tools set; return a status string or error."""
    try:
        normalized_name = validate_role_name(role_name)
        normalized_tool_name = _normalize_tool_grant(tool_name)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    if normalized_tool_name != "*" and not normalized_tool_name.endswith(".*") and normalized_tool_name not in _m.tool_registry:
        return f"Error: Tool '{tool_name}' not found."
    if normalized_tool_name.endswith(".*"):
        namespace = normalized_tool_name[:-2]
        if not any(tool.name.startswith(f"{namespace}.") for tool in _m.tool_registry._tools.values()):
            return f"Error: Tool namespace '{namespace}' not found."
    role = employee_dict[normalized_name]
    role.allowed_tools.add(normalized_tool_name)
    actor_text = f" by {granted_by}" if granted_by else ""
    return f"Granted tool access '{normalized_tool_name}' to role '{normalized_name}'{actor_text}."


def revoke_tool_access(employee_dict, role_name, tool_name, revoked_by=None):
    """Remove tool_name from a role's allowed_tools set; return a status string or error."""
    try:
        normalized_name = validate_role_name(role_name)
        normalized_tool_name = _normalize_tool_grant(tool_name)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    role = employee_dict[normalized_name]
    if normalized_tool_name not in role.allowed_tools:
        return f"Error: Role '{normalized_name}' does not have grant '{normalized_tool_name}'."
    role.allowed_tools.remove(normalized_tool_name)
    actor_text = f" by {revoked_by}" if revoked_by else ""
    return f"Revoked tool access '{normalized_tool_name}' from role '{normalized_name}'{actor_text}."


def list_registered_tools(employee_dict, role_name=None, include_system=True, only_allowed=False):
    """Return a JSON array of registered tools, optionally filtered by role access."""
    role = None
    if only_allowed and not role_name:
        return "Error: role_name is required when only_allowed is true."
    if role_name:
        try:
            normalized_name = validate_role_name(role_name)
        except ValueError as e:
            return f"Error: {e}"
        if normalized_name not in employee_dict:
            return f"Error: Role '{normalized_name}' not found."
        role = employee_dict[normalized_name]
    return json.dumps(
        _m.tool_registry.list_tools(
            include_system=include_system,
            role=role,
            include_denied=not only_allowed,
        ),
        sort_keys=True,
    )


def get_tool_proposal_store():
    """Return the global ToolProposalStore, initialising it on first access."""
    if _m.tool_proposal_store is None:
        _m.tool_proposal_store = ToolProposalStore(
            os.environ.get("ROBITS_TOOL_PROPOSALS_FILE", "var/tool_proposals.json")
        )
    return _m.tool_proposal_store


def propose_tool_change(
    employee_dict,
    requested_by,
    tool_name,
    description,
    action="create",
    parameters=None,
    required_capabilities=None,
    owner_capability=None,
    safety_notes="",
    implementation_notes="",
    code="",
):
    """Submit a create or update proposal for a tool; return a status string or error."""
    del employee_dict
    try:
        _m.tool_registry.validate_tool_name(tool_name)
        normalized_parameters = _normalize_tool_parameters(_coerce_json_object(
            parameters,
            {"type": "object", "properties": {}, "required": []},
        ))
        normalized_capabilities = _coerce_string_list(required_capabilities)
    except ValueError as e:
        return f"Error: {e}"
    if not isinstance(description, str) or not description.strip():
        return "Error: Tool proposal description must be a non-empty string."
    if action not in {"create", "update"}:
        return f"Error: Tool proposal action must be create or update."
    if tool_name in _m.tool_registry:
        tool = _m.tool_registry.get(tool_name)
        if tool.system_tool:
            return f"Error: System tool '{tool_name}' cannot be changed by SE proposal."
    normalized_code = code.strip() if isinstance(code, str) else ""
    if normalized_code:
        try:
            tree = ast.parse(normalized_code)
        except SyntaxError as e:
            return f"Error: Tool code has a syntax error: {e}"
        validation_errors = _validate_tool_code_ast(tree)
        if validation_errors:
            return f"Error: {validation_errors[0]}"
        arg_names = list(normalized_parameters.get("properties", {}).keys())
        required_arg_names = normalized_parameters.get("required", [])
        if not isinstance(required_arg_names, list):
            required_arg_names = []
        try:
            _m.tool_registry.compile_tool(tool_name, arg_names, required_arg_names, normalized_code)
        except (SyntaxError, ValueError) as e:
            return f"Error: Tool code failed to compile: {e}"
    proposal = get_tool_proposal_store().create(
        requested_by=requested_by,
        tool_name=tool_name,
        action=action,
        description=description.strip(),
        parameters=normalized_parameters,
        required_capabilities=normalized_capabilities,
        owner_capability=owner_capability,
        safety_notes=safety_notes,
        implementation_notes=implementation_notes,
        code=normalized_code,
    )
    return json.dumps(proposal, sort_keys=True)


def list_tool_proposals(employee_dict, status=None):
    """Return a JSON array of tool proposals, optionally filtered by status."""
    del employee_dict
    return json.dumps(get_tool_proposal_store().list(status=status), sort_keys=True)


def approve_tool_proposal(employee_dict, proposal_id, approved_by, implementation_notes=""):
    """Mark a proposal as approved; return the updated proposal JSON or an error."""
    del employee_dict
    store = get_tool_proposal_store()
    proposal = store.get(proposal_id)
    if proposal is None:
        return f"Error: Tool proposal '{proposal_id}' not found."
    if proposal["status"] not in {"proposed", "approved"}:
        return f"Error: Tool proposal '{proposal_id}' cannot be approved from status '{proposal['status']}'."
    updated = store.update(
        proposal_id,
        status="approved",
        approver=approved_by,
        implementation_notes=implementation_notes or proposal.get("implementation_notes", ""),
    )
    return json.dumps(updated, sort_keys=True)


def reject_tool_proposal(employee_dict, proposal_id, rejected_by, reason):
    """Mark a proposal as rejected with a mandatory reason string."""
    del employee_dict
    if not isinstance(reason, str) or not reason.strip():
        return "Error: Rejection reason must be a non-empty string."
    store = get_tool_proposal_store()
    proposal = store.get(proposal_id)
    if proposal is None:
        return f"Error: Tool proposal '{proposal_id}' not found."
    if proposal["status"] in {"operationalized", "rejected"}:
        return f"Error: Tool proposal '{proposal_id}' cannot be rejected from status '{proposal['status']}'."
    updated = store.update(
        proposal_id,
        status="rejected",
        approver=rejected_by,
        rejection_reason=reason.strip(),
    )
    return json.dumps(updated, sort_keys=True)


def rollout_tool_proposal(employee_dict, proposal_id, role_name, granted_by=None, rollout_notes=""):
    """Register or replace the tool from an approved proposal and grant it to role_name."""
    store = get_tool_proposal_store()
    proposal = store.get(proposal_id)
    if proposal is None:
        return f"Error: Tool proposal '{proposal_id}' not found."
    if proposal["status"] != "approved":
        return f"Error: Tool proposal '{proposal_id}' must be approved before rollout."
    tool_name = proposal["tool_name"]
    code = proposal.get("code", "").strip()
    action = proposal.get("action", "create")
    if tool_name not in _m.tool_registry:
        if not code:
            return f"Error: Tool '{tool_name}' is not registered and proposal has no code to register."
        try:
            _m.tool_registry.register_definition({
                "name": tool_name,
                "description": proposal["description"],
                "parameters": proposal.get("parameters", {}),
                "code": code,
                "grantable": True,
                "system_tool": False,
                "required_capabilities": proposal.get("required_capabilities", []),
                "owner_capability": proposal.get("owner_capability"),
            })
        except (SyntaxError, ValueError) as e:
            return f"Error: Failed to register tool '{tool_name}': {e}"
    elif code and action == "update":
        existing = _m.tool_registry.get(tool_name)
        if existing and existing.system_tool:
            return f"Error: Cannot replace system tool '{tool_name}' via proposal."
        try:
            _m.tool_registry.replace_definition({
                "name": tool_name,
                "description": proposal["description"],
                "parameters": proposal.get("parameters", {}),
                "code": code,
                "grantable": existing.grantable if existing else True,
                "system_tool": False,
                "required_capabilities": proposal.get("required_capabilities", []),
                "owner_capability": proposal.get("owner_capability"),
            })
        except (SyntaxError, ValueError) as e:
            return f"Error: Failed to replace tool '{tool_name}': {e}"
    grant_result = grant_tool_access(
        employee_dict,
        role_name,
        tool_name,
        granted_by=granted_by,
    )
    if grant_result.startswith("Error:"):
        return grant_result
    granted_roles = sorted(set(proposal.get("granted_roles", [])) | {validate_role_name(role_name)})
    updated = store.update(
        proposal_id,
        status="operationalized",
        rollout_notes=rollout_notes or proposal.get("rollout_notes", ""),
        granted_roles=granted_roles,
    )
    return json.dumps({"proposal": updated, "grant_result": grant_result}, sort_keys=True)


def approve_and_rollout_proposal(employee_dict, role_name, tool_name=None, proposal_id=None, approved_by=None):
    """Approve and roll out a proposal in one step, by tool_name or proposal_id."""
    store = get_tool_proposal_store()
    if proposal_id:
        proposal = store.get(proposal_id)
        if proposal is None:
            return f"Error: Proposal '{proposal_id}' not found."
    elif tool_name:
        candidates = [p for p in store.list(status="proposed") if p["tool_name"] == tool_name]
        if not candidates:
            candidates = [p for p in store.list(status="approved") if p["tool_name"] == tool_name]
        if not candidates:
            return f"Error: No pending proposal found for tool '{tool_name}'."
        proposal = candidates[-1]
        proposal_id = proposal["proposal_id"]
    else:
        return "Error: Provide tool_name or proposal_id."
    approver = approved_by or (_m.active_tool_caller_name or getattr(_m.active_tool_caller, "name", None) or "Ops")
    if proposal["status"] == "proposed":
        approve_result = approve_tool_proposal(employee_dict, proposal_id, approved_by=approver)
        if approve_result.startswith("Error:"):
            return approve_result
    return rollout_tool_proposal(employee_dict, proposal_id, role_name, granted_by=approver)


def _require_memory_store():
    """Return (memory_store, None) or (None, error_string) if no store is configured."""
    if _m.memory_store is None:
        return None, "Error: No SQLite memory store is configured."
    return _m.memory_store, None


def _write_memory_cache(agent_name, filename, content):
    """Write content to the agent's .memory-cache/ workspace directory."""
    cache_path = f".memory-cache/{filename}"
    try:
        _m.agent_workspace_store.write(agent_name, cache_path, content)
        return cache_path
    except Exception:
        return None


def _condense_if_large(agent_name, tool_label, result_json):
    """Return result_json directly, or cache it and return a snippet if too large."""
    if len(result_json) <= _m.memory_cache_threshold:
        return result_json
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{tool_label.replace('.', '_')}_{uuid4().hex[:8]}.json"
    cache_path = _write_memory_cache(agent_name, filename, result_json)
    snippet = result_json[:2048]
    condensed = {
        "truncated": True,
        "total_chars": len(result_json),
        "snippet": snippet,
    }
    if cache_path:
        condensed["cache_path"] = cache_path
        condensed["note"] = (
            f"Full result ({len(result_json)} chars) cached in agent workspace at "
            f"'{cache_path}'. Use agent.files_read to retrieve it."
        )
    else:
        condensed["note"] = "Result truncated; workspace cache unavailable."
    return json.dumps(condensed, sort_keys=True)


_WORK_CHANNEL_TYPES = {"org_chat", "work_peer"}


def memory_search(employee_dict, agent_name, query, limit=10, cascade=True, channel_type=None):
    """Search an agent's memory store and return matching results, condensing large payloads."""
    del employee_dict
    store, error = _require_memory_store()
    if error:
        return error
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect memory for '{normalized_name}'."
    if not isinstance(query, str) or not query.strip():
        return "Error: Memory query must be a non-empty string."
    effective_limit = max(1, min(int(limit or 10), _m.memory_max_rows))
    search_kwargs = dict(agent_id=normalized_name, limit=effective_limit)
    if channel_type is not None:
        search_kwargs["conversation_type"] = channel_type
    from robits.core.embeddings import embedding_model_name
    _embed_model = embedding_model_name()
    if cascade and _embed_model:
        results = store.search_hybrid(query, _embed_model, **search_kwargs)
    elif cascade:
        results = store.search_cascade(query, **search_kwargs)
    else:
        results = store.search(query, **search_kwargs)
    result_json = json.dumps([result.__dict__ for result in results], sort_keys=True)
    _clock = _m.clock_state
    if _clock in {"off", "break"} and any(
        getattr(r, "conversation_type", None) in _WORK_CHANNEL_TYPES for r in results
    ):
        if _clock == "off":
            note = (
                "[System note: work-related content surfaced in these results. "
                "Consider parking any useful insights via the work.todo tool "
                "and return focus to the current personal context.]\n"
            )
        else:
            note = (
                "[System note: work-related content surfaced during a break. "
                "You may briefly note any insights via work.todo, but keep your break focus.]\n"
            )
        result_json = note + result_json
    return _condense_if_large(normalized_name, "memory_search", result_json)


def memory_list_digests(employee_dict, agent_name, digest_type=None, limit=10):
    """List current memory digests for an agent, optionally filtered by digest_type."""
    del employee_dict
    store, error = _require_memory_store()
    if error:
        return error
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect memory for '{normalized_name}'."
    effective_limit = max(1, min(int(limit or 10), _m.memory_max_rows))
    digests = store.list_memory_digests(
        agent_id=normalized_name,
        digest_type=digest_type,
        current_only=True,
        accessible_only=True,
        limit=effective_limit,
    )
    result_json = json.dumps(digests, sort_keys=True)
    return _condense_if_large(normalized_name, "memory_list_digests", result_json)


def memory_expand_digest(employee_dict, agent_name, digest_id, recursive=True, max_depth=None, max_chars=None):
    """Expand a memory digest to its source messages, recursively if requested."""
    del employee_dict
    store, error = _require_memory_store()
    if error:
        return error
    try:
        normalized_name = validate_role_name(agent_name)
        digest_id = int(digest_id)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect memory for '{normalized_name}'."
    digest = store.get_memory_digest(digest_id)
    if digest is None:
        return f"Error: Memory digest '{digest_id}' not found."
    if digest.get("agent_id") not in {None, normalized_name}:
        return f"Error: Memory digest '{digest_id}' is not accessible to {normalized_name}."
    if digest.get("system_only") or digest.get("accessibility") != "agent":
        return f"Error: Memory digest '{digest_id}' is system-only."
    effective_depth = max(
        0,
        min(
            int(max_depth) if max_depth is not None else _m.memory_max_depth,
            _m.memory_max_depth,
        ),
    )
    rows = store.expand_memory_digest_sources(
        digest_id,
        recursive=recursive,
        max_depth=effective_depth if recursive else None,
    )
    rows = [
        {
            **row.__dict__,
            "source_path": list(row.source_path),
        }
        if hasattr(row, "__dict__")
        else row
        for row in rows
    ]
    rows = rows[:_m.memory_max_rows]
    result_json = json.dumps(rows, sort_keys=True)
    effective_max_chars = max(1, int(max_chars)) if max_chars is not None else None
    if effective_max_chars is not None and len(result_json) > effective_max_chars:
        condensed = {
            "truncated": True,
            "total_chars": len(result_json),
            "note": f"Result exceeded max_chars={effective_max_chars}. Use a higher max_chars or memory.search to narrow results.",
        }
        return json.dumps(condensed, sort_keys=True)
    return _condense_if_large(normalized_name, "memory_expand_digest", result_json)


def builtin_web_search(employee_dict, query, num_results=5):
    """Perform a web search via the configured search URL or DuckDuckGo as a fallback."""
    del employee_dict
    if not isinstance(query, str) or not query.strip():
        return "Error: Search query must be a non-empty string."
    n = max(1, min(20, int(num_results or 5)))
    if _m.builtin_search_url:
        import urllib.request
        import urllib.parse
        url = f"{_m.builtin_search_url}?q={urllib.parse.quote(query.strip())}&num={n}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "robits/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode()
        except Exception as exc:
            return f"Error: Search request failed: {exc}"
    import urllib.request
    import urllib.parse
    qs = urllib.parse.urlencode({"q": query.strip(), "format": "json", "no_html": "1", "skip_disambig": "1"})
    try:
        req = urllib.request.Request(
            f"https://api.duckduckgo.com/?{qs}",
            headers={"User-Agent": "robits/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return f"Error: Web search failed: {exc}"
    results = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", query.strip()),
            "url": data.get("AbstractURL", ""),
            "snippet": data["AbstractText"],
        })

    def _extract_topics(topics):
        for topic in topics:
            if len(results) >= n:
                break
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })
            elif isinstance(topic, dict) and "Topics" in topic:
                _extract_topics(topic["Topics"])

    _extract_topics(data.get("RelatedTopics", []))
    return json.dumps(results[:n], sort_keys=True)


def builtin_file_search(employee_dict, agent_name, query, path="", max_results=10):
    """Search an agent's workspace files for lines matching query."""
    del employee_dict
    if not isinstance(query, str) or not query.strip():
        return "Error: Search query must be a non-empty string."
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    workspace = _m.agent_workspace_store.workspace_root(normalized_name)
    search_root = workspace / path if path else workspace
    search_root = search_root.resolve()
    if workspace not in search_root.parents and search_root != workspace:
        return "Error: Path escapes the agent workspace."
    query_lower = query.strip().lower()
    limit = max(1, min(100, int(max_results or 10)))
    results = []
    try:
        candidates = search_root.rglob("*")
    except Exception:
        candidates = iter([])
    for file_path in candidates:
        if len(results) >= limit:
            break
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        idx = content.lower().find(query_lower)
        if idx == -1:
            continue
        snippet = content[max(0, idx - 80) : idx + 160].strip()
        results.append({
            "path": file_path.relative_to(workspace).as_posix(),
            "snippet": snippet,
            "size": file_path.stat().st_size,
        })
    return json.dumps(results, sort_keys=True)


def builtin_shell_run(employee_dict, agent_name, command, timeout=30):
    """Run a shell command in the agent's workspace; requires the 'shell' capability."""
    del employee_dict
    import subprocess as _subprocess
    caller_caps = getattr(_m.active_tool_caller, "capabilities", set())
    if "shell" not in caller_caps:
        caller_label = getattr(_m.active_tool_caller, "name", "unknown")
        return f"Error: Role '{caller_label}' does not have the 'shell' capability."
    if not isinstance(command, str) or not command.strip():
        return "Error: Command must be a non-empty string."
    normalized_name, error = _workspace_agent_name(agent_name)
    if error:
        return error
    workspace = _m.agent_workspace_store.workspace_root(normalized_name)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        proc = _subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, min(300, int(timeout or 30))),
            cwd=str(workspace),
        )
        return json.dumps({
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }, sort_keys=True)
    except _subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as exc:
        return f"Error: Shell execution failed: {exc}"


def builtin_tool_search(employee_dict, query, role_name=None):
    """Search registered tool names and descriptions by query; return up to 20 matches."""
    if not isinstance(query, str) or not query.strip():
        return "Error: Tool search query must be a non-empty string."
    query_lower = query.strip().lower()
    if role_name:
        try:
            from robits.core.lifecycle import validate_role_name as _vrn
            normalized = _vrn(role_name)
        except ValueError as exc:
            return f"Error: {exc}"
        role = employee_dict.get(normalized)
        if role is None:
            return f"Error: Role '{normalized}' not found."
    else:
        role = _m.active_tool_caller
    rows = _m.tool_registry.list_tools(include_system=False, role=role, include_denied=False)
    matches = [
        row for row in rows
        if query_lower in row["name"].lower() or query_lower in row["description"].lower()
    ]
    return json.dumps(matches[:20], sort_keys=True)


def builtin_mcp_call(employee_dict, server_url, tool_name, arguments=None):
    """Stub: MCP server tool calls are not implemented in this runtime."""
    del employee_dict
    return "Error: builtin.mcp_call is not implemented in this runtime. Configure an MCP server and use a supported MCP connector."


def builtin_computer_use(employee_dict, action, coordinate=None):
    """Stub: computer-use actions are not implemented in this runtime."""
    del employee_dict
    return "Error: builtin.computer_use is not implemented in this runtime."


def builtin_image_generation(employee_dict, prompt, size=None, quality=None):
    """Stub: image generation is not implemented in this runtime."""
    del employee_dict
    return "Error: builtin.image_generation is not implemented in this runtime."


def agent_think(employee_dict, content):
    """Private reasoning step — silent return. Thought is preserved in the tool-call history."""
    del employee_dict, content
    return ""


_WAIT_STATE_FILE = ".wait_state.json"


def agent_wait(employee_dict, minutes):
    """Suspend the calling agent for up to N minutes.

    The wait state is persisted to the agent's workspace so it survives session
    boundaries.  On next session start, Session.__init__ restores it.
    """
    del employee_dict
    caller = _m.active_tool_caller
    if caller is None:
        return "Error: agent.wait called outside of agent context."
    try:
        duration = float(minutes)
    except (TypeError, ValueError):
        return "Error: minutes must be a number."
    if duration <= 0:
        return "Error: minutes must be positive."
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    caller.waiting_until = now + timedelta(minutes=duration)
    caller.wait_started_turn = _m.active_session_transcript_length
    caller.wait_clock_state = getattr(caller, "runtime_clock_state", _m.clock_state)
    agent_name = _m.active_tool_caller_name or getattr(caller, "name", None)
    if agent_name and _m.agent_workspace_store is not None:
        try:
            _m.agent_workspace_store.write(
                agent_name,
                _WAIT_STATE_FILE,
                json.dumps({
                    "waiting_until": caller.waiting_until.isoformat(),
                    "wait_clock_state": caller.wait_clock_state,
                }),
            )
        except Exception:
            pass
    return ""


def _restore_wait_state(agent_name, role):
    """Check workspace for a persisted wait state and restore it on role if still active."""
    if _m.agent_workspace_store is None:
        return
    try:
        result = _m.agent_workspace_store.read(agent_name, _WAIT_STATE_FILE)
        raw = result.get("content", "") if isinstance(result, dict) else result
    except Exception:
        return
    if not raw:
        return
    try:
        state = json.loads(raw)
        waiting_until = datetime.fromisoformat(state["waiting_until"])
    except Exception:
        try:
            _m.agent_workspace_store.delete(agent_name, _WAIT_STATE_FILE)
        except Exception:
            pass
        return
    now = datetime.now(timezone.utc)
    if waiting_until <= now:
        try:
            _m.agent_workspace_store.delete(agent_name, _WAIT_STATE_FILE)
        except Exception:
            pass
        return
    role.waiting_until = waiting_until
    role.wait_clock_state = state.get("wait_clock_state")

# Functions available in the tool sandbox (exec globals for compile_tool).
# Add new sandbox-accessible functions here; compile_tool picks them up automatically.
SANDBOX_GLOBALS = {
    "workspace_list": workspace_list,
    "workspace_read": workspace_read,
    "workspace_write": workspace_write,
    "workspace_delete": workspace_delete,
    "org_chat_read": org_chat_read,
    "work_todo_add": work_todo_add,
    "memory_search": memory_search,
    "memory_list_digests": memory_list_digests,
    "memory_expand_digest": memory_expand_digest,
    "grant_tool_access": grant_tool_access,
    "revoke_tool_access": revoke_tool_access,
    "list_registered_tools": list_registered_tools,
    "propose_tool_change": propose_tool_change,
    "list_tool_proposals": list_tool_proposals,
    "approve_tool_proposal": approve_tool_proposal,
    "reject_tool_proposal": reject_tool_proposal,
    "rollout_tool_proposal": rollout_tool_proposal,
    "approve_and_rollout_proposal": approve_and_rollout_proposal,
    "builtin_web_search": builtin_web_search,
    "builtin_file_search": builtin_file_search,
    "builtin_shell_run": builtin_shell_run,
    "builtin_tool_search": builtin_tool_search,
    "builtin_mcp_call": builtin_mcp_call,
    "builtin_computer_use": builtin_computer_use,
    "builtin_image_generation": builtin_image_generation,
    "agent_think": agent_think,
    "agent_wait": agent_wait,
}

"""Runtime context and formatting utilities."""
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import main as _m


def _runtime_timezone():
    """Return the configured runtime timezone, falling back to local or UTC."""
    if _m.default_timezone:
        try:
            return ZoneInfo(_m.default_timezone)
        except ZoneInfoNotFoundError:
            pass
    return datetime.now().astimezone().tzinfo or timezone.utc


def _runtime_timezone_name(tzinfo):
    """Return a human-readable name for the given tzinfo object."""
    if _m.default_timezone:
        return _m.default_timezone
    if tzinfo is None:
        return "UTC"
    return getattr(tzinfo, "key", None) or str(tzinfo)


def _get_identity_digests(agent_id):
    """Return (primary_content, secondary_content) ordered by clock_state."""
    if _m.memory_store is None or not agent_id:
        return None, None

    def _fetch(rel_type):
        try:
            rows = _m.memory_store.list_memory_digests(
                agent_id=agent_id, digest_type="identity",
                current_only=True, limit=1, relationship_type=rel_type,
            )
            return rows[0]["content"] if rows else None
        except Exception:
            return None

    personal = _fetch("personal")
    work = _fetch("work") or _fetch(None)
    if _m.clock_state == "on":
        return work, personal
    return personal, work


def agent_runtime_context(role=None):
    """Build a dict of runtime context fields (time, location, identity) for a role."""
    now_utc = datetime.now(timezone.utc)
    local_tz = _runtime_timezone()
    now_local = now_utc.astimezone(local_tz)
    role_name = (
        (_m.active_tool_caller_name if role is _m.active_tool_caller else None)
        or getattr(role, "runtime_role_name", None)
        or getattr(role, "name", None)
    )
    agent_name = getattr(role, "name", None)
    canonical_id = getattr(role, "runtime_role_name", None) or agent_name
    primary_id, secondary_id = _get_identity_digests(canonical_id)
    context = {
        "agent_name": agent_name,
        "role_name": role_name,
        "session_id": getattr(role, "runtime_session_id", None),
        "current_datetime_utc": now_utc.isoformat(timespec="seconds"),
        "current_datetime_local": now_local.isoformat(timespec="seconds"),
        "current_date_local": now_local.date().isoformat(),
        "timezone": _runtime_timezone_name(local_tz),
        "location": _m.default_location or None,
        "clock_state": _m.clock_state,
        "identity_primary": primary_id,
        "identity_secondary": secondary_id,
    }
    return {key: value for key, value in context.items() if value is not None}


def format_agent_context(role=None):
    """Return a system-prompt snippet containing the agent's runtime context as JSON."""
    context = agent_runtime_context(role)
    return (
        "\nRuntime context available to your tools and decisions:\n"
        f"{json.dumps(context, sort_keys=True)}\n"
        "Use this context for dates, times, location, identity, and session references. "
        "Do not invent missing context; call agent.context if you need to refresh it.\n"
    )


def format_org_chat_context(transcript, limit):
    """Format the most recent `limit` transcript entries as a readable org-chat snippet."""
    if not transcript or limit == 0:
        return ""
    recent = transcript[-limit:]
    lines = []
    for e in recent:
        lines.append(f"[Turn {e.turn}] {e.sender} -> {e.receiver}: {e.prompt[:400]}")
        if e.response:
            lines.append(f"  -> {e.response[:400]}")
    return "\nRecent org chat:\n" + "\n".join(lines) + "\n"


def current_agent_context(employee_dict):
    """Tool handler: return the active caller's runtime context as a JSON string."""
    del employee_dict
    return json.dumps(agent_runtime_context(_m.active_tool_caller), sort_keys=True)


def format_verified_tool_results(system_events):
    """Format a list of tool-result strings into a system-message block for injection."""
    verified_events = [event for event in system_events if isinstance(event, str) and event.strip()]
    if not verified_events:
        return ""
    lines = "\n".join(f"- {event}" for event in verified_events)
    return (
        "Verified runtime results from recent tool calls:\n"
        f"{lines}\n"
        "You may rely on these results as completed runtime facts. "
        "Do not claim other tool-created artifacts or side effects without a verified result."
    )


def deliver_verified_tool_results(role, system_events):
    """Append a verified-tool-results system message to a role's conversation history."""
    content = format_verified_tool_results(system_events)
    if not content or not hasattr(role, "conversation_history"):
        return
    role_history = role.conversation_history.setdefault(getattr(role, "name", ""), [])
    role_history.append({"role": "system", "content": content})


def prepend_verified_tool_results(prompt, system_events):
    """Prepend a verified-tool-results block to a prompt string if any events are present."""
    content = format_verified_tool_results(system_events)
    if not content:
        return prompt
    return f"{content}\n\n{prompt}" if prompt else content

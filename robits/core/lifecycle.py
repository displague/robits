"""Lifecycle management, alarms, and access-control helpers."""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import main as _m
from robits.core.tools import _normalize_capabilities, _normalize_tool_grants
from robits.core.context import agent_runtime_context

LIFECYCLE_STATES = ("proposed", "active", "paused", "retired", "exited")
PROTECTED_ROLE_NAMES = {"CEO", "HR"}
ALARM_RECURRENCES = {"once", "hourly", "daily", "weekly"}


@dataclass
class LifecycleEvent:
    action: str
    agent_name: str
    lifecycle_state: str
    requested_by: str | None = None
    approved_by: str | None = None
    reason: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass
class Alarm:
    alarm_id: str
    agent_name: str
    reminder: str
    due_at: str
    recurrence: str = "once"
    status: str = "active"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _parse_datetime(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Timestamp must be a non-empty ISO datetime string.")
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_datetime(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def validate_role_name(role_name):
    if not isinstance(role_name, str) or not role_name.strip():
        raise ValueError("Role name must be a non-empty string.")
    normalized = role_name.strip()
    if "," in normalized:
        raise ValueError("Role name cannot contain commas.")
    if len(normalized) > 64:
        raise ValueError("Role name cannot exceed 64 characters.")
    return normalized


def validate_role_description(role_description):
    if not isinstance(role_description, str) or not role_description.strip():
        raise ValueError("Role description must be a non-empty string.")
    return role_description.strip()


def _caller_can_act_for_agent(agent_name):
    caller = _m.active_tool_caller
    if caller is None:
        return False
    if _m.active_tool_caller_name == agent_name or getattr(caller, "name", None) == agent_name:
        return True
    capabilities = getattr(caller, "capabilities", set())
    return bool({"operator", "hr"} & capabilities)


def _caller_name_for_error():
    return _m.active_tool_caller_name or getattr(_m.active_tool_caller, "name", "unknown caller")


def _workspace_agent_name(agent_name):
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return None, f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return None, f"Error: Role '{_caller_name_for_error()}' cannot access workspace for '{normalized_name}'."
    return normalized_name, None


def record_lifecycle_event(
    role,
    action,
    lifecycle_state,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    event = LifecycleEvent(
        action=action,
        agent_name=role.name,
        lifecycle_state=lifecycle_state,
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
    )
    if not hasattr(role, "lifecycle_events"):
        role.lifecycle_events = []
    role.lifecycle_events.append(event)
    role.lifecycle_state = lifecycle_state
    if _m.memory_store is not None:
        try:
            _m.memory_store.upsert_agent(
                role.name,
                type(role).__name__,
                display_name=role.name,
                lifecycle_state=lifecycle_state,
            )
            summary = f"[lifecycle:{action}] {role.name} -> {lifecycle_state}"
            if requested_by:
                summary += f" (requested_by={requested_by})"
            if approved_by:
                summary += f" (approved_by={approved_by})"
            if reason:
                summary += f" reason={reason}"
            _m.memory_store.append_memory_entry(
                agent_id=role.name,
                kind="lifecycle",
                content=summary,
                source="system",
            )
        except Exception:
            pass
    return event


def format_lifecycle_actor_text(requested_by=None, approved_by=None):
    actor_parts = []
    if requested_by:
        actor_parts.append(f"requested by {requested_by}")
    if approved_by:
        actor_parts.append(f"approved by {approved_by}")
    return f" ({', '.join(actor_parts)})" if actor_parts else ""


def create_lifecycle_role(
    employee_dict,
    role_name,
    role_description,
    capabilities=None,
    tool_grants=None,
    requested_by=None,
    approved_by=None,
):
    try:
        normalized_name = validate_role_name(role_name)
        normalized_description = validate_role_description(role_description)
        normalized_capabilities = _normalize_capabilities(capabilities)
        normalized_tool_grants = _normalize_tool_grants(tool_grants)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name in employee_dict:
        return f"Error: Role '{normalized_name}' already exists."
    if len(employee_dict) >= _m.HR.max_organization_members:
        return f"Error: The organization has reached its maximum size of {_m.HR.max_organization_members} members."

    new_role = _m.Role(normalized_name, normalized_description, employee_dict)
    new_role.capabilities = normalized_capabilities
    new_role.allowed_tools.update(normalized_tool_grants)
    record_lifecycle_event(
        new_role,
        action="create",
        lifecycle_state="active",
        requested_by=requested_by,
        approved_by=approved_by,
    )
    employee_dict[normalized_name] = new_role
    actor_text = format_lifecycle_actor_text(requested_by, approved_by)
    capabilities_text = (
        f" capabilities={sorted(normalized_capabilities)}" if normalized_capabilities else ""
    )
    return f"Created a new role: {normalized_name} (state: active{capabilities_text}){actor_text}"


def change_lifecycle_state(
    employee_dict,
    role_name,
    lifecycle_state,
    action,
    requested_by=None,
    approved_by=None,
    reason=None,
    allowed_from=None,
):
    try:
        normalized_name = validate_role_name(role_name)
    except ValueError as e:
        return f"Error: {e}"
    if lifecycle_state not in LIFECYCLE_STATES:
        return f"Error: Invalid lifecycle state '{lifecycle_state}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."

    role = employee_dict[normalized_name]
    current_state = getattr(role, "lifecycle_state", "active")
    if allowed_from is not None and current_state not in allowed_from:
        allowed_text = ", ".join(allowed_from)
        return (
            f"Error: Cannot {action} role '{normalized_name}' from lifecycle "
            f"state '{current_state}'. Expected one of: {allowed_text}."
        )
    record_lifecycle_event(
        role,
        action=action,
        lifecycle_state=lifecycle_state,
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
    )
    actor_text = format_lifecycle_actor_text(requested_by, approved_by)
    reason_text = f" Reason: {reason}" if reason else ""
    return f"Updated role '{normalized_name}' to lifecycle state '{lifecycle_state}'{actor_text}.{reason_text}"


def pause_lifecycle_role(
    employee_dict,
    role_name,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    return change_lifecycle_state(
        employee_dict,
        role_name,
        "paused",
        "pause",
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
        allowed_from=("active",),
    )


def retire_lifecycle_role(
    employee_dict,
    role_name,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    return change_lifecycle_state(
        employee_dict,
        role_name,
        "retired",
        "retire",
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
        allowed_from=("active", "paused"),
    )


def _is_role_protected(role_name, role):
    capabilities = getattr(role, "capabilities", set())
    return (
        role_name in PROTECTED_ROLE_NAMES
        or "protected" in capabilities
        or "essential" in capabilities
    )


def archive_lifecycle_role(
    employee_dict,
    role_name,
    requested_by=None,
    approved_by=None,
    reason=None,
):
    try:
        normalized_name = validate_role_name(role_name)
    except ValueError as e:
        return f"Error: {e}"
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    role = employee_dict[normalized_name]
    if _is_role_protected(normalized_name, role):
        return f"Error: Role '{normalized_name}' is protected and cannot be removed from the organization."
    return change_lifecycle_state(
        employee_dict,
        normalized_name,
        "exited",
        "archive",
        requested_by=requested_by,
        approved_by=approved_by,
        reason=reason,
        allowed_from=("active", "paused", "retired"),
    )


def list_lifecycle_roles(employee_dict, include_exited=True):
    rows = []
    for name, role in sorted(employee_dict.items()):
        state = getattr(role, "lifecycle_state", "active")
        if not include_exited and state == "exited":
            continue
        capabilities = sorted(getattr(role, "capabilities", set()))
        tool_grants = sorted(getattr(role, "allowed_tools", set()))
        rows.append(
            {
                "role_name": name,
                "lifecycle_state": state,
                "capabilities": capabilities,
                "tool_grants": tool_grants,
            }
        )
    return json.dumps(rows, sort_keys=True)


def _active_alarms(role):
    return [alarm for alarm in getattr(role, "alarms", []) if alarm.status == "active"]


def create_alarm(employee_dict, agent_name, reminder, due_at, recurrence="once"):
    try:
        normalized_name = validate_role_name(agent_name)
        due = _parse_datetime(due_at)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot manage alarms for '{normalized_name}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    if not isinstance(reminder, str) or not reminder.strip():
        return "Error: Reminder must be a non-empty string."
    recurrence = (recurrence or "once").strip().lower()
    if recurrence not in ALARM_RECURRENCES:
        return f"Error: Invalid recurrence '{recurrence}'."
    now = datetime.now(timezone.utc)
    comparable_due = due.astimezone(timezone.utc) if due.tzinfo else due.replace(tzinfo=timezone.utc)
    if comparable_due <= now:
        context = agent_runtime_context(employee_dict[normalized_name])
        return (
            f"Error: Alarm due_at '{_format_datetime(due)}' is not in the future. "
            f"Current local time is {context.get('current_datetime_local')}."
        )
    role = employee_dict[normalized_name]
    if len(_active_alarms(role)) >= 5:
        return f"Error: Role '{normalized_name}' already has the maximum of 5 active alarms."
    alarm = Alarm(
        alarm_id=f"alarm-{uuid4()}",
        agent_name=normalized_name,
        reminder=reminder.strip(),
        due_at=_format_datetime(due),
        recurrence=recurrence,
    )
    role.alarms.append(alarm)
    return f"Created alarm '{alarm.alarm_id}' for {normalized_name} at {alarm.due_at}."


def list_alarms(employee_dict, agent_name, include_inactive=False):
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot inspect alarms for '{normalized_name}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    alarms = []
    for alarm in getattr(employee_dict[normalized_name], "alarms", []):
        if alarm.status != "active" and not include_inactive:
            continue
        alarms.append(
            {
                "alarm_id": alarm.alarm_id,
                "reminder": alarm.reminder,
                "due_at": alarm.due_at,
                "recurrence": alarm.recurrence,
                "status": alarm.status,
            }
        )
    return json.dumps(alarms, sort_keys=True)


def cancel_alarm(employee_dict, agent_name, alarm_id):
    try:
        normalized_name = validate_role_name(agent_name)
    except ValueError as e:
        return f"Error: {e}"
    if not _caller_can_act_for_agent(normalized_name):
        return f"Error: Role '{_caller_name_for_error()}' cannot manage alarms for '{normalized_name}'."
    if normalized_name not in employee_dict:
        return f"Error: Role '{normalized_name}' not found."
    for alarm in getattr(employee_dict[normalized_name], "alarms", []):
        if alarm.alarm_id == alarm_id:
            alarm.status = "canceled"
            return f"Canceled alarm '{alarm_id}' for {normalized_name}."
    return f"Error: Alarm '{alarm_id}' not found for {normalized_name}."


def due_alarm_reminders(role, now=None):
    now = now or datetime.now(timezone.utc)
    reminders = []
    for alarm in _active_alarms(role):
        due = _parse_datetime(alarm.due_at)
        if due > now:
            continue
        reminders.append(f"Reminder due at {alarm.due_at}: {alarm.reminder}")
        if alarm.recurrence == "once":
            alarm.status = "completed"
        else:
            delta = {
                "hourly": timedelta(hours=1),
                "daily": timedelta(days=1),
                "weekly": timedelta(weeks=1),
            }[alarm.recurrence]
            while due <= now:
                due += delta
            alarm.due_at = _format_datetime(due)
    return reminders

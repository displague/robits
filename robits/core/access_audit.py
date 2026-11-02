"""Accountable access to other personas' workspaces (#140).

Roles with the ``hr`` or ``operator`` capability may open another persona's
private workspace. Each such call must give a reason, is appended to an audit
log outside every workspace (never with file contents), is emitted as a
``workspace.access`` runtime event, and, depending on ROBITS_ACCESS_NOTICE,
is reported to the owner on its next turn.
"""
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from robits.core.config import _config as _m

NOTICE_MODES = ("immediate", "digest", "none")
_REASON_LIMIT = 300
_lock = threading.Lock()

_OPERATION_VERBS = {
    "list": "listed files in",
    "read": "read a file in",
    "write": "wrote a file in",
    "delete": "deleted a file in",
    "search": "searched",
    "shell": "ran a command in",
}


def missing_reason_error(reason):
    """Return an error string when a cross-persona call has no usable reason, else None."""
    if isinstance(reason, str) and reason.strip():
        return None
    return ("Error: Opening another persona's workspace needs a `reason` saying why it is necessary. "
            "The access is logged for review and the owner may be told.")


def record_access(caller, caller_name, owner, operation, path, reason):
    """Append one audit entry, emit it as an event, and queue the owner's notice."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": caller_name,
        "owner": owner,
        "operation": operation,
        "path": str(path or ""),
        "reason": " ".join(str(reason).split())[:_REASON_LIMIT],
    }
    audit_file = Path(getattr(_m, "access_audit_file", None) or "var/access_audit.jsonl")
    with _lock:
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    try:
        from robits.core.io import get_logger
        get_logger().write_event("workspace_access", **entry)
    except Exception:
        pass
    stream = getattr(caller, "runtime_event_stream", None)
    if stream is not None:
        stream.emit("workspace.access", getattr(caller, "runtime_session_id", None), dict(entry),
                    visibility="private")
    if notice_mode() != "none":
        with _lock:
            _notices().setdefault(owner, []).append(entry)
    return entry


def notice_mode():
    mode = str(getattr(_m, "access_notice_mode", "immediate") or "immediate").lower()
    return mode if mode in NOTICE_MODES else "immediate"


def _notices():
    queue = getattr(_m, "access_notices", None)
    if queue is None:
        queue = _m.access_notices = {}
    return queue


def pop_notice(*owner_names):
    """Return the owner's pending notice text (empty if none) and clear it."""
    entries = []
    with _lock:
        for name in dict.fromkeys(n for n in owner_names if n):
            entries.extend(_notices().pop(name, []))
    if not entries or notice_mode() == "none":
        return ""
    if notice_mode() == "digest":
        counts = Counter((e["caller"], e["operation"]) for e in entries)
        parts = [f"{caller} {_OPERATION_VERBS.get(op, op)} it {n} time{'s' if n != 1 else ''}"
                 for (caller, op), n in sorted(counts.items())]
        return f"[Notice: your workspace was accessed since your last turn: {'; '.join(parts)}.]"
    lines = []
    for e in entries:
        where = f" ({e['path']})" if e["path"] else ""
        lines.append(f"[Notice: {e['caller']} {_OPERATION_VERBS.get(e['operation'], e['operation'])} "
                     f"your workspace{where} at {e['ts']}. Reason given: {e['reason']}]")
    return "\n".join(lines)


def recent(limit=20, owner=None, caller=None):
    """Return the most recent audit entries, newest last, optionally filtered."""
    audit_file = Path(getattr(_m, "access_audit_file", None) or "var/access_audit.jsonl")
    try:
        lines = audit_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if owner and entry.get("owner") != owner:
            continue
        if caller and entry.get("caller") != caller:
            continue
        entries.append(entry)
    limit = max(1, min(int(limit or 20), 200))
    return entries[-limit:]


def access_duty_note(role):
    """System-prompt line for roles that can open other personas' workspaces."""
    if not {"hr", "operator"} & set(getattr(role, "capabilities", ()) or ()):
        return ""
    return ("\nYou can open other personas' private workspaces. Each such access needs a reason, "
            "is logged for review, and the owner may be told. Use it only for lifecycle and "
            "wellbeing duties, never out of curiosity.\n")

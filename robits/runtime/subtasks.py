"""Bounded cooperative self-delegation, with isolated resumable chat histories.

No threads: trusted tool execution currently uses process-global caller state.
Task IDs and events are local to the owning runtime session, never org chat.
"""
import json
from types import SimpleNamespace
from uuid import uuid4

from robits.core.config import _config as config
from robits.runtime.mcp import owner
from robits.runtime.task_store import TERMINAL, TaskStore

# The plain settings of a subtask role that a checkpoint keeps. Event streams and the employee
# dictionary are live objects: recovery takes them from the caller instead.
ROLE_FIELDS = ("name", "allowed_tools", "capabilities", "max_tokens", "tool_ceiling", "tool_budget",
               "max_tool_rounds", "temperature", "top_p", "runtime_session_id", "runtime_role_name",
               "runtime_task_depth", "runtime_task_root", "runtime_task_id", "runtime_input_request")


class SubtaskManager:
    def __init__(self, store=None):
        self.tasks = {}
        self.store = store

    def create(self, role, model, messages, parent):
        scope = owner(parent)
        if sum(record["root"] == getattr(parent, "runtime_task_root", scope)
               for record in self.tasks.values()) >= config.subtask_max_tasks:
            raise ValueError("Subtask count budget exceeded")
        task_id = uuid4().hex
        role.runtime_task_id = task_id
        role.runtime_task_root = getattr(parent, "runtime_task_root", scope)
        self.tasks[task_id] = {"task_id": task_id, "owner": scope,
            "root": role.runtime_task_root, "role": role, "model": model,
            "messages": messages, "status": "queued", "result": None, "events": []}
        self.event(task_id, "created", {"depth": role.runtime_task_depth})
        return task_id

    def checkpoint(self, task_id):
        """Save the record, so a restart can find it again (a no-op without a store)."""
        if self.store is None:
            return
        record = self.tasks[task_id]
        role = record["role"]
        self.store.save({
            "task_id": task_id, "owner": record["owner"], "root": record["root"], "model": record["model"],
            "messages": record["messages"], "status": record["status"], "result": record["result"],
            "events": record["events"], "role": {f: getattr(role, f, None) for f in ROLE_FIELDS}})

    def recover(self, caller):
        """Reload this caller's persisted tasks that are not in memory yet.

        Authority is revalidated, not restored: a task's tools are cut down to what ``caller`` may
        use now, and only tasks of the caller's own root come back. Work that was running when
        the process stopped becomes ``interrupted``: it is shown, never re-run, because its tools
        may already have had side effects. Queued and input-required tasks continue as they were.
        """
        if self.store is None:
            return
        from robits.core.tools import role_can_use_tool
        root = getattr(caller, "runtime_task_root", None) or owner(caller)
        for snap in self.store.load_root(root):
            if snap["task_id"] in self.tasks:
                continue
            fields = snap["role"]
            role = SimpleNamespace(**fields, employee_dict=getattr(caller, "employee_dict", None),
                                   runtime_event_stream=getattr(caller, "runtime_event_stream", None),
                                   runtime_tool_results=[])
            registry = config.tool_registry
            if registry is not None and hasattr(caller, "allowed_tools"):
                role.allowed_tools = {name for name in (fields["allowed_tools"] or ())
                                      if name in registry and role_can_use_tool(caller, registry.get(name))}
            if fields.get("runtime_input_request"):
                role.runtime_chat_messages = snap["messages"]
            record = {"task_id": snap["task_id"], "owner": snap["owner"], "root": snap["root"], "role": role,
                      "model": snap["model"], "messages": snap["messages"], "status": snap["status"],
                      "result": snap["result"], "events": snap["events"]}
            self.tasks[snap["task_id"]] = record
            if record["status"] == "running":
                record["status"] = "interrupted"
                record["result"] = "Error: interrupted by a restart; its tools may have run. Not repeated."
                self.event(snap["task_id"], "interrupted", {"result": record["result"]})

    def event(self, task_id, kind, payload):
        record = self.tasks[task_id]
        event = {"sequence": len(record["events"]), "kind": kind, **payload}
        record["events"].append(event)
        self.checkpoint(task_id)
        role = record["role"]
        stream = getattr(role, "runtime_event_stream", None)
        if stream is not None:
            stream.emit(f"subtask.{kind}", getattr(role, "runtime_session_id", None),
                        {"task_id": task_id, "parent": record["owner"][1], **event}, visibility="private")

    def get(self, task_id, caller):
        record = self.tasks.get(task_id)
        if record is None and self.store is not None:
            self.recover(caller)
            record = self.tasks.get(task_id)
        if record is None or not self.descends_from(record, owner(caller)):
            raise ValueError("Unknown subtask for this caller")
        return record

    def descends_from(self, record, caller_scope):
        parents = {owner(task["role"]): task for task in self.tasks.values()}
        seen = set()
        while record and record["task_id"] not in seen:
            if record["owner"] == caller_scope:
                return True
            seen.add(record["task_id"])
            record = parents.get(record["owner"])
        return False

    def run(self, record):
        from robits.core.providers import ChatCompletionsProvider
        if record["status"] != "queued":
            return
        role = record["role"]
        record["status"] = "running"
        self.event(record["task_id"], "running", {})
        try:
            result = ChatCompletionsProvider(config.client, config.tool_registry).generate(
                role, record["model"], record["owner"][1], record["messages"])
            if getattr(role, "runtime_input_request", None):
                record["status"] = "input_required"
                record["messages"] = role.runtime_chat_messages
                self.event(record["task_id"], "input_required", {"request": role.runtime_input_request})
            else:
                record["result"] = result or "Error: sub-agent returned no response."
                record["status"] = "failed" if record["result"].startswith("Error:") else "completed"
                self.event(record["task_id"], record["status"], {"result": record["result"]})
        except Exception as exc:
            record["status"], record["result"] = "failed", f"Error: sub-agent failed: {exc}"
            self.event(record["task_id"], "failed", {"result": record["result"]})

    def brief(self, record):
        """Routine handoff: return findings or a question, without replaying work."""
        return {key: value for key, value in self.view(record).items()
                if key not in {"events", "event_count", "next_sequence"}}

    def view(self, record, after_sequence=None, limit=10):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("event limit must be between 1 and 50")
        if after_sequence is not None and (not isinstance(after_sequence, int) or isinstance(after_sequence, bool)):
            raise ValueError("after_sequence must be an integer")
        events = record["events"][-limit:] if after_sequence is None else [
            e for e in record["events"] if e["sequence"] > after_sequence][:limit]
        # Tool transcripts remain private; inspection returns bounded previews.
        previews = [{k: v[:2048] if isinstance(v, str) else v for k, v in e.items()} for e in events]
        return {key: record[key] for key in ("task_id", "status", "result")} | {
            "events": previews, "event_count": len(record["events"]),
            "next_sequence": events[-1]["sequence"] if events else after_sequence,
            "depth": record["role"].runtime_task_depth,
            "input_request": getattr(record["role"], "runtime_input_request", None),
            "children": [r["task_id"] for r in self.tasks.values()
                         if r["owner"] == owner(record["role"])]}


def get_manager():
    if config.subtask_manager is None:
        store = TaskStore(config.subtask_db, config.subtask_retention_s) if config.subtask_db else None
        config.subtask_manager = SubtaskManager(store)
    return config.subtask_manager


def subtask_control(employee_dict, task_id, action="inspect", response=None, after_sequence=None, limit=10):
    try:
        manager = get_manager()
        record = manager.get(task_id, config.active_tool_caller)
        if action == "run":
            manager.run(record)
        elif action == "respond":
            if record["status"] != "input_required" or not isinstance(response, str) or not response.strip():
                raise ValueError("A waiting task and a non-empty response are required")
            record["messages"].append({"role": "user", "content": response})
            record["role"].runtime_input_request = None
            record["status"] = "queued"
            manager.event(task_id, "response", {"response": response})
        elif action == "cancel":
            if record["status"] not in TERMINAL:
                record["status"] = "cancelled"
                manager.event(task_id, "cancelled", {})
            for child in list(manager.tasks.values()):
                if (manager.descends_from(child, owner(record["role"]))
                        and child["status"] not in TERMINAL):
                    child["status"] = "cancelled"
                    manager.event(child["task_id"], "cancelled", {})
        elif action != "inspect":
            raise ValueError("action must be inspect, run, respond or cancel")
        return json.dumps(manager.view(record, after_sequence, limit) if action == "inspect" else manager.brief(record))
    except Exception as exc:
        return f"Error: {exc}"


def subtask_elicit(employee_dict, message):
    role = config.active_tool_caller
    if not getattr(role, "runtime_task_id", None):
        return "Error: Only subtasks can request input from their parent."
    if not isinstance(message, str) or not message.strip():
        return "Error: message must be a non-empty string."
    role.runtime_input_request = message
    return "Input requested. Execution is paused until the parent responds."

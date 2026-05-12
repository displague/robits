"""Session management, scheduling, and transcript classes."""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from robits.core.config import _config as _m
from termcolor import colored

from robits.memory.sqlite import CHANNEL_ORG_CHAT, SOCIAL_PROFESSIONAL, compute_phase_shift
from robits.core.roles import System, build_employee_dict, parse_tool_instruction, parse_agent_action
from robits.core.context import (
    format_org_chat_context,
    deliver_verified_tool_results,
    prepend_verified_tool_results,
)
from robits.core.lifecycle import due_alarm_reminders
from robits.core.tool_functions import _restore_wait_state, _WAIT_STATE_FILE


@dataclass
class TranscriptEntry:
    """A single turn record in the session transcript."""

    turn: int
    sender: str
    receiver: str
    prompt: str
    response: str
    directed: bool = False
    system_events: list = field(default_factory=list)
    memory_recorded: bool = False


@dataclass
class RuntimeEvent:
    """An immutable event emitted by the runtime and delivered to subscribers."""

    sequence: int
    event_type: str
    session_id: str
    payload: dict
    visibility: str = "public"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class RuntimeEventStream:
    """Ordered log of RuntimeEvents with optional subscriber callbacks."""

    def __init__(self):
        self._events = []
        self._subscribers = []
        self.subscriber_errors = []
        self._sequence = 0

    def subscribe(self, callback):
        """Register a callback to be called on every emitted event; return the callback."""
        self._subscribers.append(callback)
        return callback

    def emit(self, event_type, session_id, payload=None, visibility="public"):
        """Create and store a RuntimeEvent, then notify all subscribers."""
        self._sequence += 1
        event = RuntimeEvent(
            sequence=self._sequence,
            event_type=event_type,
            session_id=session_id,
            payload=payload or {},
            visibility=visibility,
        )
        self._events.append(event)
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception as e:
                self.subscriber_errors.append(
                    {
                        "event_type": event_type,
                        "error": str(e),
                    }
                )
        return event

    def events(self, visibility=None):
        """Return all events, optionally filtered to a specific visibility level."""
        if visibility is None:
            return list(self._events)
        return [event for event in self._events if event.visibility == visibility]


@dataclass
class RoutedMessage:
    """The result of routing a message: the receiving role, the prompt text, and whether it was directed."""

    receiver: object
    prompt: str
    directed: bool = False


class RoundRobinScheduler:
    """Cycles through participant names in order, skipping the last receiver."""

    def __init__(self, participant_names):
        self.participant_names = list(participant_names)
        if not self.participant_names:
            raise ValueError("Scheduler requires at least one participant.")
        self.index = 0

    def next(self, last_receiver_name=None):
        """Return the next participant name, skipping last_receiver_name if possible."""
        for _ in range(len(self.participant_names)):
            name = self.participant_names[self.index % len(self.participant_names)]
            self.index += 1
            if len(self.participant_names) == 1 or name != last_receiver_name:
                return name
        return self.participant_names[(self.index - 1) % len(self.participant_names)]

    def observe(self, participant_name):
        """Advance the index so that participant_name is next in rotation."""
        if participant_name in self.participant_names:
            self.index = self.participant_names.index(participant_name) + 1

    def add_participant(self, participant_name):
        """Add a new participant to the rotation if not already present."""
        if participant_name not in self.participant_names:
            self.participant_names.append(participant_name)

    def remove_participant(self, participant_name):
        """Remove a participant from the rotation, adjusting the index."""
        if participant_name in self.participant_names and len(self.participant_names) > 1:
            self.participant_names.remove(participant_name)
            self.index %= len(self.participant_names)


class Session:
    """Orchestrates a multi-turn conversation between participants, routing messages and managing memory."""

    def __init__(
        self,
        participants=None,
        system=None,
        scheduler=None,
        run_id=None,
        max_turns=None,
        event_stream=None,
        clock_state=None,
    ):
        self.participants = participants if participants is not None else build_employee_dict()
        self.system = system if system is not None else System(self.participants)
        scheduler_names = list(self.participants)
        self.scheduler = scheduler if scheduler is not None else RoundRobinScheduler(scheduler_names)
        self.run_id = run_id or f"run-{uuid4()}"
        self.max_turns = max_turns
        self.event_stream = event_stream if event_stream is not None else RuntimeEventStream()
        self.transcript = []
        self.turns_completed = 0
        self._last_digest_turn = 0
        self._last_digest_at = time.monotonic()
        self._meaningful_turns_completed = 0
        self._last_identity_digest_meaningful_turn = 0
        self._last_goal_digest_meaningful_turn = 0
        self.last_receiver = self.participants.get("CEO") or next(iter(self.participants.values()))
        self._name_to_key = {getattr(p, "name", k): k for k, p in self.participants.items()}
        self._role_to_key = {id(p): k for k, p in self.participants.items()}
        _cs = clock_state or _m.clock_state
        self.clock_state = _cs if _cs in {"on", "off"} else "on"
        self.event_stream.emit(
            "session.created",
            self.run_id,
            {
                "participants": list(self.participants),
                "max_turns": self.max_turns,
                "clock_state": self.clock_state,
            },
        )
        for name, participant in self.participants.items():
            _restore_wait_state(name, participant)

        self._org_workspace = _m._org_workspace
        self._org_chat_channel_id = None
        if _m.memory_store is not None:
            try:
                _m.memory_store.create_session(self.run_id)
                for name, participant in self.participants.items():
                    role_type = type(participant).__name__
                    _m.memory_store.upsert_agent(name, role_type, display_name=name)
                self._org_chat_channel_id = _m.memory_store.get_or_create_channel(
                    CHANNEL_ORG_CHAT,
                    social_distance=SOCIAL_PROFESSIONAL,
                )
            except Exception:
                pass

    def _clear_wait_state_file(self, role):
        """Remove the persisted wait-state file for role if agent_workspace_store is available."""
        agent_name = self._role_to_key.get(id(role))
        if agent_name and _m.agent_workspace_store is not None:
            try:
                _m.agent_workspace_store.delete(agent_name, _WAIT_STATE_FILE)
            except Exception:
                pass

    def _build_wait_summary(self, role):
        """Clear a role's wait state and return a summary of what happened while it was waiting."""
        started = getattr(role, "wait_started_turn", None) or 0
        since = self.transcript[started:]
        role.waiting_until = None
        role.wait_started_turn = None
        role.wait_clock_state = None
        self._clear_wait_state_file(role)
        if not since:
            return "Your wait has ended. Nothing notable happened while you were waiting."
        lines = ["Your wait has ended. Here is what happened while you were waiting:"]
        for entry in since:
            if entry.response and entry.response.strip():
                snippet = entry.response.strip()[:200]
                lines.append(f"[Turn {entry.turn}] {entry.sender} → {entry.receiver}: {snippet}")
        if len(lines) == 1:
            return "Your wait has ended. Nothing notable happened while you were waiting."
        return "\n".join(lines)

    def _interrupt_wait_for_phase(self, role):
        """Cancel a role's wait early if the circadian clock state has changed since it began."""
        wait_cs = getattr(role, "wait_clock_state", None)
        current_cs = self.clock_state
        if wait_cs is not None and wait_cs != current_cs:
            role.waiting_until = None
            role.wait_started_turn = None
            role.wait_clock_state = None
            self._clear_wait_state_file(role)
            return True
        return False

    def route_message(self, message, last_receiver_name=None):
        """Determine the next receiver for message; honour 'Name, prompt' directed syntax."""
        prompt_split = message.split(",", 1) if isinstance(message, str) else []
        if len(prompt_split) > 1:
            receiver_name = prompt_split[0].strip()
            if receiver_name in self.participants:
                print(colored(f"// Directed to {receiver_name}", "grey"))
                self.scheduler.observe(receiver_name)
                self.event_stream.emit(
                    "message.routed",
                    self.run_id,
                    {
                        "receiver": receiver_name,
                        "directed": True,
                    },
                )
                return RoutedMessage(self.participants[receiver_name], prompt_split[1].strip(), True)

        now = datetime.now(timezone.utc)
        tried: set = set()
        receiver_name = self.scheduler.next(last_receiver_name)
        while receiver_name not in tried:
            receiver = self.participants.get(receiver_name)
            if receiver is None:
                break
            wu = getattr(receiver, "waiting_until", None)
            if wu is None or now >= wu or self._interrupt_wait_for_phase(receiver):
                break
            tried.add(receiver_name)
            receiver_name = self.scheduler.next(receiver_name)

        self.event_stream.emit(
            "message.routed",
            self.run_id,
            {
                "receiver": receiver_name,
                "directed": False,
            },
        )
        return RoutedMessage(self.participants[receiver_name], message, False)

    def process_tool_instruction(self, message, sender=None):
        """Execute a JSON exec instruction found in message; return a list of result strings."""
        if not isinstance(message, str) or message == "":
            return []
        tool_instruction = parse_tool_instruction(message)
        if tool_instruction is None or tool_instruction == "":
            return []
        try:
            obj = json.loads(tool_instruction)
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict) or "exec" not in obj:
            return []

        system_response = self.system.interact(tool_instruction, caller=sender)
        print(colored(f"System: {system_response}", "blue"))
        self.event_stream.emit(
            "tool.executed",
            self.run_id,
            {
                "instruction": tool_instruction,
                "response": system_response,
            },
        )
        if system_response is not None and system_response != "" and "Ops" in self.participants:
            ops = self.participants["Ops"]
            if hasattr(ops, "update_group_conversations"):
                ops.update_group_conversations({"role": "system", "content": system_response})
        return [system_response]

    def process_agent_action(self, message, sender=None):
        """Parse a structured agent action from message; return (response_text, system_events)."""
        if not isinstance(message, str) or message.strip() == "":
            return "", []
        action = parse_agent_action(message)
        if action is None:
            return message, []
        action_type = action.get("action")
        if action_type == "wait":
            self.event_stream.emit(
                "agent.waited",
                self.run_id,
                {
                    "agent": getattr(sender, "name", None),
                },
            )
            return "", []
        if action_type == "think":
            content = action.get("content", "")
            if isinstance(content, str) and content.strip():
                self.record_thought(getattr(sender, "name", "unknown"), content.strip())
            return "", []
        if action_type == "reply":
            content = action.get("content", "")
            return (content if isinstance(content, str) else ""), []
        if "exec" in action:
            system_events = self.process_tool_instruction(json.dumps(action), sender=sender)
            return "", system_events
        return message, []

    def record_turn(self, sender, receiver, prompt, response, directed=False, system_events=None):
        """Append a TranscriptEntry, update memory, and trigger auto-digest if due."""
        meaningful_response = bool(str(response or "").strip())
        system_events = system_events or []
        entry = TranscriptEntry(
            turn=self.turns_completed + 1,
            sender=sender,
            receiver=receiver,
            prompt=prompt,
            response=response,
            directed=directed,
            system_events=system_events,
            memory_recorded=meaningful_response,
        )
        self.transcript.append(entry)
        self.turns_completed += 1
        if meaningful_response:
            self._meaningful_turns_completed += 1
        self.event_stream.emit(
            "message.recorded",
            self.run_id,
            {
                "turn": entry.turn,
                "sender": entry.sender,
                "receiver": entry.receiver,
                "directed": entry.directed,
                "system_event_count": len(entry.system_events),
            },
        )
        if _m.memory_store is not None:
            canonical_sender = self._canonical_agent_id(sender)
            canonical_receiver = self._canonical_agent_id(receiver)
            try:
                sender_phase = _m.memory_store.get_agent_phase(canonical_sender)
                receiver_phase = _m.memory_store.get_agent_phase(canonical_receiver)
                if meaningful_response and prompt:
                    _m.memory_store.append_message(
                        session_id=self.run_id,
                        sender_agent_id=canonical_sender,
                        receiver_agent_id=canonical_receiver,
                        content=prompt,
                        kind="message",
                        channel_id=self._org_chat_channel_id,
                        sender_phase=sender_phase,
                    )
                if meaningful_response and response:
                    _m.memory_store.append_message(
                        session_id=self.run_id,
                        sender_agent_id=canonical_receiver,
                        receiver_agent_id=canonical_sender,
                        content=response,
                        kind="message",
                        channel_id=self._org_chat_channel_id,
                        sender_phase=receiver_phase,
                    )
                if self._org_chat_channel_id is not None and sender_phase is not None and receiver_phase is not None:
                    try:
                        social_distance = _m.memory_store.get_channel_social_distance(
                            self._org_chat_channel_id
                        )
                        if social_distance is not None:
                            shifted = compute_phase_shift(
                                receiver_phase, sender_phase, social_distance
                            )
                            _m.memory_store.set_agent_phase(canonical_receiver, shifted)
                    except Exception:
                        pass
            except Exception:
                pass
            digest_reasons = self._auto_digest_reasons()
            if digest_reasons:
                self._auto_digest(digest_reasons)
            if (
                _m.org_digest_interval > 0
                and self.turns_completed % _m.org_digest_interval == 0
            ):
                self._auto_org_digest()
            if (
                _m.memory_identity_digest_interval > 0
                and meaningful_response
                and self._meaningful_turns_completed - self._last_identity_digest_meaningful_turn
                >= _m.memory_identity_digest_interval
            ):
                self._auto_state_digest("identity")
                self._last_identity_digest_meaningful_turn = self._meaningful_turns_completed
            if (
                _m.memory_goal_digest_interval > 0
                and meaningful_response
                and self._meaningful_turns_completed - self._last_goal_digest_meaningful_turn
                >= _m.memory_goal_digest_interval
            ):
                self._auto_state_digest("goal_short_term")
                self._last_goal_digest_meaningful_turn = self._meaningful_turns_completed
        self._write_org_chat_jsonl(entry)
        return entry

    def _auto_digest_reasons(self):
        """Return a list of trigger reason strings if a digest should be created now."""
        reasons = []
        meaningful_window = [
            e for e in self.transcript[self._last_digest_turn:]
            if e.memory_recorded
        ]
        if not meaningful_window:
            return reasons
        if (
            _m.memory_digest_interval > 0
            and self.turns_completed - self._last_digest_turn >= _m.memory_digest_interval
        ):
            reasons.append("turn_interval")
        if _m.memory_digest_context_chars > 0:
            chars = sum(len(e.prompt or "") + len(e.response or "") for e in meaningful_window)
            if chars >= _m.memory_digest_context_chars:
                reasons.append("context_chars")
        if _m.memory_digest_elapsed_seconds > 0:
            elapsed = time.monotonic() - self._last_digest_at
            if elapsed >= _m.memory_digest_elapsed_seconds:
                reasons.append("elapsed_seconds")
        return reasons

    def _auto_digest(self, reasons=None):
        """Create an episodic memory digest from meaningful turns since the last digest."""
        if _m.memory_digest_interval > 0 and not reasons:
            window = [e for e in self.transcript[-_m.memory_digest_interval:] if e.memory_recorded]
        else:
            window = [
                e for e in self.transcript[self._last_digest_turn:]
                if e.memory_recorded
            ]
        lines = []
        for e in window:
            if e.prompt:
                lines.append(f"[turn {e.turn}] {e.sender} -> {e.receiver}: {e.prompt[:512]}")
            if e.response:
                lines.append(f"[turn {e.turn}] {e.receiver}: {e.response[:512]}")
        content = "\n".join(lines)
        if not content.strip():
            return
        source_refs = []
        try:
            msg_ids = _m.memory_store.list_recent_message_ids(
                self.run_id, max(2, len(window) * 2)
            )
            source_refs = [
                {"source_table": "messages", "source_id": mid}
                for mid in msg_ids
            ]
        except Exception:
            pass
        if not source_refs:
            return
        try:
            for agent_id in list(self.participants):
                _m.memory_store.append_memory_digest(
                    content=content,
                    source_refs=source_refs,
                    agent_id=agent_id,
                    session_id=self.run_id,
                    digest_type="episodic",
                    accessibility="agent",
                    system_only=False,
                    metadata={"trigger_reasons": list(reasons or ["turn_interval"])},
                )
            self._last_digest_turn = self.turns_completed
            self._last_digest_at = time.monotonic()
        except Exception:
            pass

    def _auto_state_digest(self, digest_type):
        """Create an identity or goal digest checkpoint for all participants."""
        if _m.memory_store is None:
            return
        source_refs = []
        try:
            msg_ids = _m.memory_store.list_recent_message_ids(
                self.run_id,
                max(2, _m.memory_digest_interval * 2 or 10),
            )
            source_refs = [{"source_table": "messages", "source_id": mid} for mid in msg_ids]
        except Exception:
            return
        if not source_refs:
            return
        label = "identity" if digest_type == "identity" else "short-term goal"
        window = [e for e in self.transcript if e.memory_recorded][-10:]
        content_lines = [
            f"[turn {e.turn}] {e.sender}->{e.receiver}: {e.response[:300]}"
            for e in window
            if e.response
        ]
        if not content_lines:
            return
        for agent_id in list(self.participants):
            try:
                _m.memory_store.append_memory_digest(
                    content=f"Automatic {label} checkpoint:\n" + "\n".join(content_lines),
                    source_refs=source_refs,
                    agent_id=agent_id,
                    session_id=self.run_id,
                    digest_type=digest_type,
                    accessibility="agent",
                    system_only=False,
                    metadata={"trigger_reasons": [f"{digest_type}_interval"]},
                )
            except Exception:
                pass

    def _write_org_chat_jsonl(self, entry):
        """Append a transcript entry as a JSONL line to the org workspace chat log."""
        if self._org_workspace is None:
            return
        line = json.dumps({
            "turn": entry.turn,
            "sender": entry.sender,
            "receiver": entry.receiver,
            "prompt": entry.prompt,
            "response": entry.response,
        })
        try:
            self._org_workspace.write("org", "org_chat.jsonl", line + "\n", append=True)
        except Exception:
            pass

    def _auto_org_digest(self):
        """Create an org-chat memory digest for all participants from recent transcript turns."""
        if _m.memory_store is None:
            return
        window = self.transcript[-_m.org_digest_interval:]
        lines = [
            f"[turn {e.turn}] {e.sender}->{e.receiver}: {e.prompt[:300]} | {e.response[:300]}"
            for e in window
            if e.memory_recorded
        ]
        if not lines:
            return
        content = "Org chat digest:\n" + "\n".join(lines)
        source_refs = []
        try:
            msg_ids = _m.memory_store.list_recent_message_ids_by_channel(
                self.run_id, _m.org_digest_interval * 2, self._org_chat_channel_id
            )
            source_refs = [{"source_table": "messages", "source_id": mid} for mid in msg_ids]
        except Exception:
            pass
        if not source_refs:
            return
        for agent_id in list(self.participants):
            try:
                _m.memory_store.append_memory_digest(
                    content=content,
                    source_refs=source_refs,
                    agent_id=agent_id,
                    session_id=self.run_id,
                    digest_type="episodic",
                    conversation_type="org_chat",
                    accessibility="agent",
                    system_only=False,
                )
            except Exception:
                pass

    def record_thought(self, agent_name, content, visibility="private"):
        """Persist a private thought to memory and emit a thought.recorded event."""
        if _m.memory_store is not None:
            try:
                _m.memory_store.append_thought(
                    agent_id=self._canonical_agent_id(agent_name),
                    content=content,
                    session_id=self.run_id,
                    visibility=visibility,
                )
            except Exception:
                pass
        return self.event_stream.emit(
            "thought.recorded",
            self.run_id,
            {
                "agent": agent_name,
                "content": content,
            },
            visibility=visibility,
        )

    def _canonical_agent_id(self, name):
        """Resolve a role display name to its participant dict key for FK-safe storage."""
        return self._name_to_key.get(name, name)

    def sync_scheduler_participants(self):
        """Add active roles to and remove non-active roles from the scheduler."""
        for name, participant in self.participants.items():
            if getattr(participant, "lifecycle_state", "active") == "active":
                self.scheduler.add_participant(name)
            else:
                self.scheduler.remove_participant(name)

    def prepare_agent_runtime(self, role):
        """Attach session-level runtime attributes (event stream, session ID, clock state) to a role."""
        role.runtime_event_stream = self.event_stream
        role.runtime_session_id = self.run_id
        role.runtime_tool_results = []
        role.runtime_clock_state = self.clock_state
        for name, participant in self.participants.items():
            if participant is role:
                role.runtime_role_name = name
                return
        role.runtime_role_name = getattr(role, "name", None)

    def recent_system_events(self, limit=5):
        """Return the most recent non-empty system event strings from the transcript."""
        events = []
        for entry in reversed(self.transcript):
            for event in reversed(entry.system_events):
                if isinstance(event, str) and event.strip():
                    events.append(event)
                    if len(events) >= limit:
                        return list(reversed(events))
        return list(reversed(events))

    def step(self, message):
        """Execute one turn: route message, run the agent, record the transcript entry."""
        sender = self.last_receiver
        self.sync_scheduler_participants()
        _m.active_session_transcript_length = len(self.transcript)
        routed = self.route_message(message, sender.name)
        system_events = self.process_tool_instruction(message, sender=sender)
        if system_events:
            deliver_verified_tool_results(sender, system_events)
        self.sync_scheduler_participants()
        prompt = routed.prompt
        now = datetime.now(timezone.utc)
        wu = getattr(routed.receiver, "waiting_until", None)
        if wu is not None and now >= wu:
            wait_summary = self._build_wait_summary(routed.receiver)
            prompt = wait_summary + "\n\n" + prompt
        reminders = due_alarm_reminders(routed.receiver)
        if reminders:
            prompt = "\n".join(reminders + [prompt])
        raw_prompt = prepend_verified_tool_results(
            prompt,
            self.recent_system_events() + system_events,
        )
        effective_org_lines = _m.org_chat_context_lines if self.clock_state == "on" else 0
        org_context = format_org_chat_context(self.transcript, effective_org_lines)
        model_prompt = org_context + raw_prompt if org_context else raw_prompt
        self.prepare_agent_runtime(routed.receiver)
        response = routed.receiver.interact(sender.name, model_prompt)
        response = "" if response is None else response
        native_tool_events = list(getattr(routed.receiver, "runtime_tool_results", []))
        if native_tool_events:
            system_events.extend(native_tool_events)
            deliver_verified_tool_results(routed.receiver, native_tool_events)
        response, response_events = self.process_agent_action(response, sender=routed.receiver)
        if response_events:
            system_events.extend(response_events)
            deliver_verified_tool_results(routed.receiver, response_events)
            self.sync_scheduler_participants()
        if routed.receiver.name != "CEO" and response != "":
            print(colored(f"{routed.receiver.name} responds: {response}", "cyan"))
        self.record_turn(
            sender=sender.name,
            receiver=routed.receiver.name,
            prompt=raw_prompt,
            response=response,
            directed=routed.directed,
            system_events=system_events,
        )
        self.last_receiver = routed.receiver
        return response

    def run(self, initial_message=None, max_turns=None):
        """Run the session loop until max_turns is reached or a SystemExit is raised."""
        effective_max_turns = self.max_turns if max_turns is None else max_turns
        last_response = (
            initial_message
            if initial_message is not None
            else self.last_receiver.interact()
        )
        if last_response is None:
            last_response = ""

        try:
            while effective_max_turns is None or self.turns_completed < effective_max_turns:
                last_response = self.step(last_response)
        except SystemExit:
            print(colored("\nSession ended by CEO.", "yellow"))

        self.event_stream.emit(
            "session.completed",
            self.run_id,
            {
                "turns_completed": self.turns_completed,
            },
        )
        if _m.memory_store is not None:
            try:
                _m.memory_store.end_session(self.run_id)
            except Exception:
                pass
        return self

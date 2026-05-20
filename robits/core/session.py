"""Session management, scheduling, and transcript classes."""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from robits.core.config import _config as _m
from termcolor import colored

from robits.memory.sqlite import (
    CHANNEL_AGENT_DM,
    CHANNEL_AGENT_THOUGHT,
    CHANNEL_ORG_CHAT,
    SOCIAL_FAMILIAL,
    SOCIAL_PROFESSIONAL,
    compute_phase_shift,
)
from robits.core.roles import System, build_employee_dict, parse_tool_instruction, parse_agent_action
from robits.core.context import (
    format_org_chat_context,
    deliver_verified_tool_results,
    prepend_verified_tool_results,
)
from robits.core.lifecycle import due_alarm_reminders
from robits.core.tool_functions import _restore_wait_state, _WAIT_STATE_FILE
from robits.core.persona import seed_persona
from robits.core.io import get_logger


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

    def unsubscribe(self, callback):
        """Unsubscribe a callback from the event stream."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

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


_CLOCK_TEMP_MULTIPLIER = {"on": 0.6, "break": 0.9, "off": 1.3}


def _modulate_temperature(role, clock_state):
    """Adjust role.temperature and role.top_p around their base values for clock state.

    Initialises base values from current values on first call so repeated calls
    always modulate from the same stable baseline.
    """
    if not hasattr(role, "base_temperature"):
        role.base_temperature = getattr(role, "temperature", 0.7)
    multiplier = _CLOCK_TEMP_MULTIPLIER.get(clock_state, 1.0)
    role.temperature = max(0.05, min(1.0, role.base_temperature * multiplier))
    if hasattr(role, "base_top_p"):
        role.top_p = max(0.1, min(1.0, role.base_top_p * multiplier))


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
        self.participants = (
            participants if participants is not None
            else build_employee_dict(_m.persona_entries or None)
        )
        self.system = system if system is not None else System(self.participants)
        scheduler_names = list(self.participants)
        self.scheduler = scheduler if scheduler is not None else RoundRobinScheduler(scheduler_names)
        self.run_id = run_id or f"run-{uuid4()}"
        self.max_turns = max_turns
        self.event_stream = event_stream if event_stream is not None else RuntimeEventStream()
        
        # Initialize token usage tracking and DB event persistence
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._db_ready = False

        def _handle_event(event):
            if event.session_id != self.run_id:
                return

            if event.event_type == "token_usage":
                self.prompt_tokens += event.payload.get("prompt_tokens", 0) or 0
                self.completion_tokens += event.payload.get("completion_tokens", 0) or 0
                self.total_tokens += event.payload.get("total_tokens", 0) or 0

            if self._db_ready and _m.memory_store is not None:
                try:
                    _m.memory_store.append_runtime_event_object(event)
                except Exception:
                    pass

        self._event_callback = _handle_event
        self.event_stream.subscribe(self._event_callback)

        self.transcript = []
        self.turns_completed = 0
        self._last_digest_turn = 0
        self._last_digest_at = time.monotonic()
        self._meaningful_turns_completed = 0
        self._last_identity_digest_meaningful_turn = 0
        self._last_goal_digest_meaningful_turn = 0
        self.last_receiver = self.participants.get("CEO") or next(iter(self.participants.values()))
        _name_map: dict = {}
        for k, p in self.participants.items():
            for form in (
                k,
                getattr(p, "name", None),
                getattr(p, "role_key", None),
                getattr(p, "full_name", None),
                getattr(p, "first_name", None),
            ):
                if form and isinstance(form, str):
                    _name_map.setdefault(form.lower(), k)
        self._name_to_key = _name_map
        self._role_to_key = {id(p): k for k, p in self.participants.items()}
        _cs = clock_state or _m.clock_state
        self.clock_state = _cs if _cs in {"on", "off", "break"} else "on"

        # Pre-create the database session before emitting "session.created" event
        if _m.memory_store is not None:
            try:
                _m.memory_store.create_session(self.run_id)
                self._db_ready = True
            except Exception:
                pass

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
        self._thought_channel_ids: dict = {}
        self._dm_channel_ids: dict = {}

        if _m.memory_store is not None:
            try:
                for name, participant in self.participants.items():
                    role_type = type(participant).__name__
                    persona = _m.persona_entries.get(name) or {}
                    full_name = persona.get("full_name") if isinstance(persona, dict) else None
                    
                    metadata = {}
                    if isinstance(persona, dict):
                        if persona.get("preprompt_work"):
                            metadata["preprompt_work"] = persona["preprompt_work"]
                        if persona.get("preprompt_personal"):
                            metadata["preprompt_personal"] = persona["preprompt_personal"]
                    # Add fallbacks from participant if missing in persona
                    if "preprompt_work" not in metadata and hasattr(participant, "preprompt_work"):
                        metadata["preprompt_work"] = participant.preprompt_work
                    if "preprompt_personal" not in metadata and hasattr(participant, "preprompt_personal"):
                        metadata["preprompt_personal"] = participant.preprompt_personal

                    _m.memory_store.upsert_agent(
                        name, role_type,
                        display_name=full_name or name,
                        username=name,
                        full_name=full_name,
                        first_name=getattr(participant, "first_name", None),
                        last_name=getattr(participant, "last_name", None),
                        metadata=metadata,
                    )
                    entries = persona.get("entries") if isinstance(persona, dict) else persona
                    if entries:
                        seed_persona(_m.memory_store, name, entries)
                self._org_chat_channel_id = _m.memory_store.get_or_create_channel(
                    CHANNEL_ORG_CHAT,
                    social_distance=SOCIAL_PROFESSIONAL,
                )
            except Exception:
                pass

    def _get_thought_channel_id(self, agent_name):
        """Return the agent_thought channel ID for agent_name, creating it if needed."""
        if _m.memory_store is None:
            return None
        if agent_name not in self._thought_channel_ids:
            try:
                cid = _m.memory_store.get_or_create_channel(
                    CHANNEL_AGENT_THOUGHT,
                    participants=[agent_name],
                    visibility="private",
                    social_distance=0.0,
                )
                self._thought_channel_ids[agent_name] = cid
            except Exception:
                return None
        return self._thought_channel_ids.get(agent_name)

    def _get_dm_channel_id(self, agent_a, agent_b):
        """Return the agent_dm channel ID for a pair of agents, creating it if needed."""
        if _m.memory_store is None:
            return None
        key = frozenset((agent_a, agent_b))
        if key not in self._dm_channel_ids:
            try:
                cid = _m.memory_store.get_or_create_channel(
                    CHANNEL_AGENT_DM,
                    participants=sorted([agent_a, agent_b]),
                    visibility="private",
                    social_distance=SOCIAL_FAMILIAL,
                )
                self._dm_channel_ids[key] = cid
            except Exception:
                return None
        return self._dm_channel_ids.get(key)

    def _clear_wait_state_file(self, role):
        """Remove the persisted wait-state file for role if agent_workspace_store is available."""
        agent_name = self._role_to_key.get(id(role))
        if agent_name and _m.agent_workspace_store is not None:
            try:
                _m.agent_workspace_store.delete(agent_name, _WAIT_STATE_FILE)
            except Exception:
                pass

    def _embed_pending(self):
        """Backfill embeddings for any unembedded memory records (sleep-cycle task)."""
        if _m.memory_store is None:
            return
        from robits.core.embeddings import embedding_model_name
        model = embedding_model_name()
        if model:
            try:
                _m.memory_store.embed_pending(model)
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
        self._embed_pending()
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
        """Determine the next receiver for message; honour 'Name, prompt', 'Name: prompt', and '@Name prompt' directed syntax."""
        directed_receiver = None
        remaining_prompt = message

        if isinstance(message, str) and message:
            import re as _re
            m = _re.match(r"^@?([a-zA-Z0-9_\s]+?)\s*[:,]\s*(.*)$", message)
            if not m:
                m = _re.match(r"^@([a-zA-Z0-9_]+?)\s+(.*)$", message)
            if m:
                receiver_token = m.group(1).strip()
                resolved_key = self._name_to_key.get(receiver_token.lower())
                if resolved_key is not None:
                    directed_receiver = resolved_key
                    remaining_prompt = m.group(2).strip()

        if directed_receiver is not None:
            print(colored(f"// Directed to {directed_receiver}", "grey"))
            self.scheduler.observe(directed_receiver)
            self.event_stream.emit(
                "message.routed",
                self.run_id,
                {
                    "receiver": directed_receiver,
                    "directed": True,
                },
            )
            return RoutedMessage(self.participants[directed_receiver], remaining_prompt, True)

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
        get_logger().write_event("system_result", content=system_response)
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

    def record_turn(self, sender, receiver, prompt, response, directed=False, system_events=None, thinking=None):
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
            msg_channel_id = (
                self._get_dm_channel_id(canonical_sender, canonical_receiver)
                if directed
                else self._org_chat_channel_id
            )
            msg_visibility = "private" if directed else "public"
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
                        visibility=msg_visibility,
                        channel_id=msg_channel_id,
                        sender_phase=sender_phase,
                    )
                response_msg_id = None
                if meaningful_response and response:
                    response_msg_id = _m.memory_store.append_message(
                        session_id=self.run_id,
                        sender_agent_id=canonical_receiver,
                        receiver_agent_id=canonical_sender,
                        content=response,
                        kind="message",
                        visibility=msg_visibility,
                        channel_id=msg_channel_id,
                        sender_phase=receiver_phase,
                    )
                if thinking and response_msg_id is not None:
                    thought_channel_id = self._get_thought_channel_id(canonical_receiver)
                    try:
                        _m.memory_store.append_thought(
                            agent_id=canonical_receiver,
                            content=thinking,
                            session_id=self.run_id,
                            visibility="private",
                            source="model_thinking",
                            channel_id=thought_channel_id,
                            metadata={"linked_message_id": response_msg_id},
                        )
                    except Exception as _e:
                        self.event_stream.emit(
                            "thought.store_failed",
                            self.run_id,
                            {"agent_id": canonical_receiver, "error": str(_e)},
                        )
                if msg_channel_id is not None and sender_phase is not None and receiver_phase is not None:
                    try:
                        social_distance = _m.memory_store.get_channel_social_distance(
                            msg_channel_id
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
            agents = list(self.participants)
            for agent_id in agents:
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
            trigger = ",".join(reasons or ["turn_interval"])
            chars = len(content)
            print(colored(
                f"[memory] episodic digest — {len(agents)} agent(s) turn {self.turns_completed}"
                f" ({len(window)} turns, {chars} chars, trigger: {trigger})",
                "dark_grey",
            ))
            get_logger().write_event("memory_digest", kind="episodic", agents=len(agents), turn=self.turns_completed, window=len(window), chars=chars, trigger=trigger)
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
        agents = list(self.participants)
        for agent_id in agents:
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
        print(colored(
            f"[memory] {digest_type} digest — {len(agents)} agent(s) turn {self.turns_completed}",
            "dark_grey",
        ))
        get_logger().write_event("memory_digest", kind=digest_type, agents=len(agents), turn=self.turns_completed)

    def _write_org_chat_jsonl(self, entry):
        """Append a non-directed transcript entry as a JSONL line to the org workspace chat log."""
        if self._org_workspace is None or getattr(entry, "directed", False):
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
        agents = list(self.participants)
        for agent_id in agents:
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
        print(colored(
            f"[memory] org chat digest — {len(agents)} agent(s) turn {self.turns_completed}"
            f" ({len(lines)} messages)",
            "dark_grey",
        ))
        get_logger().write_event("memory_digest", kind="org_chat", agents=len(agents), turn=self.turns_completed, messages=len(lines))

    def record_thought(self, agent_name, content, visibility="private"):
        """Persist a private thought to memory and emit a thought.recorded event."""
        if _m.memory_store is not None:
            canonical = self._canonical_agent_id(agent_name)
            try:
                _m.memory_store.append_thought(
                    agent_id=canonical,
                    content=content,
                    session_id=self.run_id,
                    visibility=visibility,
                    channel_id=self._get_thought_channel_id(canonical),
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
        return self._name_to_key.get(name.lower(), name) if isinstance(name, str) else name

    def _detect_mentions(self, message: str) -> set:
        """Return the set of participant keys mentioned in message.

        Matching rules (in priority order):
        - ``@username``: case-insensitive, always matched.
        - Full name (multi-word): phrase match on normalised text, always matched.
        - Single-word username / first / last name: only matched when the token
          appears CAPITALISED in the original message, reducing false positives for
          common English words (will, may, mark, grace, …).
        """
        import re as _re
        if not isinstance(message, str) or not message:
            return set()
        normalised = _re.sub(r"[^a-z0-9@_]+", " ", message.lower())
        mentioned: set = set()
        if _m.memory_store is not None:
            try:
                name_tokens = _m.memory_store.get_agent_name_tokens()
                for agent_id, tokens in name_tokens.items():
                    if agent_id not in self.participants:
                        continue
                    for token in tokens:
                        if not token:
                            continue
                        norm_token = _re.sub(r"[^a-z0-9@_]+", " ", token.lower()).strip()
                        if not norm_token:
                            continue
                        # @username — always match
                        if f"@{norm_token}" in normalised:
                            mentioned.add(agent_id)
                            break
                        # Multi-word full name — phrase match
                        if " " in norm_token and f" {norm_token} " in f" {normalised} ":
                            mentioned.add(agent_id)
                            break
                        # Single-word token — only if capitalised in original text
                        cap = token[0].upper() + token[1:].lower() if len(token) > 1 else token.upper()
                        if _re.search(r"\b" + _re.escape(cap) + r"\b", message):
                            mentioned.add(agent_id)
                            break
            except Exception:
                pass
        return mentioned

    def _effective_clock_state(self) -> str:
        """Return 'break' if inside a configured break window and base state is 'on', else clock_state.

        Scheduled breaks only promote 'on' → 'break'; they never override 'off' (which
        is more restrictive than 'break' and has a higher temperature multiplier).
        """
        if self.clock_state == "on":
            schedule = getattr(_m, "break_schedule", [])
            if schedule:
                now = datetime.now(timezone.utc).astimezone().strftime("%H:%M")
                for start, end in schedule:
                    if start <= now < end:
                        return "break"
        return self.clock_state

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
        role.runtime_thinking = None
        effective_clock = self._effective_clock_state()
        role.runtime_clock_state = effective_clock
        _modulate_temperature(role, effective_clock)
        for name, participant in self.participants.items():
            if participant is role:
                role.runtime_role_name = name
                return
        role.runtime_role_name = getattr(role, "name", None)

    def _decay_agent_adjustments(self, agent_id):
        """Decrement turn counter for active preprompt adjustments and clear them if expired."""
        if _m.memory_store is None or not agent_id:
            return
        try:
            if not hasattr(_m.memory_store, "get_agent_metadata") or not hasattr(_m.memory_store, "update_agent_metadata"):
                return
            metadata = _m.memory_store.get_agent_metadata(agent_id)
            updated = False
            
            # If the clock state transitioned, clear the opposing mode's adjustments
            effective_clock = self._effective_clock_state()
            if effective_clock == "on" and (metadata.get("personal_adjustment") is not None):
                metadata["personal_adjustment"] = None
                metadata["personal_adjustment_turns"] = None
                updated = True
            elif effective_clock in ("off", "break") and (metadata.get("work_adjustment") is not None):
                metadata["work_adjustment"] = None
                metadata["work_adjustment_turns"] = None
                updated = True
                
            # Decay work adjustment
            w_turns = metadata.get("work_adjustment_turns")
            if w_turns is not None:
                try:
                    w_turns = int(w_turns) - 1
                    if w_turns <= 0:
                        metadata["work_adjustment"] = None
                        metadata["work_adjustment_turns"] = None
                    else:
                        metadata["work_adjustment_turns"] = w_turns
                    updated = True
                except Exception:
                    pass
                    
            # Decay personal adjustment
            p_turns = metadata.get("personal_adjustment_turns")
            if p_turns is not None:
                try:
                    p_turns = int(p_turns) - 1
                    if p_turns <= 0:
                        metadata["personal_adjustment"] = None
                        metadata["personal_adjustment_turns"] = None
                    else:
                        metadata["personal_adjustment_turns"] = p_turns
                    updated = True
                except Exception:
                    pass
                    
            if updated:
                _m.memory_store.update_agent_metadata(agent_id, metadata)
        except Exception:
            pass

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
        receiver_key = self._role_to_key.get(id(routed.receiver))
        mentions = self._detect_mentions(message)
        if receiver_key and receiver_key in mentions:
            prompt = f"[Note: you were mentioned in this message]\n{prompt}"
        raw_prompt = prepend_verified_tool_results(
            prompt,
            system_events,
        )
        effective_org_lines = _m.org_chat_context_lines if self._effective_clock_state() == "on" else 0
        org_context = format_org_chat_context(self.transcript, effective_org_lines)
        model_prompt = org_context + raw_prompt if org_context else raw_prompt
        self.prepare_agent_runtime(routed.receiver)
        
        # Track previous token totals to measure current step's usage
        prev_p = self.prompt_tokens
        prev_c = self.completion_tokens
        prev_t = self.total_tokens

        response = routed.receiver.interact(sender.name, model_prompt)
        self._decay_agent_adjustments(routed.receiver.name)
        response = "" if response is None else response
        model_thinking = getattr(routed.receiver, "runtime_thinking", None)
        native_tool_events = list(getattr(routed.receiver, "runtime_tool_results", []))
        if native_tool_events:
            system_events.extend(native_tool_events)
            deliver_verified_tool_results(routed.receiver, native_tool_events)
        response, response_events = self.process_agent_action(response, sender=routed.receiver)
        if response_events:
            system_events.extend(response_events)
            deliver_verified_tool_results(routed.receiver, response_events)
            self.sync_scheduler_participants()
        if model_thinking and routed.receiver.name != "CEO":
            print(colored(f"[{routed.receiver.name} \U0001f914 {len(model_thinking)} chars]", "dark_grey"))
        if routed.receiver.name != "CEO" and response != "":
            print(colored(f"{routed.receiver.name} responds: {response}", "cyan"))
            get_logger().write_event("agent_response", agent=routed.receiver.name, content=response)
        
        # Display current turn token usage and session running totals
        turn_p = self.prompt_tokens - prev_p
        turn_c = self.completion_tokens - prev_c
        turn_t = self.total_tokens - prev_t
        if turn_t > 0:
            token_str = f"[Tokens: {turn_p} in, {turn_c} out, {turn_t} total | running: {self.prompt_tokens} in, {self.completion_tokens} out, {self.total_tokens} total]"
            import os
            import sys
            if os.environ.get("ROBITS_LOG_TOKENS") == "1" or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
                print(colored(token_str, "dark_grey"))
        self.record_turn(
            sender=sender.name,
            receiver=routed.receiver.name,
            prompt=raw_prompt,
            response=response,
            directed=routed.directed,
            system_events=system_events,
            thinking=model_thinking,
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

        _loop_window: list[str] = []
        _loop_threshold = _m.loop_detect_threshold
        try:
            while effective_max_turns is None or self.turns_completed < effective_max_turns:
                last_response = self.step(last_response)
                _normalized = " ".join((last_response or "").lower().split())
                # Only count idle turns — if this turn had tool calls or directed
                # routing, useful work is happening, so reset the window.
                _last_entry = self.transcript[-1] if self.transcript else None
                _turn_was_active = _last_entry is not None and (
                    bool(getattr(_last_entry, "system_events", None))
                    or getattr(_last_entry, "directed", False)
                )
                if _turn_was_active:
                    _loop_window.clear()
                else:
                    _loop_window.append(_normalized)
                    if len(_loop_window) > _loop_threshold:
                        _loop_window.pop(0)
                if (
                    len(_loop_window) >= _loop_threshold
                    and len(set(_loop_window)) == 1
                    and _normalized
                ):
                    print(colored(
                        f"\n[Loop detector] {_loop_threshold} identical consecutive responses "
                        f"detected with no tool calls or directed routing — possible greeting loop. Halting.",
                        "yellow",
                    ))
                    get_logger().write_event("loop_detected", consecutive=_loop_threshold)
                    self.event_stream.emit(
                        "session.loop_detected",
                        self.run_id,
                        {"response": last_response, "consecutive": _loop_threshold},
                    )
                    break
        except SystemExit:
            print(colored("\nSession ended by CEO.", "yellow"))
            get_logger().write_event("session_ended")

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
        self._embed_pending()
        return self

    def close(self):
        """Unsubscribe from the event stream to prevent memory leaks."""
        if hasattr(self, "_event_callback") and self._event_callback:
            try:
                self.event_stream.unsubscribe(self._event_callback)
            except Exception:
                pass
            self._event_callback = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

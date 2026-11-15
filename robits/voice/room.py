"""A call's conversation: the caller's words in, persona replies out (no audio, no Pipecat).

Each call is a robits ``Session`` holding only the line's personas, so replies use the
personas' prompts, memories, briefings and tools, and the call is recorded in memory like
any other session. Who answers:

- a one-persona line: always that persona;
- a group line: the persona the caller names ("Alex, ...", "@alex ..."), else the only one
  mentioned, else whoever spoke last (a call is a conversation, not a round-robin), and at
  the start the line's first persona. On a line with ``handoffs`` (off by default: small
  models pass questions back and forth instead of answering), a persona that hands the
  question on ("Sam, what do you think?") brings that persona in, at most that many times
  per caller turn.

A line's ``heard_as`` words (what speech recognition writes for a name: "Psalm" for "Sam")
are replaced by the persona's first name before any of this, so the persona hears its name too.

``say()`` is synchronous (model calls block); run it off the event loop.
"""
import re
import threading
from dataclasses import dataclass

from robits.voice import speech

@dataclass(frozen=True)
class Reply:
    persona: str      # username
    name: str         # what to call them ("Alex Chen")
    text: str         # speakable
    voice: str | None


class Caller:
    """The caller as a session sender: no tool grants."""

    capabilities = frozenset()
    allowed_tools = frozenset()

    def __init__(self, name):
        self.name = name


class Room:
    def __init__(self, line, caller="caller", session=None, session_factory=None):
        self.line = line
        self.caller = Caller(caller)
        self._lock = threading.Lock()
        self.session = session or (session_factory or _new_session)(line, self.caller.name)
        self.last = None
        self.closed = False
        # The call note is the same on every turn of the call, so it goes in each persona's system
        # message (where a model server's prompt cache keeps it) rather than ahead of every message.
        for key in line.personas:
            self.session.participants[key].situation_note = self._note(key)

    # --- who answers -------------------------------------------------------------------------

    def _names(self, key):
        role = self.session.participants[key]
        return getattr(role, "full_name", None) or key

    def _addressed(self, text):
        """(persona key, remaining words) when the words begin by naming a persona on the line."""
        key, rest = self.session._parse_directed(text)
        if key in self.line.personas:
            return key, rest
        return None, text

    def pick(self, text):
        """(persona key, words for them) for the caller's words."""
        if not self.line.group:
            return self.line.personas[0], text
        key, rest = self._addressed(text)
        if key:
            return key, rest
        mentioned = [k for k in self._mentioned(text) if k in self.line.personas]
        if len(mentioned) == 1:
            return mentioned[0], text
        return (self.last or self.line.personas[0]), text

    def _as_heard(self, text):
        """The words with the line's heard_as mishearings replaced by the persona's first name."""
        for heard, key in self.line.heard_as:
            role = self.session.participants[key]
            name = getattr(role, "first_name", None) or (getattr(role, "full_name", None) or key).split()[0]
            pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in heard.split()) + r"(?!\w)"
            text = re.sub(pattern, lambda _: name, text, flags=re.I)
        return text

    def _mentioned(self, text):
        """Persona keys the words mention: robits' own detection (from memory), else a capitalised
        first or full name, as speech recognition writes names."""
        found = self.session._detect_mentions(text)
        if found:
            return found
        found = set()
        for key in self.line.personas:
            role = self.session.participants[key]
            for form in (getattr(role, "full_name", None), getattr(role, "first_name", None), key):
                if form and re.search(r"\b" + re.escape(form[0].upper() + form[1:]) + r"\b", text):
                    found.add(key)
                    break
        return found

    def _note(self, key):
        others = [self._names(k) for k in self.line.personas if k != key] if self.line.group else ()
        return speech.call_note(self.caller.name if self.caller.name != "caller" else "", others,
                                handoffs=bool(self.line.handoffs))

    # --- turns -------------------------------------------------------------------------------

    def _turn(self, key, words, sender):
        response = self.session.step(words, sender=sender, receiver_name=key)
        self.last = key
        return Reply(key, self._names(key), self._unaddressed(key, speech.speakable(response or "")),
                     self.line.voices.get(key))

    def _opening_name(self, text):
        """(persona key, the rest) when the words open with a persona's name and a comma ("Alex,",
        "Alex Chen,", "@alex_chen"), else (None, text)."""
        to, rest = self.session._parse_directed(text)
        if to is not None:
            return to, rest
        for k in self.line.personas:
            role = self.session.participants[k]
            for form in (getattr(role, "full_name", None), getattr(role, "first_name", None), k):
                m = form and re.match(r"\s*" + re.escape(form) + r"\s*[,:]\s*", text, re.I)
                if m:
                    return k, text[m.end():]
        return None, text

    def _unaddressed(self, key, text):
        """Small models open replies with a name, even their own ("Alex Chen, I'm on the vote
        story"): drop it unless it puts a question to another persona on a line with hand-offs."""
        to, rest = self._opening_name(text)
        if to is None or not rest:
            return text
        if self.line.handoffs and to in self.line.personas and to != key and "?" in rest:
            return text
        return rest[:1].upper() + rest[1:]

    def say(self, text):
        """The replies to the caller's words, in order (empty words get none)."""
        text = (text or "").strip()
        if not text or self.closed:
            return []
        with self._lock:
            key, words = self.pick(self._as_heard(text))
            replies = [self._turn(key, words, self.caller)]
            asked_by = None
            for _ in range(self.line.handoffs):
                said = replies[-1]
                to, rest = self._opening_name(said.text)
                # A hand-off is a question put to another persona, and never straight back to the one
                # who just asked (small models answer "Alex, ..." to Alex and would bounce forever).
                if not to or to in (said.persona, asked_by) or "?" not in rest:
                    break
                speaker = self.session.participants[said.persona]
                replies.append(self._turn(to, f"{said.name} asked you, on the call: {rest}", speaker))
                asked_by = said.persona
            return [r for r in replies if r.text]

    def close(self):
        self.closed = True
        for key in self.line.personas:
            self.session.participants[key].__dict__.pop("situation_note", None)
        try:
            self.session.close()
        except Exception:
            pass


def _new_session(line, caller_name):
    """A robits Session holding just the line's personas (robits must be configured: import main)."""
    from robits.core.config import _config
    from robits.core.roles import build_employee_dict, load_tools
    from robits.core.session import Session
    personas = {k: v for k, v in (_config.persona_entries or {}).items() if k in line.personas}
    participants = build_employee_dict(personas, headless=True)
    missing = [k for k in line.personas if k not in participants]
    if missing:
        raise RuntimeError(f"line {line.name}: personas not available: {', '.join(missing)}")
    session = Session(participants=participants, headless=True)
    if getattr(_config, "tools_file", None):
        load_tools(session.system, _config.tools_file)
    store = getattr(_config, "memory_store", None)
    if store is not None:
        try:
            store.upsert_agent(caller_name, "External", display_name=caller_name)
        except Exception:
            pass
    return session

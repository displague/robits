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

``say()`` is synchronous (model calls block); run it off the event loop. With ``on_speech`` it
also hands over each reply sentence as soon as the model has written it (see ``Speaker``), so a
caller hears the first sentence while the rest is still being written.
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
    spoken: bool = False   # already handed to on_speech, sentence by sentence


TOOL_CUE = "Let me check."   # said once per turn when a tool runs before anything else was said

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "sr", "jr", "gov", "sen", "rep", "gen", "col", "lt", "sgt",
                  "capt", "prof", "rev", "hon", "vs", "etc", "inc", "co", "corp", "ltd", "no", "vol", "approx",
                  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s+")


def split_sentences(text):
    """(complete sentences, the unfinished rest) of streamed text. A sentence ends at ., ! or ?
    followed by a space, except after a short abbreviation ("Gov. Ruiz") or a single letter ("U.S.")."""
    done, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        before = text[start:m.start()].rsplit(None, 1)
        word = before[-1].rstrip(".").lower() if before else ""
        if text[m.start()] == "." and (word in _ABBREVIATIONS or len(word) == 1 or "." in word):
            continue
        sentence = text[start:m.end()].strip()
        if sentence:
            done.append(sentence)
        start = m.end()
    return done, text[start:]


class Speaker:
    """A persona's streamed reply, cut into speakable sentences for ``on_speech``.

    Called by the model provider (``role.runtime_text_stream``) with ``("text", delta)`` as the
    reply is written and ``("tool", names)`` before tools run. Replies that start like a
    structured action, markup or code (``{``, ``[``, ``<``, a backtick) are never streamed: they
    are handled whole after the turn, as before. The first sentence loses an opening name as
    ``Room._unaddressed`` would, and the spoken total stays within ``speech.MAX_CHARS``.
    """

    def __init__(self, room, key, on_speech):
        self.room, self.key, self.on_speech = room, key, on_speech
        self.buffer = ""
        self.mode = None          # None: undecided, True: streaming, False: not streamed
        self.said = 0             # characters spoken
        self.cued = False
        self.cut = False          # an inline <tool_call> began: the rest of this round is not speech
        self.streamed = False     # some reply text was spoken (the cue doesn't count)

    def __call__(self, kind, value):
        if kind == "text":
            self._text(value)
        elif kind == "tool":
            self._flush()
            self.mode, self.buffer, self.cut = None, "", False   # the next round is a new reply
            if not self.cued and not self.said and self.mode is not False:
                self.cued = True
                self._say(TOOL_CUE)

    def _text(self, delta):
        if self.mode is False or self.cut:
            return
        self.buffer += delta
        marker = self.buffer.find("<tool_call")
        if marker >= 0:
            self.buffer, self.cut = self.buffer[:marker], True
        if self.mode is None:
            head = self.buffer.lstrip()
            if not head:
                return
            self.mode = head[0] not in "{[<`"
            if not self.mode:
                return
        sentences, self.buffer = split_sentences(self.buffer)
        for sentence in sentences:
            self._speak(sentence)

    def _flush(self):
        if self.mode and self.buffer.strip():
            rest, self.buffer = self.buffer, ""
            self._speak(rest)

    def _speak(self, sentence):
        text = speech.speakable(sentence)
        if not self.streamed:
            text = self.room._unaddressed(self.key, text)
        if not text or self.said + len(text) > speech.MAX_CHARS:
            return
        self.streamed = True
        self._say(text)

    def _say(self, text):
        self.said += len(text)
        self.on_speech(Reply(self.key, self.room._names(self.key), text,
                             self.room.line.voices.get(self.key), spoken=True))

    def finish(self):
        """Speak what is left; True when the reply's text was spoken here."""
        self._flush()
        return self.streamed


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

    def _turn(self, key, words, sender, on_speech=None):
        role = self.session.participants[key]
        speaker = Speaker(self, key, on_speech) if on_speech else None
        if speaker:
            role.runtime_text_stream = speaker
        try:
            response = self.session.step(words, sender=sender, receiver_name=key)
        finally:
            if speaker:
                role.__dict__.pop("runtime_text_stream", None)
        self.last = key
        spoken = bool(speaker and speaker.finish())
        return Reply(key, self._names(key), self._unaddressed(key, speech.speakable(response or "")),
                     self.line.voices.get(key), spoken)

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

    def say(self, text, on_speech=None):
        """The replies to the caller's words, in order (empty words get none).

        ``on_speech(reply)``, when given, receives each sentence as soon as it is written (from
        this thread); replies whose text went that way come back with ``spoken=True``."""
        text = (text or "").strip()
        if not text or self.closed:
            return []
        with self._lock:
            key, words = self.pick(self._as_heard(text))
            replies = [self._turn(key, words, self.caller, on_speech)]
            asked_by = None
            for _ in range(self.line.handoffs):
                said = replies[-1]
                to, rest = self._opening_name(said.text)
                # A hand-off is a question put to another persona, and never straight back to the one
                # who just asked (small models answer "Alex, ..." to Alex and would bounce forever).
                if not to or to in (said.persona, asked_by) or "?" not in rest:
                    break
                speaker = self.session.participants[said.persona]
                replies.append(self._turn(to, f"{said.name} asked you, on the call: {rest}", speaker, on_speech))
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
